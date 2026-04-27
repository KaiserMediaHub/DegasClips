import os
import json
import threading
import uuid

from flask import (
    Flask, render_template, request, redirect,
    url_for, session, jsonify, send_file
)
from werkzeug.utils import secure_filename

from database import init_db, get_db
import transcription
import captions
from sync_logic import update_words_from_segments

# ── Config ────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key    = os.environ.get("SECRET_KEY", "change-me-in-production")
APP_PASSWORD      = os.environ.get("APP_PASSWORD", "degas2024")
UPLOAD_FOLDER     = os.environ.get("UPLOAD_FOLDER", "uploads")
OUTPUT_FOLDER     = os.environ.get("OUTPUT_FOLDER", "outputs")
MAX_CONTENT_MB    = int(os.environ.get("MAX_CONTENT_MB", "2048"))

app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_MB * 1024 * 1024

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


def allowed(filename):
    return os.path.splitext(filename.lower())[1] in ALLOWED_EXTENSIONS


# ── Auth ──────────────────────────────────────────────────────────────────────
@app.before_request
def require_login():
    public = {"login", "static"}
    if request.endpoint in public:
        return
    if not session.get("logged_in"):
        return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        if request.form.get("password") == APP_PASSWORD:
            session["logged_in"] = True
            return redirect(url_for("projects"))
        error = "Incorrect password — try again."
    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ── Projects ──────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return redirect(url_for("projects"))


@app.route("/projects")
def projects():
    db   = get_db()
    rows = db.execute(
        "SELECT * FROM projects ORDER BY created_at DESC"
    ).fetchall()

    project_list = []
    for p in rows:
        total    = db.execute(
            "SELECT COUNT(*) FROM clips WHERE project_id = ?", (p["id"],)
        ).fetchone()[0]
        exported = db.execute(
            "SELECT COUNT(*) FROM clips WHERE project_id = ? AND status = 'exported'",
            (p["id"],)
        ).fetchone()[0]
        project_list.append({
            "id":           p["id"],
            "name":         p["name"],
            "assigned_to":  p["assigned_to"],
            "created_at":   p["created_at"],
            "clip_count":   total,
            "exported_count": exported,
        })
    db.close()
    return render_template("projects.html", projects=project_list)


@app.route("/projects/new", methods=["POST"])
def new_project():
    name        = request.form.get("name", "").strip()
    assigned_to = request.form.get("assigned_to", "").strip()
    if name:
        db = get_db()
        db.execute(
            "INSERT INTO projects (name, assigned_to) VALUES (?, ?)",
            (name, assigned_to)
        )
        db.commit()
        db.close()
    return redirect(url_for("projects"))


@app.route("/projects/<int:project_id>/delete", methods=["POST"])
def delete_project(project_id):
    db = get_db()
    db.execute("DELETE FROM projects WHERE id = ?", (project_id,))
    db.commit()
    db.close()
    return redirect(url_for("projects"))


# ── Project detail ────────────────────────────────────────────────────────────
@app.route("/projects/<int:project_id>")
def project(project_id):
    db   = get_db()
    proj = db.execute(
        "SELECT * FROM projects WHERE id = ?", (project_id,)
    ).fetchone()
    if not proj:
        db.close()
        return redirect(url_for("projects"))
    clips = db.execute(
        "SELECT * FROM clips WHERE project_id = ? ORDER BY created_at",
        (project_id,)
    ).fetchall()
    db.close()
    return render_template(
        "project.html",
        project=proj,
        clips=clips,
        styles=captions.STYLES,
    )


# ── Upload (chunked) ──────────────────────────────────────────────────────────
CHUNK_SIZE = 50 * 1024 * 1024  # 50 MB per chunk

@app.route("/projects/<int:project_id>/upload/chunk", methods=["POST"])
def upload_chunk(project_id):
    """
    Receives one chunk at a time.
    Form fields:
      file_uid      — client-generated UUID for this file
      chunk_index   — 0-based integer
      total_chunks  — total number of chunks for this file
      filename      — original filename
      data          — the chunk bytes (file field)
    """
    file_uid     = request.form.get("file_uid")
    chunk_index  = int(request.form.get("chunk_index", 0))
    total_chunks = int(request.form.get("total_chunks", 1))
    original     = request.form.get("filename", "video.mp4")
    chunk_data   = request.files.get("data")

    if not file_uid or not chunk_data:
        return jsonify({"error": "missing fields"}), 400

    if not allowed(original):
        return jsonify({"error": "file type not allowed"}), 400

    # Save chunk to temp dir
    clip_dir  = os.path.join(UPLOAD_FOLDER, str(project_id))
    temp_dir  = os.path.join(clip_dir, "chunks", file_uid)
    os.makedirs(temp_dir, exist_ok=True)

    chunk_path = os.path.join(temp_dir, f"{chunk_index:05d}")
    chunk_data.save(chunk_path)

    # Check if all chunks have arrived
    saved_chunks = len(os.listdir(temp_dir))
    if saved_chunks < total_chunks:
        return jsonify({"status": "chunk_received", "chunks": saved_chunks, "total": total_chunks})

    # All chunks received — reassemble
    safe_name = secure_filename(original)
    uid       = uuid.uuid4().hex[:8]
    save_name = f"{uid}_{safe_name}"
    final_path = os.path.join(clip_dir, save_name)

    with open(final_path, "wb") as out:
        for i in range(total_chunks):
            part = os.path.join(temp_dir, f"{i:05d}")
            with open(part, "rb") as pf:
                out.write(pf.read())

    # Clean up temp chunks
    import shutil
    shutil.rmtree(temp_dir)

    # Register in DB
    db = get_db()
    db.execute(
        """INSERT INTO clips
           (project_id, filename, original_filename, status)
           VALUES (?, ?, ?, 'uploaded')""",
        (project_id, save_name, original)
    )
    db.commit()
    db.close()

    return jsonify({"status": "complete", "filename": original})


# ── Transcription ─────────────────────────────────────────────────────────────
def _run_transcribe(project_id, clip_id, video_path, words_path, segments_path):
    db = get_db()
    try:
        transcription.transcribe(video_path, words_path, segments_path)
        db.execute(
            "UPDATE clips SET status = 'transcribed' WHERE id = ?", (clip_id,)
        )
    except Exception as e:
        db.execute(
            "UPDATE clips SET status = 'error', error_message = ? WHERE id = ?",
            (str(e), clip_id)
        )
    finally:
        db.commit()
        db.close()


@app.route("/projects/<int:project_id>/clips/<int:clip_id>/transcribe", methods=["POST"])
def transcribe_clip(project_id, clip_id):
    db   = get_db()
    clip = db.execute(
        "SELECT * FROM clips WHERE id = ? AND project_id = ?", (clip_id, project_id)
    ).fetchone()
    if not clip:
        db.close()
        return jsonify({"error": "not found"}), 404

    db.execute("UPDATE clips SET status = 'transcribing' WHERE id = ?", (clip_id,))
    db.commit()
    db.close()

    clip_dir      = os.path.join(UPLOAD_FOLDER, str(project_id))
    video_path    = os.path.join(clip_dir, clip["filename"])
    words_path    = os.path.join(clip_dir, f"{clip_id}.words.json")
    segments_path = os.path.join(clip_dir, f"{clip_id}.segments.json")

    threading.Thread(
        target=_run_transcribe,
        args=(project_id, clip_id, video_path, words_path, segments_path),
        daemon=True,
    ).start()

    return jsonify({"status": "transcribing"})


@app.route("/projects/<int:project_id>/transcribe-all", methods=["POST"])
def transcribe_all(project_id):
    db    = get_db()
    clips = db.execute(
        "SELECT * FROM clips WHERE project_id = ? AND status = 'uploaded'",
        (project_id,)
    ).fetchall()

    for clip in clips:
        db.execute(
            "UPDATE clips SET status = 'transcribing' WHERE id = ?", (clip["id"],)
        )
    db.commit()
    db.close()

    def run_all():
        for clip in clips:
            clip_dir      = os.path.join(UPLOAD_FOLDER, str(project_id))
            video_path    = os.path.join(clip_dir, clip["filename"])
            words_path    = os.path.join(clip_dir, f"{clip['id']}.words.json")
            segments_path = os.path.join(clip_dir, f"{clip['id']}.segments.json")
            _run_transcribe(project_id, clip["id"], video_path, words_path, segments_path)

    threading.Thread(target=run_all, daemon=True).start()
    return redirect(url_for("project", project_id=project_id))


# ── Status polling ────────────────────────────────────────────────────────────
@app.route("/projects/<int:project_id>/clips/<int:clip_id>/status")
def clip_status(project_id, clip_id):
    db   = get_db()
    clip = db.execute(
        "SELECT status, error_message FROM clips WHERE id = ? AND project_id = ?",
        (clip_id, project_id)
    ).fetchone()
    db.close()
    if not clip:
        return jsonify({"error": "not found"}), 404
    return jsonify({"status": clip["status"], "error": clip["error_message"]})


# ── Transcript editor ─────────────────────────────────────────────────────────
@app.route("/projects/<int:project_id>/clips/<int:clip_id>/editor")
def editor(project_id, clip_id):
    db   = get_db()
    proj = db.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
    clip = db.execute(
        "SELECT * FROM clips WHERE id = ? AND project_id = ?", (clip_id, project_id)
    ).fetchone()
    db.close()

    if not clip or clip["status"] not in ("transcribed", "exported"):
        return redirect(url_for("project", project_id=project_id))

    segments_path = os.path.join(UPLOAD_FOLDER, str(project_id), f"{clip_id}.segments.json")
    segments = []
    if os.path.exists(segments_path):
        with open(segments_path, "r", encoding="utf-8") as f:
            segments = json.load(f)

    return render_template(
        "editor.html",
        project=proj,
        clip=clip,
        segments=segments,
    )


@app.route("/projects/<int:project_id>/clips/<int:clip_id>/save", methods=["POST"])
def save_transcript(project_id, clip_id):
    data     = request.get_json()
    segments = data.get("segments", [])

    clip_dir      = os.path.join(UPLOAD_FOLDER, str(project_id))
    words_path    = os.path.join(clip_dir, f"{clip_id}.words.json")
    segments_path = os.path.join(clip_dir, f"{clip_id}.segments.json")

    # Persist updated segment text
    if os.path.exists(segments_path):
        with open(segments_path, "r", encoding="utf-8") as f:
            orig_segs = json.load(f)
        for i, seg in enumerate(orig_segs):
            if i < len(segments):
                seg["text"] = segments[i]["text"]
        with open(segments_path, "w", encoding="utf-8") as f:
            json.dump(orig_segs, f, indent=2)

    # Update word-level timestamps
    update_words_from_segments(segments, words_path)

    return jsonify({"status": "saved"})


# ── Video preview ─────────────────────────────────────────────────────────────
@app.route("/projects/<int:project_id>/clips/<int:clip_id>/video")
def serve_video(project_id, clip_id):
    db   = get_db()
    clip = db.execute(
        "SELECT filename FROM clips WHERE id = ? AND project_id = ?",
        (clip_id, project_id)
    ).fetchone()
    db.close()
    if not clip:
        return "", 404
    video_path = os.path.join(UPLOAD_FOLDER, str(project_id), clip["filename"])
    return send_file(video_path)


# ── Export ────────────────────────────────────────────────────────────────────
def _run_export(project_id, clip_id, video_path, words_path, style_key, output_path):
    db = get_db()
    try:
        captions.export_video_with_captions(video_path, words_path, style_key, output_path)
        db.execute(
            "UPDATE clips SET status = 'exported' WHERE id = ?", (clip_id,)
        )
    except Exception as e:
        db.execute(
            "UPDATE clips SET status = 'error', error_message = ? WHERE id = ?",
            (str(e), clip_id)
        )
    finally:
        db.commit()
        db.close()


@app.route("/projects/<int:project_id>/clips/<int:clip_id>/export", methods=["POST"])
def export_clip(project_id, clip_id):
    style_key = request.form.get("style", "1")
    db        = get_db()
    clip      = db.execute(
        "SELECT * FROM clips WHERE id = ? AND project_id = ?", (clip_id, project_id)
    ).fetchone()
    if not clip:
        db.close()
        return jsonify({"error": "not found"}), 404

    db.execute(
        "UPDATE clips SET status = 'exporting', style = ? WHERE id = ?",
        (style_key, clip_id)
    )
    db.commit()
    db.close()

    clip_dir   = os.path.join(UPLOAD_FOLDER, str(project_id))
    output_dir = os.path.join(OUTPUT_FOLDER, str(project_id))
    os.makedirs(output_dir, exist_ok=True)

    video_path  = os.path.join(clip_dir, clip["filename"])
    words_path  = os.path.join(clip_dir, f"{clip_id}.words.json")
    output_path = os.path.join(output_dir, f"{clip_id}_captioned.mp4")

    threading.Thread(
        target=_run_export,
        args=(project_id, clip_id, video_path, words_path, style_key, output_path),
        daemon=True,
    ).start()

    return jsonify({"status": "exporting"})


@app.route("/projects/<int:project_id>/export-all", methods=["POST"])
def export_all(project_id):
    style_key = request.form.get("style", "1")
    db        = get_db()
    clips     = db.execute(
        "SELECT * FROM clips WHERE project_id = ? AND status = 'transcribed'",
        (project_id,)
    ).fetchall()

    for clip in clips:
        db.execute(
            "UPDATE clips SET status = 'exporting', style = ? WHERE id = ?",
            (style_key, clip["id"])
        )
    db.commit()
    db.close()

    def run_all():
        for clip in clips:
            clip_dir   = os.path.join(UPLOAD_FOLDER, str(project_id))
            output_dir = os.path.join(OUTPUT_FOLDER, str(project_id))
            os.makedirs(output_dir, exist_ok=True)
            video_path  = os.path.join(clip_dir, clip["filename"])
            words_path  = os.path.join(clip_dir, f"{clip['id']}.words.json")
            output_path = os.path.join(output_dir, f"{clip['id']}_captioned.mp4")
            _run_export(project_id, clip["id"], video_path, words_path, style_key, output_path)

    threading.Thread(target=run_all, daemon=True).start()
    return redirect(url_for("project", project_id=project_id))


# ── Download ──────────────────────────────────────────────────────────────────
@app.route("/projects/<int:project_id>/clips/<int:clip_id>/download")
def download_clip(project_id, clip_id):
    db   = get_db()
    clip = db.execute(
        "SELECT * FROM clips WHERE id = ? AND project_id = ?", (clip_id, project_id)
    ).fetchone()
    db.close()
    if not clip:
        return redirect(url_for("project", project_id=project_id))

    output_path = os.path.join(OUTPUT_FOLDER, str(project_id), f"{clip_id}_captioned.mp4")
    if not os.path.exists(output_path):
        return redirect(url_for("project", project_id=project_id))

    base = os.path.splitext(clip["original_filename"])[0]
    return send_file(
        output_path,
        as_attachment=True,
        download_name=f"{base}_captioned.mp4",
    )


# ── Startup ───────────────────────────────────────────────────────────────────
# Run on import so gunicorn also initialises the DB and folders
init_db()
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
