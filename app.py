import os
import json
import threading
import time
import uuid

try:
    import fcntl  # POSIX only (Hetzner server). Not available on Windows.
except ImportError:
    fcntl = None

from dotenv import load_dotenv
load_dotenv()

from flask import (
    Flask, render_template, request, redirect,
    url_for, session, jsonify, send_file
)
from werkzeug.utils import secure_filename

from database import init_db, get_db
import transcription
import captions
import flagging
from sync_logic import update_words_from_segments

# ── Config ────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key    = os.environ.get("SECRET_KEY", "change-me-in-production")
APP_PASSWORD      = os.environ.get("APP_PASSWORD", "degas2024")
UPLOAD_FOLDER     = os.environ.get("UPLOAD_FOLDER", "uploads")
OUTPUT_FOLDER     = os.environ.get("OUTPUT_FOLDER", "outputs")
MAX_CONTENT_MB    = int(os.environ.get("MAX_CONTENT_MB", "2048"))

app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_MB * 1024 * 1024

_transcribe_start_times = {}
TRANSCRIBE_TIMEOUT = 600

# Cross-process lock so only one transcription runs system-wide at a time --
# NOT a threading.Semaphore, on purpose. A Semaphore only coordinates threads
# inside one OS process; once gunicorn runs more than one worker process
# (separate processes, separate memory), a Semaphore does nothing to stop two
# workers from each loading a model at once, which is exactly what OOM'd the
# server on 2026-08-18. fcntl.flock works across processes, so it actually
# guards the thing that matters (memory), which is what let us put gunicorn
# back to multiple workers -- 1 worker had been starving Studio's simple
# requests (e.g. GET /projects/<id>) via GIL contention with the transcription
# thread, causing Studio's "Couldn't reach Degas... Read timed out" errors
# (2026-08-19).
_TRANSCRIBE_LOCK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".transcribe.lock")

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
ALLOWED_AUDIO_EXTENSIONS = {".mp3", ".m4a", ".wav"}
PODCAST_AUDIO_FOLDER = os.path.join(UPLOAD_FOLDER, "_podcast_audio")
os.makedirs(PODCAST_AUDIO_FOLDER, exist_ok=True)


@app.template_filter("datestr")
def datestr_filter(v):
    return str(v)[:10] if v else ""


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
    client_id   = request.form.get("client_id", "").strip()
    client_id   = int(client_id) if client_id.isdigit() else None
    if name:
        db = get_db()
        db.execute(
            "INSERT INTO projects (name, assigned_to, client_id) VALUES (?, ?, ?)",
            (name, assigned_to, client_id)
        )
        db.commit()
        proj_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]
        db.close()
        if request.headers.get("Accept") == "application/json":
            return jsonify({"id": proj_id, "name": name, "client_id": client_id})
    return redirect(url_for("projects"))


@app.route("/projects/<int:project_id>/edit", methods=["POST"])
def edit_project(project_id):
    name        = request.form.get("name", "").strip()
    assigned_to = request.form.get("assigned_to", "").strip()
    client_id   = request.form.get("client_id", "").strip()
    if name:
        db = get_db()
        if client_id.isdigit():
            db.execute(
                "UPDATE projects SET name = ?, assigned_to = ?, client_id = ? WHERE id = ?",
                (name, assigned_to, int(client_id), project_id)
            )
        else:
            db.execute(
                "UPDATE projects SET name = ?, assigned_to = ? WHERE id = ?",
                (name, assigned_to, project_id)
            )
        db.commit()
        db.close()
    return redirect(request.referrer or url_for("projects"))


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
    if request.headers.get("Accept") == "application/json":
        return jsonify({
            "id": proj["id"],
            "name": proj["name"],
            "assigned_to": proj["assigned_to"],
            "client_id": proj["client_id"] if "client_id" in proj.keys() else None,
            "clips": [
                {
                    "id": c["id"],
                    "filename": c["filename"],
                    "original_filename": c["original_filename"],
                    "status": c["status"],
                    "error_message": c["error_message"],
                    "style": c["style"],
                }
                for c in clips
            ],
        })
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
def _run_transcribe(project_id, clip_id, video_path, words_path, segments_path, original_path):
    _transcribe_start_times[clip_id] = time.time()
    db = get_db()
    lock_file = open(_TRANSCRIBE_LOCK_PATH, "w") if fcntl else None
    try:
        # Blocks here until any other transcription (this process or another
        # gunicorn worker) finishes. That's intentional -- only one Whisper
        # model may be loaded system-wide at a time. On Windows (local dev,
        # no fcntl) this is a no-op -- fine, since local dev never runs more
        # than one process anyway.
        if fcntl:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
        transcription.transcribe(video_path, words_path, segments_path, original_path)
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
        if fcntl:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
            lock_file.close()


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
    _transcribe_start_times[clip_id] = time.time()

    clip_dir      = os.path.join(UPLOAD_FOLDER, str(project_id))
    video_path    = os.path.join(clip_dir, clip["filename"])
    words_path    = os.path.join(clip_dir, f"{clip_id}.words.json")
    segments_path = os.path.join(clip_dir, f"{clip_id}.segments.json")
    original_path = os.path.join(clip_dir, f"{clip_id}.original.json")

    threading.Thread(
        target=_run_transcribe,
        args=(project_id, clip_id, video_path, words_path, segments_path, original_path),
        daemon=True,
    ).start()

    return jsonify({"status": "transcribing"})


@app.route("/projects/<int:project_id>/transcribe-all", methods=["POST"])
def transcribe_all(project_id):
    db    = get_db()
    clips = db.execute(
        "SELECT * FROM clips WHERE project_id = ? AND status IN ('uploaded', 'error')",
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
            original_path = os.path.join(clip_dir, f"{clip['id']}.original.json")
            _run_transcribe(project_id, clip["id"], video_path, words_path, segments_path, original_path)

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
    if clip["status"] == "transcribing":
        start = _transcribe_start_times.get(clip_id)
        if start and (time.time() - start) > TRANSCRIBE_TIMEOUT:
            db2 = get_db()
            db2.execute("UPDATE clips SET status='error', error_message='Timed out' WHERE id=?", (clip_id,))
            db2.commit()
            db2.close()
            return jsonify({"status": "error", "error": "Timed out"})
        elapsed = int(time.time() - start) if start else 0
        return jsonify({"status": "transcribing", "error": None, "elapsed": elapsed})
    return jsonify({"status": clip["status"], "error": clip["error_message"]})


@app.route("/projects/<int:project_id>/clips/<int:clip_id>/segments")
def clip_segments(project_id, clip_id):
    """KMG Studio task #8: exposes both the original (immutable, as
    transcribed) and current (possibly human-edited) segments for a clip,
    so Studio can diff them to detect glossary candidates."""
    clip_dir = os.path.join(UPLOAD_FOLDER, str(project_id))
    original_path = os.path.join(clip_dir, f"{clip_id}.original.json")
    segments_path = os.path.join(clip_dir, f"{clip_id}.segments.json")

    original = []
    if os.path.exists(original_path):
        with open(original_path, "r", encoding="utf-8") as f:
            original = json.load(f)

    current = []
    if os.path.exists(segments_path):
        with open(segments_path, "r", encoding="utf-8") as f:
            current = json.load(f)

    return jsonify({"original": original, "current": current})


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
    words_path = os.path.join(UPLOAD_FOLDER, str(project_id), f"{clip_id}.words.json")
    segments = []
    if os.path.exists(segments_path):
        with open(segments_path, "r", encoding="utf-8") as f:
            segments = json.load(f)
    words = []
    if os.path.exists(words_path):
        with open(words_path, "r", encoding="utf-8") as f:
            words = json.load(f)
    segments = flagging.annotate_segments_with_flags(segments, words)

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


# ── Standalone audio transcription ──────────────────────────────────────────
# Studio's Podcast Page Generator tab (Ben's ask, 2026-09-17) needs Whisper
# transcription for a raw MP3 episode upload that isn't tied to any Degas
# project/clip. Rather than have Studio load a SECOND Whisper model in its
# own process -- this server has only 3.7GB RAM total, and a full-precision
# model is exactly what OOM'd it the first time (see the accuracy-history
# comment in transcription.py) -- Studio calls this endpoint and Degas
# reuses its already-loaded medium/int8 model plus the same cross-process
# file lock used for clip transcription, so a podcast job can never overlap
# with a clip job and repeat that crash.
def _run_audio_transcribe(job_id, audio_path):
    lock_file = open(_TRANSCRIBE_LOCK_PATH, "w") if fcntl else None
    words_path = audio_path + ".words.json"
    segments_path = audio_path + ".segments.json"
    try:
        if fcntl:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
        transcription.transcribe(audio_path, words_path, segments_path)
        with open(segments_path, encoding="utf-8") as f:
            segments = json.load(f)
        text = " ".join(s["text"] for s in segments if s.get("text")).strip()
        db = get_db()
        db.execute(
            "UPDATE podcast_jobs SET status = 'done', transcript = ? WHERE id = ?",
            (text, job_id)
        )
        db.commit()
        db.close()
    except Exception as e:
        db = get_db()
        db.execute(
            "UPDATE podcast_jobs SET status = 'error', error_message = ? WHERE id = ?",
            (str(e), job_id)
        )
        db.commit()
        db.close()
    finally:
        # This audio file and its intermediate transcription JSON only ever
        # exist to answer this one job -- unlike clip media, nothing else
        # references them afterward, so clean up rather than accumulate
        # podcast episode audio on disk indefinitely.
        for p in (audio_path, words_path, segments_path):
            try:
                os.remove(p)
            except OSError:
                pass
        if fcntl:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
            lock_file.close()


def _start_transcription_job(job_id, audio_path):
    """Shared by both upload paths below: registers a podcast_jobs row and
    fires the background transcription thread."""
    db = get_db()
    db.execute("INSERT INTO podcast_jobs (id, status) VALUES (?, 'transcribing')", (job_id,))
    db.commit()
    db.close()
    threading.Thread(target=_run_audio_transcribe, args=(job_id, audio_path), daemon=True).start()


@app.route("/transcribe-audio", methods=["POST"])
def transcribe_audio():
    """Kicks off a standalone transcription job for a raw audio upload.
    Fire-and-forget, same async pattern as clip transcription: returns a
    job_id immediately, poll /transcribe-audio/<job_id>/status for the
    result. Job state lives in the podcast_jobs table, not an in-process
    dict -- gunicorn runs 2 worker processes, and a POST landing on worker A
    with a status GET landing on worker B need to see the same state (see
    the fcntl lock comment above for the same class of bug, resolved the
    same way: a shared store, not process memory).

    This single-shot route is only safe for small files -- Degas's own
    nginx accepts it, but Studio's nginx caps browser-originated request
    bodies at 10MB (see the CHUNK_SIZE comment on /upload/chunk above), and
    a typical hour-long episode MP3 is well past that. Studio's podcast tab
    uses the chunked variant below instead; this one exists for small
    files and for testing."""
    audio_file = request.files.get("audio")
    if not audio_file or not audio_file.filename:
        return jsonify({"error": "audio file is required"}), 400
    ext = os.path.splitext(audio_file.filename.lower())[1]
    if ext not in ALLOWED_AUDIO_EXTENSIONS:
        return jsonify({"error": f"unsupported audio type '{ext}' -- use mp3, m4a, or wav"}), 400

    job_id = uuid.uuid4().hex
    audio_path = os.path.join(PODCAST_AUDIO_FOLDER, f"{job_id}{ext}")
    audio_file.save(audio_path)

    _start_transcription_job(job_id, audio_path)
    return jsonify({"job_id": job_id, "status": "transcribing"})


@app.route("/transcribe-audio/chunk", methods=["POST"])
def transcribe_audio_chunk():
    """Chunked variant of /transcribe-audio, same reassembly pattern as
    /projects/<id>/upload/chunk above -- exists because a real podcast
    episode MP3 (30-100+ MB for an hour-long show) is well past Studio's
    own nginx's 10MB request-body cap. Studio's browser-facing upload sends
    ~8MB chunks to its own proxy route, which forwards each one here
    one-by-one using the same file_uid/chunk_index/total_chunks fields as
    the video chunk upload. Once the last chunk arrives, reassembles the
    file and starts the transcription job, returning the job_id instead of
    a clip filename (there's no project/clip here to register it against).

    Form fields: file_uid, chunk_index, total_chunks, filename, data (file)."""
    file_uid     = request.form.get("file_uid")
    chunk_index  = int(request.form.get("chunk_index", 0))
    total_chunks = int(request.form.get("total_chunks", 1))
    original     = request.form.get("filename", "episode.mp3")
    chunk_data   = request.files.get("data")

    if not file_uid or not chunk_data:
        return jsonify({"error": "missing fields"}), 400

    ext = os.path.splitext(original.lower())[1]
    if ext not in ALLOWED_AUDIO_EXTENSIONS:
        return jsonify({"error": f"unsupported audio type '{ext}' -- use mp3, m4a, or wav"}), 400

    temp_dir = os.path.join(PODCAST_AUDIO_FOLDER, "chunks", file_uid)
    os.makedirs(temp_dir, exist_ok=True)
    chunk_path = os.path.join(temp_dir, f"{chunk_index:05d}")
    chunk_data.save(chunk_path)

    saved_chunks = len(os.listdir(temp_dir))
    if saved_chunks < total_chunks:
        return jsonify({"status": "chunk_received", "chunks": saved_chunks, "total": total_chunks})

    # All chunks received -- reassemble into one file, named by a fresh
    # job-scoped uuid (not file_uid, so a retried upload with the same
    # file_uid can't collide with a job already in flight).
    job_id = uuid.uuid4().hex
    audio_path = os.path.join(PODCAST_AUDIO_FOLDER, f"{job_id}{ext}")
    with open(audio_path, "wb") as out:
        for i in range(total_chunks):
            part = os.path.join(temp_dir, f"{i:05d}")
            with open(part, "rb") as pf:
                out.write(pf.read())

    import shutil
    shutil.rmtree(temp_dir)

    _start_transcription_job(job_id, audio_path)
    return jsonify({"status": "complete", "job_id": job_id})


@app.route("/transcribe-audio/<job_id>/status")
def transcribe_audio_status(job_id):
    db = get_db()
    job = db.execute(
        "SELECT status, transcript, error_message FROM podcast_jobs WHERE id = ?", (job_id,)
    ).fetchone()
    db.close()
    if not job:
        return jsonify({"error": "unknown job_id"}), 404
    return jsonify({
        "status": job["status"],
        "transcript": job["transcript"],
        "error": job["error_message"],
    })



# -- Bulk transcript editor ---------------------------------------------------
@app.route("/projects/<int:project_id>/editor-all")
def editor_all(project_id):
    db   = get_db()
    proj = db.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
    if not proj:
        db.close()
        return redirect(url_for("projects"))
    clips = db.execute(
        "SELECT * FROM clips WHERE project_id = ? AND status IN ('transcribed', 'exported') ORDER BY created_at",
        (project_id,)
    ).fetchall()
    db.close()
    clips_data = []
    for clip in clips:
        import os, json
        segments_path = os.path.join(UPLOAD_FOLDER, str(project_id), f"{clip['id']}.segments.json")
        words_path = os.path.join(UPLOAD_FOLDER, str(project_id), f"{clip['id']}.words.json")
        segments = []
        words = []
        if os.path.exists(segments_path):
            with open(segments_path, "r", encoding="utf-8") as f2:
                segments = json.load(f2)
        if os.path.exists(words_path):
            with open(words_path, "r", encoding="utf-8") as f2:
                words = json.load(f2)
        segments = flagging.annotate_segments_with_flags(segments, words)
        clips_data.append({"clip": clip, "segments": segments})
    return render_template("editor_all.html", project=proj, clips_data=clips_data)


@app.route("/projects/<int:project_id>/save-all", methods=["POST"])
def save_all_transcripts(project_id):
    import os, json
    data = request.get_json()
    for item in data.get("clips", []):
        clip_id  = item["clip_id"]
        segments = item["segments"]
        clip_dir      = os.path.join(UPLOAD_FOLDER, str(project_id))
        words_path    = os.path.join(clip_dir, f"{clip_id}.words.json")
        segments_path = os.path.join(clip_dir, f"{clip_id}.segments.json")
        if os.path.exists(segments_path):
            with open(segments_path, "r", encoding="utf-8") as f2:
                orig_segs = json.load(f2)
            for i, seg in enumerate(orig_segs):
                if i < len(segments):
                    seg["text"] = segments[i]["text"]
            with open(segments_path, "w", encoding="utf-8") as f2:
                json.dump(orig_segs, f2, indent=2)
        update_words_from_segments(segments, words_path)
    return jsonify({"status": "saved"})

# ── Startup ───────────────────────────────────────────────────────────────────
# Deferred init — runs on first request so the Railway volume is mounted first
_initialized = False

@app.before_request
def ensure_initialized():
    global _initialized
    if not _initialized:
        init_db()
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        _reset_db = get_db()
        _reset_db.execute("UPDATE clips SET status='error', error_message='Server restarted during transcription' WHERE status='transcribing'")
        _reset_db.commit()
        _reset_db.close()
        _initialized = True

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)


# ── KMG Studio: storage cleanup (task #22) ──────────────────────────────────
@app.route("/projects/<int:project_id>/clips/<int:clip_id>/delete-media", methods=["POST"])
def delete_clip_media(project_id, clip_id):
    db = get_db()
    clip = db.execute(
        "SELECT * FROM clips WHERE id = ? AND project_id = ?", (clip_id, project_id)
    ).fetchone()
    if not clip:
        db.close()
        return jsonify({"error": "not found"}), 404

    clip_dir = os.path.join(UPLOAD_FOLDER, str(project_id))
    paths_to_remove = [
        os.path.join(clip_dir, clip["filename"]) if clip["filename"] else None,
        os.path.join(clip_dir, f"{clip_id}.words.json"),
        os.path.join(clip_dir, f"{clip_id}.segments.json"),
        os.path.join(clip_dir, f"{clip_id}.original.json"),
        os.path.join(OUTPUT_FOLDER, str(project_id), f"{clip_id}_captioned.mp4"),
    ]
    for path in paths_to_remove:
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass

    db.execute("UPDATE clips SET status = 'deleted', error_message = NULL WHERE id = ?", (clip_id,))
    db.commit()
    db.close()
    return jsonify({"status": "deleted"})
