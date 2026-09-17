"""
Tests for the standalone audio transcription endpoint (Studio's Podcast Page
Generator tab, Ben's ask 2026-09-17): POST /transcribe-audio and
GET /transcribe-audio/<job_id>/status.

This is the first Flask-route test file for degas-clips app.py -- follows
the same DB_PATH-env-before-import pattern used across Studio's tests.

Run: python test_podcast_audio.py
"""

import io
import os
import sys
import time
import json

sys.path.insert(0, os.path.dirname(__file__))
os.environ["DB_PATH"] = "/tmp/degas_test/test_podcast_audio.db"
os.makedirs(os.path.dirname(os.environ["DB_PATH"]), exist_ok=True)
if os.path.exists(os.environ["DB_PATH"]):
    os.remove(os.environ["DB_PATH"])
os.environ.setdefault("UPLOAD_FOLDER", "/tmp/degas_test/uploads")
os.environ.setdefault("OUTPUT_FOLDER", "/tmp/degas_test/outputs")

import app as app_module
import database
import transcription

database.init_db()

results = {"pass": 0, "fail": 0}


def check(name, cond, detail=""):
    if cond:
        results["pass"] += 1
        print(f"PASS  {name}")
    else:
        results["fail"] += 1
        print(f"FAIL  {name}  {detail}")


client = app_module.app.test_client()
with client.session_transaction() as sess:
    sess["logged_in"] = True


def fake_transcribe_success(audio_path, words_path, segments_path, original_path=None):
    """Stands in for transcription.transcribe -- writes the JSON shape the
    route expects without touching Whisper/ffmpeg."""
    segments = [
        {"start": 0.0, "end": 2.0, "text": "Hello and welcome to the show."},
        {"start": 2.0, "end": 4.0, "text": "Today we're talking about podcasts."},
    ]
    words = [{"word": "Hello", "start": 0.0, "end": 0.5, "confidence": 0.99}]
    with open(segments_path, "w", encoding="utf-8") as f:
        json.dump(segments, f)
    with open(words_path, "w", encoding="utf-8") as f:
        json.dump(words, f)


def wait_for_status(job_id, timeout=5):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        resp = client.get(f"/transcribe-audio/{job_id}/status")
        last = resp.get_json()
        if last["status"] != "transcribing":
            return last
        time.sleep(0.05)
    return last


def test_transcribe_audio_success_flow():
    transcription.transcribe = fake_transcribe_success

    data = {"audio": (io.BytesIO(b"fake mp3 bytes"), "episode.mp3")}
    resp = client.post("/transcribe-audio", data=data, content_type="multipart/form-data")
    check("transcribe_audio: 200 OK", resp.status_code == 200, resp.status_code)
    body = resp.get_json()
    check("transcribe_audio: returns job_id", bool(body.get("job_id")), body)
    check("transcribe_audio: initial status is transcribing", body.get("status") == "transcribing", body)

    job_id = body["job_id"]
    final = wait_for_status(job_id)
    check("status: reaches done", final["status"] == "done", final)
    check("status: transcript joins segment texts",
          final["transcript"] == "Hello and welcome to the show. Today we're talking about podcasts.",
          final)

    # Intermediate audio/words/segments files must be cleaned up -- they
    # exist only to answer this one job, unlike clip media.
    audio_dir = os.path.join(os.environ["UPLOAD_FOLDER"], "_podcast_audio")
    leftover = os.listdir(audio_dir) if os.path.exists(audio_dir) else []
    check("cleanup: no leftover files for this job", not any(job_id in f for f in leftover), leftover)


def test_transcribe_audio_rejects_missing_file():
    resp = client.post("/transcribe-audio", data={}, content_type="multipart/form-data")
    check("transcribe_audio: 400 when no audio file given", resp.status_code == 400)


def test_transcribe_audio_rejects_bad_extension():
    data = {"audio": (io.BytesIO(b"not audio"), "episode.mp4")}
    resp = client.post("/transcribe-audio", data=data, content_type="multipart/form-data")
    check("transcribe_audio: 400 for unsupported extension", resp.status_code == 400)
    check("transcribe_audio: error message names the bad extension", ".mp4" in resp.get_json().get("error", ""))


def test_transcribe_audio_status_unknown_job():
    resp = client.get("/transcribe-audio/does-not-exist/status")
    check("status: 404 for unknown job_id", resp.status_code == 404)


def test_transcribe_audio_records_error_and_cleans_up_on_failure():
    def fake_transcribe_fail(audio_path, words_path, segments_path, original_path=None):
        raise RuntimeError("boom")
    transcription.transcribe = fake_transcribe_fail

    data = {"audio": (io.BytesIO(b"fake mp3 bytes"), "episode2.mp3")}
    resp = client.post("/transcribe-audio", data=data, content_type="multipart/form-data")
    job_id = resp.get_json()["job_id"]

    final = wait_for_status(job_id)
    check("status: reaches error", final["status"] == "error", final)
    check("status: error message surfaced", final["error"] == "boom", final)

    audio_dir = os.path.join(os.environ["UPLOAD_FOLDER"], "_podcast_audio")
    leftover = os.listdir(audio_dir) if os.path.exists(audio_dir) else []
    check("cleanup: no leftover files after a failed job", not any(job_id in f for f in leftover), leftover)

    transcription.transcribe = fake_transcribe_success


def test_transcribe_audio_chunk_partial_and_completion():
    """This is the route Studio's podcast tab actually uses (not the
    single-shot /transcribe-audio above) -- a real episode MP3 is well past
    Studio's own nginx's 10MB cap, so the browser uploads in chunks."""
    transcription.transcribe = fake_transcribe_success

    file_uid = "chunk-test-uid"
    chunk_a = b"a" * 10
    chunk_b = b"b" * 10

    resp = client.post("/transcribe-audio/chunk", data={
        "file_uid": file_uid, "chunk_index": "0", "total_chunks": "2", "filename": "episode.mp3",
        "data": (io.BytesIO(chunk_a), "episode.mp3"),
    }, content_type="multipart/form-data")
    body = resp.get_json()
    check("chunk: 200 on first of two chunks", resp.status_code == 200, resp.status_code)
    check("chunk: reports chunk_received, not complete", body.get("status") == "chunk_received", body)
    check("chunk: no job_id yet", "job_id" not in body, body)

    resp = client.post("/transcribe-audio/chunk", data={
        "file_uid": file_uid, "chunk_index": "1", "total_chunks": "2", "filename": "episode.mp3",
        "data": (io.BytesIO(chunk_b), "episode.mp3"),
    }, content_type="multipart/form-data")
    body = resp.get_json()
    check("chunk: 200 on final chunk", resp.status_code == 200, resp.status_code)
    check("chunk: reports complete", body.get("status") == "complete", body)
    check("chunk: returns a job_id", bool(body.get("job_id")), body)

    final = wait_for_status(body["job_id"])
    check("chunk: job reaches done", final["status"] == "done", final)

    # The chunks/<file_uid> temp dir should be cleaned up after reassembly.
    chunks_dir = os.path.join(os.environ["UPLOAD_FOLDER"], "_podcast_audio", "chunks", file_uid)
    check("chunk: temp chunk dir cleaned up", not os.path.exists(chunks_dir))


def test_transcribe_audio_chunk_rejects_bad_extension():
    resp = client.post("/transcribe-audio/chunk", data={
        "file_uid": "bad-ext-uid", "chunk_index": "0", "total_chunks": "1", "filename": "episode.mp4",
        "data": (io.BytesIO(b"bytes"), "episode.mp4"),
    }, content_type="multipart/form-data")
    check("chunk: 400 for unsupported extension", resp.status_code == 400, resp.status_code)


if __name__ == "__main__":
    test_transcribe_audio_success_flow()
    test_transcribe_audio_rejects_missing_file()
    test_transcribe_audio_rejects_bad_extension()
    test_transcribe_audio_status_unknown_job()
    test_transcribe_audio_records_error_and_cleans_up_on_failure()
    test_transcribe_audio_chunk_partial_and_completion()
    test_transcribe_audio_chunk_rejects_bad_extension()

    print()
    print(f"TOTAL: {results['pass']} passed, {results['fail']} failed")
    if results["fail"]:
        sys.exit(1)
