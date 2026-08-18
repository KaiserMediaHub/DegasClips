import json
import os
import subprocess
import tempfile

from flagging import LOW_CONFIDENCE_THRESHOLD

_model = None
_model_pass2 = None

PASS1_MODEL = "small"
PASS2_MODEL = "medium"
FFMPEG_BIN = os.environ.get("FFMPEG_BIN", "ffmpeg")


def get_model():
    global _model
    if _model is None:
        import whisper
        _model = whisper.load_model(PASS1_MODEL)
    return _model


def get_pass2_model():
    """Lazily loads the larger pass-2 model -- only paid for on clips that
    actually have a low-confidence segment worth re-checking."""
    global _model_pass2
    if _model_pass2 is None:
        import whisper
        _model_pass2 = whisper.load_model(PASS2_MODEL)
    return _model_pass2


def _extract_audio_segment(video_path, start, end, pad=0.3):
    """Cuts out [start-pad, end+pad] of video_path into a temp wav file for
    re-transcription. pad gives the bigger model a little surrounding audio
    for context. Returns (temp_path, clip_start) where clip_start is the
    absolute video time the temp clip begins at, needed to remap the
    re-transcribed word timestamps back onto the full video's timeline."""
    clip_start = max(0.0, start - pad)
    duration = (end + pad) - clip_start
    fd, temp_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    cmd = [
        FFMPEG_BIN, "-y",
        "-ss", str(clip_start),
        "-i", video_path,
        "-t", str(duration),
        "-ac", "1", "-ar", "16000",
        temp_path,
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return temp_path, clip_start


def _rerun_segment_with_bigger_model(video_path, segment):
    """Pass 2: re-transcribes one low-confidence segment's audio with a
    larger Whisper model and returns fresh word dicts with timestamps
    remapped back onto the full video's timeline. Returns None (rather than
    raising) if anything goes wrong -- ffmpeg missing, corrupt audio, etc --
    so the caller can just fall back to pass 1's words for that segment
    instead of failing the whole transcription over one bad spot."""
    temp_path = None
    try:
        temp_path, clip_start = _extract_audio_segment(
            video_path, segment["start"], segment["end"]
        )
        model = get_pass2_model()
        result = model.transcribe(temp_path, word_timestamps=True)
        new_words = []
        for seg in result["segments"]:
            for w in seg.get("words", []):
                new_words.append({
                    "word":       w["word"].strip(),
                    "start":      round(w["start"] + clip_start, 3),
                    "end":        round(w["end"] + clip_start, 3),
                    "confidence": round(w.get("probability", 0.0), 4),
                    "revised":    True,
                })
        return new_words or None
    except Exception:
        return None
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def transcribe(video_path, words_path, segments_path, original_path=None):
    """
    Two-pass transcription:

    Pass 1 -- Whisper "small" transcribes the full clip and records a
    per-word confidence score (Whisper's own token probability).

    Pass 2 -- any segment that came out of pass 1 with at least one word
    below LOW_CONFIDENCE_THRESHOLD gets that slice of audio cut out and
    re-transcribed with a larger model ("medium"), and the improved words
    are spliced back in. This only re-processes the shaky parts, not the
    whole clip, so clips with no confidence issues pay no pass-2 cost at all.

    Pass 3 is intentionally not automated here -- it's the manual Caption
    Review step that already exists in the UI (task #8's flagging system).
    Anything still below threshold after pass 2 stays flagged for a human
    to check.

    Saves word-level timestamps to words_path (.words.json)
    and segment-level data to segments_path (.segments.json).
    """
    model = get_model()
    result = model.transcribe(video_path, word_timestamps=True)

    all_segments = []

    for segment in result["segments"]:
        seg_words = []
        for w in segment.get("words", []):
            entry = {
                "word":       w["word"].strip(),
                "start":      w["start"],
                "end":        w["end"],
                "confidence": round(w.get("probability", 0.0), 4),
            }
            seg_words.append(entry)

        all_segments.append({
            "start": segment["start"],
            "end":   segment["end"],
            "text":  segment["text"].strip(),
            "words": seg_words,
        })

    # Pass 2: re-check any segment pass 1 flagged as low-confidence.
    for segment in all_segments:
        if not any(w["confidence"] < LOW_CONFIDENCE_THRESHOLD for w in segment["words"]):
            continue
        revised_words = _rerun_segment_with_bigger_model(video_path, segment)
        if not revised_words:
            continue  # pass 2 failed or found nothing better -- keep pass 1's words, still flagged
        segment["words"] = revised_words
        segment["text"] = " ".join(w["word"] for w in revised_words).strip()

    # Flat word list is derived from the (possibly pass-2-revised) segments,
    # since pass 2 can change word counts/timestamps within a segment.
    all_words = [w for segment in all_segments for w in segment["words"]]

    with open(words_path, "w", encoding="utf-8") as f:
        json.dump(all_words, f, indent=2)

    with open(segments_path, "w", encoding="utf-8") as f:
        json.dump(all_segments, f, indent=2)

    # KMG Studio glossary system (task #8) needs an immutable as-transcribed
    # snapshot to diff against later Caption Review edits -- segments_path
    # itself gets overwritten by /save, so this is written once, here, and
    # never touched again.
    if original_path:
        with open(original_path, "w", encoding="utf-8") as f:
            json.dump(all_segments, f, indent=2)
