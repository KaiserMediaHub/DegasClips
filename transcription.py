import json
import os
import subprocess
import tempfile

from flagging import LOW_CONFIDENCE_THRESHOLD

_model = None

PASS1_MODEL = "small"
FFMPEG_BIN = os.environ.get("FFMPEG_BIN", "ffmpeg")

# Pass 2 originally re-transcribed low-confidence segments with a larger
# "medium" Whisper model. That OOM-crashed the whole Degas service in
# production (2026-08-18) -- this server has 3.7GB RAM total, and "medium"
# needs roughly 4-5GB in CPU/FP32 mode (CPU doesn't support the more
# memory-efficient FP16 path) on top of the "small" model pass 1 already has
# loaded. Rather than fight that memory ceiling, pass 2 now re-decodes with
# the SAME small model already resident in memory, but using a much more
# thorough (and slower) decoding strategy -- beam search instead of the
# default greedy decode -- which costs extra CPU time per flagged segment,
# not extra memory.
PASS2_BEAM_SIZE = 5
PASS2_BEST_OF = 5


def get_model():
    global _model
    if _model is None:
        import whisper
        _model = whisper.load_model(PASS1_MODEL)
    return _model


def _extract_audio_segment(video_path, start, end, pad=0.3):
    """Cuts out [start-pad, end+pad] of video_path into a temp wav file for
    re-transcription. pad gives the re-decode a little surrounding audio for
    context. Returns (temp_path, clip_start) where clip_start is the
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


def _rerun_segment_with_careful_decode(video_path, segment):
    """Pass 2: re-transcribes one low-confidence segment's audio using the
    SAME already-loaded small model, but with beam search (beam_size=5,
    best_of=5) instead of the default greedy decode -- a much more thorough
    search of possible transcriptions, at the cost of more CPU time on just
    that short segment, not more memory. Returns fresh word dicts with
    timestamps remapped back onto the full video's timeline. Returns None
    (rather than raising) if anything goes wrong -- ffmpeg missing, corrupt
    audio, etc -- so the caller can just fall back to pass 1's words for
    that segment instead of failing the whole transcription over one spot."""
    temp_path = None
    try:
        temp_path, clip_start = _extract_audio_segment(
            video_path, segment["start"], segment["end"]
        )
        model = get_model()
        result = model.transcribe(
            temp_path,
            word_timestamps=True,
            beam_size=PASS2_BEAM_SIZE,
            best_of=PASS2_BEST_OF,
            temperature=0.0,
            condition_on_previous_text=False,
        )
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
    re-decoded with the same model using beam search (slower, more thorough
    than the default greedy decode), and the improved words are spliced
    back in. This only re-processes the shaky parts, not the whole clip, so
    clips with no confidence issues pay no pass-2 cost at all -- and it never
    loads a second model, so it can't repeat the OOM crash a "bigger model"
    version of this caused in production on 2026-08-18.

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
        revised_words = _rerun_segment_with_careful_decode(video_path, segment)
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
