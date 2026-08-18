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

# Pass 2 originally re-transcribed low-confidence segments with a full-
# precision (FP32) "medium" Whisper model via the openai-whisper package.
# That OOM-crashed the whole Degas service in production (2026-08-18) --
# this server has 3.7GB RAM total, and FP32 "medium" needs roughly 4-5GB on
# CPU (CPU doesn't support the more memory-efficient FP16 path) on top of
# "small" already being resident.
#
# Fix: switched from openai-whisper to faster-whisper (CTranslate2), which
# supports int8 quantization. An int8 "medium" model needs roughly a
# quarter of FP32's memory footprint -- small enough to actually fit
# alongside an int8 "small" model on this box. This is what makes it safe
# to go back to a genuinely bigger model for pass 2, instead of the
# same-model-plus-beam-search workaround used as the first fix.
#
# Both models also get beam search on pass 2 (bigger model AND a more
# thorough decode), since that's now cheap.
COMPUTE_TYPE = "int8"
PASS2_BEAM_SIZE = 5
PASS2_BEST_OF = 5


def get_model():
    """Lazily loads pass 1's model -- needed for every transcription."""
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        _model = WhisperModel(PASS1_MODEL, device="cpu", compute_type=COMPUTE_TYPE)
    return _model


def get_pass2_model():
    """Lazily loads pass 2's larger model -- only paid for on clips that
    actually have a low-confidence segment worth re-checking. Kept as a
    separate cached global (not reloaded per segment) so repeat low-
    confidence segments across a project don't re-pay the load cost."""
    global _model_pass2
    if _model_pass2 is None:
        from faster_whisper import WhisperModel
        _model_pass2 = WhisperModel(PASS2_MODEL, device="cpu", compute_type=COMPUTE_TYPE)
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


def _words_from_segments(segments):
    """faster-whisper returns Segment/Word objects (attribute access, e.g.
    segment.start, word.probability), not dicts -- this normalizes them into
    the plain-dict shape the rest of Degas (flagging.py, sync_logic.py, the
    editor templates, JSON storage) expects."""
    all_segments = []
    for segment in segments:
        seg_words = []
        for w in (segment.words or []):
            seg_words.append({
                "word":       w.word.strip(),
                "start":      round(w.start, 3),
                "end":        round(w.end, 3),
                "confidence": round(w.probability, 4),
            })
        all_segments.append({
            "start": segment.start,
            "end":   segment.end,
            "text":  segment.text.strip(),
            "words": seg_words,
        })
    return all_segments


def _rerun_segment_with_bigger_model(video_path, segment):
    """Pass 2: re-transcribes one low-confidence segment's audio with the
    larger int8-quantized model, using beam search, and returns fresh word
    dicts with timestamps remapped back onto the full video's timeline.
    Returns None (rather than raising) if anything goes wrong -- ffmpeg
    missing, corrupt audio, etc -- so the caller can just fall back to pass
    1's words for that segment instead of failing the whole transcription
    over one bad spot."""
    temp_path = None
    try:
        temp_path, clip_start = _extract_audio_segment(
            video_path, segment["start"], segment["end"]
        )
        model = get_pass2_model()
        segments, _info = model.transcribe(
            temp_path,
            word_timestamps=True,
            beam_size=PASS2_BEAM_SIZE,
            best_of=PASS2_BEST_OF,
            temperature=0.0,
            condition_on_previous_text=False,
        )
        new_words = []
        for seg in segments:
            for w in (seg.words or []):
                new_words.append({
                    "word":       w.word.strip(),
                    "start":      round(w.start + clip_start, 3),
                    "end":        round(w.end + clip_start, 3),
                    "confidence": round(w.probability, 4),
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

    Pass 1 -- an int8-quantized "small" model (faster-whisper/CTranslate2)
    transcribes the full clip and records a per-word confidence score.

    Pass 2 -- any segment that came out of pass 1 with at least one word
    below LOW_CONFIDENCE_THRESHOLD gets that slice of audio cut out and
    re-transcribed with a larger int8-quantized model ("medium") using beam
    search, and the improved words are spliced back in. This only
    re-processes the shaky parts, not the whole clip, so clips with no
    confidence issues pay no pass-2 cost at all. int8 quantization is what
    makes it safe to use a genuinely bigger model here without repeating the
    OOM crash a full-precision "medium" model caused in production on
    2026-08-18 -- see the COMPUTE_TYPE comment above for the memory math.

    Pass 3 is intentionally not automated here -- it's the manual Caption
    Review step that already exists in the UI (task #8's flagging system).
    Anything still below threshold after pass 2 stays flagged for a human
    to check.

    Saves word-level timestamps to words_path (.words.json)
    and segment-level data to segments_path (.segments.json).
    """
    model = get_model()
    segments, _info = model.transcribe(video_path, word_timestamps=True)
    all_segments = _words_from_segments(segments)

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
