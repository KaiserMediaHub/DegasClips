import json
import os
import subprocess
import tempfile

from flagging import LOW_CONFIDENCE_THRESHOLD

_model = None

# Accuracy history, in order:
# 1. Pass 1 was "small", pass 2 loaded a full-precision (FP32) "medium"
#    model via openai-whisper for flagged segments only. OOM-crashed the
#    whole Degas service in production (2026-08-18) -- 3.7GB RAM total,
#    FP32 medium needs ~4-5GB.
# 2. Switched to faster-whisper (CTranslate2) with int8 quantization --
#    ~1/4 FP32's memory. Pass 1 stayed "small", pass 2 became a genuinely
#    bigger int8 "medium" model for flagged segments. Verified safe on
#    this server's actual RAM (peak ~2.8GB with both models loaded).
# 3. Real-world testing (2026-08-18, same day) surfaced a harder problem:
#    "small" was sometimes confidently WRONG on segments it never flagged
#    (a real example: "the fit and finishes" came out wrong but scored high
#    enough confidence to skip pass 2 entirely). Confidence-based flagging
#    can only catch "the model wasn't sure" -- it can't catch "the model
#    was sure but incorrect." Since int8 "medium" was already confirmed to
#    fit comfortably alone (a single medium model uses meaningfully less
#    memory than the small+medium combination already tested safe), the
#    fix is to stop gating accuracy behind a flag that can be fooled:
#    "medium" is now PASS1_MODEL, used for every clip. Pass 2 no longer
#    loads a second model at all -- it re-decodes the same medium model
#    with beam search for whatever medium itself still flags as
#    low-confidence, same shape as the original small-model-only fix.
#    LOW_CONFIDENCE_THRESHOLD was also raised (see flagging.py) so more
#    borderline words get a second, more careful look instead of only the
#    clearly-uncertain ones.
PASS1_MODEL = "medium"
COMPUTE_TYPE = "int8"
FFMPEG_BIN = os.environ.get("FFMPEG_BIN", "ffmpeg")
PASS2_BEAM_SIZE = 5
PASS2_BEST_OF = 5


def get_model():
    """Lazily loads the model -- used for both pass 1 (every clip) and
    pass 2 (beam-search re-decode of flagged segments, same model)."""
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        _model = WhisperModel(PASS1_MODEL, device="cpu", compute_type=COMPUTE_TYPE)
    return _model


def _extract_audio_segment(video_path, start, end, pad=0.3):
    """Cuts out [start-pad, end+pad] of video_path into a temp wav file for
    re-transcription. pad gives the re-decode a little surrounding audio
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


def _rerun_segment_with_careful_decode(video_path, segment, min_time=None, max_time=None):
    """Pass 2: re-decodes one low-confidence segment's audio with the SAME
    already-loaded model, using beam search instead of the default decode --
    a more thorough (and slower) search of possible transcriptions for just
    this short segment. Returns fresh word dicts with timestamps remapped
    back onto the full video's timeline.

    min_time/max_time define the "safe zone" this segment is allowed to
    claim words from -- the previous segment's end and the next segment's
    start, NOT this segment's own start/end. This segment's own boundaries
    came from the same pass that was already uncertain about this exact
    stretch of audio (that's why it got flagged), so its timing can be
    slightly off too. Trimming to this segment's own boundary was tried
    first and cut off real trailing words (e.g. "you know" got clipped down
    to just "you") whenever the declared boundary landed a little early.
    Trimming to the NEIGHBORING segments' boundaries instead only discards
    a word if it overlaps territory a neighbor already owns -- which is
    what actually causes duplication -- while letting this segment claim as
    much of the gap between segments as the re-decode actually found. If a
    neighbor doesn't exist (first/last segment) or isn't provided, that
    side is unbounded.

    Returns None (rather than raising) if anything goes wrong -- ffmpeg
    missing, corrupt audio, etc -- so the caller can just fall back to pass
    1's words for that segment instead of failing the whole transcription
    over one bad spot."""
    temp_path = None
    try:
        temp_path, clip_start = _extract_audio_segment(
            video_path, segment["start"], segment["end"]
        )
        model = get_model()
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
                word_start = round(w.start + clip_start, 3)
                word_end = round(w.end + clip_start, 3)
                # Only exclude a word if it actually overlaps a neighboring
                # segment's own territory -- that's the real duplication
                # risk from the padding this clip was cut with.
                midpoint = (word_start + word_end) / 2
                if min_time is not None and midpoint < min_time:
                    continue
                if max_time is not None and midpoint > max_time:
                    continue
                new_words.append({
                    "word":       w.word.strip(),
                    "start":      word_start,
                    "end":        word_end,
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

    Pass 1 -- an int8-quantized "medium" model (faster-whisper/CTranslate2)
    transcribes the full clip and records a per-word confidence score.
    "medium" (not "small") is used for every clip, not just ones that turn
    out to need a re-check -- see the accuracy-history comment above for
    why gating a better model behind a confidence flag wasn't good enough.

    Pass 2 -- any segment that came out of pass 1 with at least one word
    below LOW_CONFIDENCE_THRESHOLD gets that slice of audio cut out and
    re-decoded with the SAME model using beam search (slower, more
    thorough than the default decode), and the improved words are spliced
    back in. This only re-processes the shaky parts, not the whole clip,
    so clips with no confidence issues pay no pass-2 cost at all -- and it
    never loads a second model.

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
    for i, segment in enumerate(all_segments):
        if not any(w["confidence"] < LOW_CONFIDENCE_THRESHOLD for w in segment["words"]):
            continue
        # Neighboring segments' boundaries, not this segment's own -- see
        # _rerun_segment_with_careful_decode's docstring for why.
        min_time = all_segments[i - 1]["end"] if i > 0 else None
        max_time = all_segments[i + 1]["start"] if i + 1 < len(all_segments) else None
        revised_words = _rerun_segment_with_careful_decode(video_path, segment, min_time, max_time)
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
