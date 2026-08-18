import json
import os

_model = None


def get_model():
    global _model
    if _model is None:
        import whisper
        _model = whisper.load_model("small")
    return _model


def transcribe(video_path, words_path, segments_path, original_path=None):
    """
    Transcribe a video file using Whisper small (bumped up from base for
    better accuracy on names/jargon).
    Saves word-level timestamps to words_path (.words.json)
    and segment-level data to segments_path (.segments.json).
    """
    model = get_model()
    result = model.transcribe(video_path, word_timestamps=True)

    all_words = []
    all_segments = []

    for segment in result["segments"]:
        seg_words = []
        for w in segment.get("words", []):
            entry = {
                "word":  w["word"].strip(),
                "start": w["start"],
                "end":   w["end"],
            }
            all_words.append(entry)
            seg_words.append(entry)

        all_segments.append({
            "start": segment["start"],
            "end":   segment["end"],
            "text":  segment["text"].strip(),
            "words": seg_words,
        })

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
