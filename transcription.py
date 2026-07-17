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
    Transcribe a video file using Whisper small. Saves word-level timestamps
    to words_path (.words.json) and segment-level data to segments_path
    (.segments.json).

    KMG Studio task #8: each word now also carries Whisper's own confidence
    score (was previously discarded), and if original_path is given, an
    immutable snapshot of the segments is written there too -- Caption
    Review edits overwrite segments_path, but the glossary system needs a
    "before" version to diff against to detect candidate name/company/
    figure corrections.
    """
    model = get_model()
    result = model.transcribe(video_path, word_timestamps=True)

    all_words = []
    all_segments = []

    for segment in result["segments"]:
        seg_words = []
        for w in segment.get("words", []):
            entry = {
                "word":       w["word"].strip(),
                "start":      w["start"],
                "end":        w["end"],
                "confidence": round(w.get("probability", 1.0), 3),
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

    if original_path:
        with open(original_path, "w", encoding="utf-8") as f:
            json.dump(all_segments, f, indent=2)
