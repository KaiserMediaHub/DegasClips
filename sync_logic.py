import json
import os


def update_words_from_segments(edited_segments, words_path):
    """
    Update words.json with edited segment text.

    edited_segments: list of dicts with keys: start (float), end (float), text (str)

    - If word count matches the original: preserve exact Whisper timestamps.
    - If word count differs: interpolate timestamps across the segment window.
    """
    if not os.path.exists(words_path):
        return

    with open(words_path, "r", encoding="utf-8") as f:
        original_words = json.load(f)

    new_words = []

    for seg in edited_segments:
        seg_start      = float(seg["start"])
        seg_end        = float(seg["end"])
        new_text_words = seg["text"].split()

        if not new_text_words:
            continue

        orig_seg_words = [
            w for w in original_words
            if w["start"] >= seg_start - 0.05 and w["end"] <= seg_end + 0.05
        ]

        if len(orig_seg_words) == len(new_text_words):
            for orig, new_word in zip(orig_seg_words, new_text_words):
                new_words.append({
                    "word":  new_word,
                    "start": orig["start"],
                    "end":   orig["end"],
                })
        else:
            duration      = seg_end - seg_start
            word_duration = duration / len(new_text_words)
            for j, new_word in enumerate(new_text_words):
                new_words.append({
                    "word":  new_word,
                    "start": seg_start + j * word_duration,
                    "end":   seg_start + (j + 1) * word_duration,
                })

    with open(words_path, "w", encoding="utf-8") as f:
        json.dump(new_words, f, indent=2)
