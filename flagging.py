"""
KMG Studio task #8: turns Whisper's per-word confidence scores into a
per-segment "needs review" flag, so Caption Review can default to showing
only the segments that actually need a human look instead of the full
transcript. See STUDIO_SYSTEM_DESIGN.md, "Caption Review: reducing a full
manual read-through to a flagged queue."

Starting threshold set by Ben 7/17: 0.8 (conservative -- flags more than
strictly necessary at first). Easy to lower later once real transcripts
show whether it's too noisy.
"""

LOW_CONFIDENCE_THRESHOLD = 0.8


def annotate_segments_with_flags(segments, words):
    """Attaches each segment's own words (with a 'flagged' bool per word)
    and a segment-level 'flagged' bool, by matching words to segments on
    their time window -- same tolerance sync_logic.py already uses so the
    two stay consistent."""
    annotated = []
    for seg in segments:
        seg_start = seg["start"]
        seg_end = seg["end"]
        seg_words = [
            dict(w) for w in words
            if w["start"] >= seg_start - 0.05 and w["end"] <= seg_end + 0.05
        ]
        for w in seg_words:
            confidence = w.get("confidence")
            w["flagged"] = confidence is not None and confidence < LOW_CONFIDENCE_THRESHOLD

        seg_out = dict(seg)
        seg_out["words"] = seg_words
        seg_out["flagged"] = any(w["flagged"] for w in seg_words)
        annotated.append(seg_out)
    return annotated
