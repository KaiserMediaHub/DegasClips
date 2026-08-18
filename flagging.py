# Raised from 0.8 to 0.9 on 2026-08-18 after a real miss: a wrong word
# ("the fit and finishes" transcribed incorrectly) scored confident enough
# to slip past 0.8 and never got flagged for pass 2's second look or a
# human's attention in Caption Review. A higher bar means more borderline
# words get double-checked instead of only the clearly-uncertain ones --
# trades some extra false-positive flags (a correct word that just happened
# to score under 0.9) for catching more real mistakes. See
# transcription.py's accuracy-history comment for the fuller story.
LOW_CONFIDENCE_THRESHOLD = 0.9


def annotate_segments_with_flags(segments, words):
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
