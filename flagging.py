LOW_CONFIDENCE_THRESHOLD = 0.8


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
