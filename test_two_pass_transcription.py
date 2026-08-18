"""
Tests for the two-pass transcription accuracy improvement (Ben's ask:
"pass one is 90% confident, calls out things it's not confident on, pass two
re-checks those, pass three is manual check").

Pass 1 = an int8-quantized "medium" model (faster-whisper/CTranslate2)
transcribes the full clip, recording a per-word confidence score.

Pass 2 = any segment with a low-confidence word gets its audio re-cut and
re-decoded with the SAME model using beam search, and the improved words
are spliced back in.

HISTORY, in order:
1. Original version: pass 1 was "small", pass 2 loaded a full-precision
   (FP32) "medium" model via openai-whisper for flagged segments only.
   OOM-crashed the whole Degas service in production (2026-08-18) -- server
   has 3.7GB RAM, FP32 medium needs ~4-5GB.
2. Switched to faster-whisper (CTranslate2) with int8 quantization (~1/4
   FP32's memory). Pass 1 stayed "small", pass 2 became a genuinely bigger
   int8 "medium" model for flagged segments. Verified safe on this server's
   actual RAM (peak ~2.8GB with both models loaded simultaneously).
3. Same day: pass 2's padded re-transcription bled into the NEXT segment's
   audio and duplicated a phrase ("of that" appeared twice). Fixed by
   trimming pass-2 words outside the segment's own [start, end].
4. That trim then over-corrected: a real trailing word ("you know" -> just
   "you") got clipped because the ORIGINAL pass's declared segment boundary
   was imprecise -- unsurprising, since this exact segment was flagged for
   being uncertain in the first place. Fixed by trimming against the
   NEIGHBORING segments' boundaries instead of this segment's own.
5. Real-world testing then surfaced a harder problem: "small" was
   sometimes confidently WRONG on segments it never flagged at all (e.g.
   "the fit and finishes" came out wrong but scored confident enough to
   skip pass 2 entirely). Confidence-based flagging can only catch "the
   model wasn't sure" -- it can't catch "the model was sure but incorrect."
   Fixed by making "medium" the PRIMARY model (pass 1) for every clip
   instead of gating it behind a flag that can be fooled -- pass 2 no
   longer loads a second model at all, it just re-decodes the same medium
   model with beam search. LOW_CONFIDENCE_THRESHOLD was also raised
   (0.8 -> 0.9, see flagging.py) so more borderline words get flagged.

Pass 3 = the existing manual Caption Review UI -- not automated, intentionally.

faster-whisper is mocked (matching its real attribute-based Segment/Word
API, not dicts) so this runs without real audio, a real model download, or
real transcription.

Run with: python -m pytest test_two_pass_transcription.py -v
(or just: python test_two_pass_transcription.py)
"""

import json
import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import transcription
from flagging import LOW_CONFIDENCE_THRESHOLD, annotate_segments_with_flags


def _word(word, start, end, probability):
    # faster-whisper's real Word objects use attribute access, not dict
    # access -- SimpleNamespace mimics that shape exactly.
    return SimpleNamespace(word=word, start=start, end=end, probability=probability)


def _segment(start, end, text, words):
    return SimpleNamespace(start=start, end=end, text=text, words=words)


def _fake_pass1_segments():
    # Segment 0: high confidence -- should NOT trigger pass 2.
    # Segment 1: one low-confidence word -- SHOULD trigger pass 2. Its own
    #   declared end (4.0) is deliberately a little early/imprecise, same
    #   as real Whisper output for an uncertain stretch of audio.
    # Segment 2: high confidence, starts at 4.5 -- gives a real gap (4.0 to
    #   4.5) between segment 1's own (imprecise) end and segment 2's actual
    #   start, which is exactly the zone pass 2 needs to be allowed to use.
    return [
        _segment(0.0, 2.0, "Hello there friend", [
            _word(" Hello", 0.0, 0.5, 0.98),
            _word(" there", 0.5, 1.0, 0.95),
            _word(" friend", 1.0, 2.0, 0.92),
        ]),
        _segment(2.0, 4.0, "Mumbled unclear word", [
            _word(" Mumbled", 2.0, 2.5, 0.55),
            _word(" unclear", 2.5, 3.0, 0.60),
            _word(" word", 3.0, 4.0, 0.85),
        ]),
        _segment(4.5, 6.0, "Totally different sentence", [
            _word(" Totally", 4.5, 4.8, 0.99),
            _word(" different", 4.8, 5.3, 0.98),
            _word(" sentence", 5.3, 6.0, 0.97),
        ]),
    ]


def _fake_pass2_segments():
    # The re-decoded (cut, padded) audio for segment 1 -- times are
    # relative to the start of the extracted clip, not the full video.
    # clip_start = 2.0 - 0.3 = 1.7.
    return [
        _segment(0.0, 3.1, "Bundled up clear word know next", [
            _word(" Bundled", 0.1, 0.6, 0.97),
            _word(" up", 0.6, 0.9, 0.96),
            _word(" clear", 0.9, 1.4, 0.94),
            _word(" word", 1.4, 1.9, 0.99),
            # local 2.3-2.6 -> full timeline 4.0-4.3. Past segment 1's OWN
            # declared end (4.0) but well before segment 2's real start
            # (4.5) -- this is the "you know" case: a legitimate trailing
            # word the original pass just didn't bound tightly enough.
            # Must be KEPT.
            _word(" know", 2.3, 2.6, 0.93),
            # local 2.9-3.1 -> full timeline 4.6-4.8. This is INSIDE segment
            # 2's real territory (4.5-6.0) -- genuine padding bleed into the
            # next segment's actual speech. Must be TRIMMED.
            _word(" next", 2.9, 3.1, 0.90),
        ]),
    ]


class FakeModel:
    """Same model instance used for both pass 1 (whole clip, default decode)
    and pass 2 (short segment, beam search decode) -- exactly like
    production, where get_model() is called both times and returns the
    same object, since "medium" is now the only model Degas loads."""

    def __init__(self):
        self.calls = []

    def transcribe(self, path, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            return _fake_pass1_segments(), SimpleNamespace(language="en")
        return _fake_pass2_segments(), SimpleNamespace(language="en")


def test_pass1_uses_medium_and_pass2_only_hits_low_confidence_segment():
    model = FakeModel()
    pass2_extract_calls = []

    def fake_extract(video_path, start, end, pad=0.3):
        pass2_extract_calls.append((start, end))
        return "/tmp/fake_extracted.wav", max(0.0, start - pad)

    with patch.object(transcription, "get_model", return_value=model), \
         patch.object(transcription, "_extract_audio_segment", side_effect=fake_extract), \
         patch("os.path.exists", return_value=False):  # skip the temp-file cleanup branch

        words_path = "/tmp/test_words.json"
        segments_path = "/tmp/test_segments.json"
        transcription.transcribe("fake_video.mp4", words_path, segments_path)

    with open(words_path) as f:
        words = json.load(f)
    with open(segments_path) as f:
        segments = json.load(f)

    # Only 2 total transcribe() calls: one full-clip pass 1, one segment-level
    # pass 2 -- no second model was ever loaded, both used the same FakeModel.
    assert len(model.calls) == 2
    assert pass2_extract_calls == [(2.0, 4.0)], f"expected pass 2 only on the shaky segment, got {pass2_extract_calls}"

    # Pass 1 must actually load "medium", not "small" -- this is the whole point
    # of this round of fixes: accuracy shouldn't be gated behind a confidence flag.
    assert transcription.PASS1_MODEL == "medium"

    # Pass 2's call must request beam search.
    pass2_kwargs = model.calls[1]
    assert pass2_kwargs.get("beam_size") == transcription.PASS2_BEAM_SIZE
    assert pass2_kwargs.get("best_of") == transcription.PASS2_BEST_OF

    # Segment 0 (high confidence) must be untouched by pass 2
    assert segments[0]["text"] == "Hello there friend"
    assert all(not w.get("revised") for w in segments[0]["words"])

    # Segment 2 (high confidence, never sent through pass 2) must also be untouched
    assert segments[2]["text"] == "Totally different sentence"
    assert all(not w.get("revised") for w in segments[2]["words"])

    # Segment 1 must contain pass 2's improved words, remapped onto the full timeline.
    # "know" extends past segment 1's OWN (imprecise) declared end but stays well
    # short of segment 2's real start -- it's legitimate speech and must be KEPT.
    # "next" actually overlaps segment 2's real territory -- genuine padding
    # bleed -- and must be TRIMMED.
    assert segments[1]["text"] == "Bundled up clear word know", segments[1]["text"]
    assert all(w.get("revised") for w in segments[1]["words"])
    # clip_start for segment starting at 2.0 with pad 0.3 = 1.7; pass2 word at
    # local 0.1 -> should land at 1.7 + 0.1 = 1.8 on the full timeline
    assert abs(segments[1]["words"][0]["start"] - 1.8) < 0.01

    # Flat word list must be rebuilt from the (possibly revised) segments, with
    # "know" kept and the bled-over "next" excluded
    assert [w["word"] for w in words] == [
        "Hello", "there", "friend",
        "Bundled", "up", "clear", "word", "know",
        "Totally", "different", "sentence",
    ]

    os.remove(words_path)
    os.remove(segments_path)


def test_pass2_never_loads_a_second_model():
    # Regression test: pass 2 must reuse get_model(), not load anything else --
    # this is what keeps memory usage safe (a single medium model, not two).
    source = open("transcription.py").read()
    assert "get_pass2_model" not in source, "pass 2 must reuse get_model(), not a second model loader"
    assert 'COMPUTE_TYPE = "int8"' in source, "the model must load with int8 quantization"


def test_pass2_falls_back_to_pass1_words_on_failure():
    model = FakeModel()

    with patch.object(transcription, "get_model", return_value=model), \
         patch.object(transcription, "_rerun_segment_with_careful_decode", return_value=None):

        words_path = "/tmp/test_words_fallback.json"
        segments_path = "/tmp/test_segments_fallback.json"
        transcription.transcribe("fake_video.mp4", words_path, segments_path)

    with open(segments_path) as f:
        segments = json.load(f)

    # Pass 2 "failed" (returned None) -- pass 1's original shaky words must survive
    assert segments[1]["text"] == "Mumbled unclear word"
    assert segments[1]["words"][0]["confidence"] == 0.55

    os.remove(words_path)
    os.remove(segments_path)


def test_flagging_threshold_raised_to_catch_confidently_wrong_words():
    # Regression test for the actual production incident: a word scoring
    # 0.85 (confident enough to have slipped past the old 0.8 threshold)
    # must now be flagged, since the threshold was raised to 0.9.
    assert LOW_CONFIDENCE_THRESHOLD == 0.9
    words = [
        {"word": "clear", "start": 0.0, "end": 0.5, "confidence": 0.95},
        {"word": "wrongish", "start": 0.5, "end": 1.0, "confidence": 0.85},
    ]
    segments = [{"start": 0.0, "end": 1.0, "text": "clear wrongish"}]
    annotated = annotate_segments_with_flags(segments, words)
    assert annotated[0]["flagged"] is True
    assert annotated[0]["words"][0]["flagged"] is False
    assert annotated[0]["words"][1]["flagged"] is True


if __name__ == "__main__":
    test_pass1_uses_medium_and_pass2_only_hits_low_confidence_segment()
    print("PASS: test_pass1_uses_medium_and_pass2_only_hits_low_confidence_segment")
    test_pass2_never_loads_a_second_model()
    print("PASS: test_pass2_never_loads_a_second_model")
    test_pass2_falls_back_to_pass1_words_on_failure()
    print("PASS: test_pass2_falls_back_to_pass1_words_on_failure")
    test_flagging_threshold_raised_to_catch_confidently_wrong_words()
    print("PASS: test_flagging_threshold_raised_to_catch_confidently_wrong_words")
    print("\nALL TESTS PASSED")
