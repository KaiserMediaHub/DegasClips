"""
Tests for the two-pass transcription accuracy improvement (Ben's ask:
"pass one is 90% confident, calls out things it's not confident on, pass two
re-checks those, pass three is manual check").

Pass 1 = an int8-quantized "small" model (faster-whisper/CTranslate2)
transcribes the full clip, recording a per-word confidence score
(previously discarded under the old openai-whisper implementation --
flagging.py's confidence flagging was silently dead code before that fix).

Pass 2 = any segment with a low-confidence word gets its audio re-cut and
re-transcribed with a larger int8-quantized model ("medium") using beam
search, and the improved words are spliced back in.

HISTORY, in order:
1. Original version: pass 2 loaded a full-precision (FP32) "medium" model
   via openai-whisper. OOM-crashed the whole Degas service in production
   (2026-08-18) -- server has 3.7GB RAM, FP32 medium needs ~4-5GB.
2. First fix: pass 2 re-decoded with the SAME small model plus beam search,
   avoiding a second model entirely. Safe, but a real accuracy ceiling since
   it can't fix anything "small" fundamentally doesn't know.
3. Current version (this file): switched the whole library from
   openai-whisper to faster-whisper (CTranslate2), which supports int8
   quantization -- roughly 1/4 the memory of FP32 for the same model. This
   makes a genuinely bigger "medium" model safe to load for pass 2. Do not
   revert to full-precision/openai-whisper without re-checking actual
   server memory headroom first.

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
    # Segment 1: all high confidence -- should NOT trigger pass 2.
    # Segment 2: one low-confidence word -- SHOULD trigger pass 2.
    return [
        _segment(0.0, 2.0, "Hello there friend", [
            _word(" Hello", 0.0, 0.5, 0.98),
            _word(" there", 0.5, 1.0, 0.95),
            _word(" friend", 1.0, 2.0, 0.92),
        ]),
        _segment(2.0, 4.0, "Mumbled unclear word", [
            _word(" Mumbled", 2.0, 2.5, 0.55),
            _word(" unclear", 2.5, 3.0, 0.60),
            _word(" word", 3.0, 4.0, 0.90),
        ]),
    ]


def _fake_pass2_segments():
    # The re-transcribed (cut, padded) audio -- times are relative to the
    # start of the extracted clip, not the full video.
    return [
        _segment(0.0, 2.0, "Bundled up clear word", [
            _word(" Bundled", 0.1, 0.6, 0.97),
            _word(" up", 0.6, 0.9, 0.96),
            _word(" clear", 0.9, 1.4, 0.94),
            _word(" word", 1.4, 1.9, 0.99),
        ]),
    ]


class FakePass1Model:
    def __init__(self):
        self.calls = []

    def transcribe(self, path, **kwargs):
        self.calls.append(kwargs)
        return _fake_pass1_segments(), SimpleNamespace(language="en")


class FakePass2Model:
    def __init__(self):
        self.calls = []

    def transcribe(self, path, **kwargs):
        self.calls.append(kwargs)
        return _fake_pass2_segments(), SimpleNamespace(language="en")


def test_pass1_records_confidence_and_pass2_only_hits_low_confidence_segment():
    pass1_model = FakePass1Model()
    pass2_model = FakePass2Model()
    pass2_extract_calls = []

    def fake_extract(video_path, start, end, pad=0.3):
        pass2_extract_calls.append((start, end))
        return "/tmp/fake_extracted.wav", max(0.0, start - pad)

    with patch.object(transcription, "get_model", return_value=pass1_model), \
         patch.object(transcription, "get_pass2_model", return_value=pass2_model), \
         patch.object(transcription, "_extract_audio_segment", side_effect=fake_extract), \
         patch("os.path.exists", return_value=False):  # skip the temp-file cleanup branch

        words_path = "/tmp/test_words.json"
        segments_path = "/tmp/test_segments.json"
        transcription.transcribe("fake_video.mp4", words_path, segments_path)

    with open(words_path) as f:
        words = json.load(f)
    with open(segments_path) as f:
        segments = json.load(f)

    # Pass 2 should have run exactly once, only for the low-confidence segment (2.0-4.0)
    assert pass2_extract_calls == [(2.0, 4.0)], f"expected pass 2 only on the shaky segment, got {pass2_extract_calls}"
    assert len(pass1_model.calls) == 1
    assert len(pass2_model.calls) == 1

    # Pass 2's call must actually request beam search on the bigger model.
    pass2_kwargs = pass2_model.calls[0]
    assert pass2_kwargs.get("beam_size") == transcription.PASS2_BEAM_SIZE
    assert pass2_kwargs.get("best_of") == transcription.PASS2_BEST_OF

    # Segment 1 (high confidence) must be untouched by pass 2
    assert segments[0]["text"] == "Hello there friend"
    assert all(not w.get("revised") for w in segments[0]["words"])

    # Segment 2 must now contain pass 2's improved words, remapped onto the full timeline
    assert segments[1]["text"] == "Bundled up clear word"
    assert all(w.get("revised") for w in segments[1]["words"])
    # clip_start for segment starting at 2.0 with pad 0.3 = 1.7; pass2 word at
    # local 0.1 -> should land at 1.7 + 0.1 = 1.8 on the full timeline
    assert abs(segments[1]["words"][0]["start"] - 1.8) < 0.01

    # Flat word list must be rebuilt from the (possibly revised) segments
    assert [w["word"] for w in words] == ["Hello", "there", "friend", "Bundled", "up", "clear", "word"]

    os.remove(words_path)
    os.remove(segments_path)


def test_both_models_use_int8_quantization():
    # Regression test for the actual production incident: neither model may
    # ever load at full precision -- that's what caused the OOM crash.
    source = open("transcription.py").read()
    assert 'COMPUTE_TYPE = "int8"' in source, \
        "both pass 1 and pass 2 models must load with int8 quantization -- see test docstring history"


def test_pass2_falls_back_to_pass1_words_on_failure():
    pass1_model = FakePass1Model()

    with patch.object(transcription, "get_model", return_value=pass1_model), \
         patch.object(transcription, "_rerun_segment_with_bigger_model", return_value=None):

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


def test_flagging_now_actually_flags_low_confidence_words():
    # This is the regression test for the original dead-code bug: flagging.py
    # expected a "confidence" key that transcription.py never wrote, so
    # nothing was ever flagged. Confirm the two now actually connect.
    words = [
        {"word": "clear", "start": 0.0, "end": 0.5, "confidence": 0.95},
        {"word": "shaky", "start": 0.5, "end": 1.0, "confidence": 0.4},
    ]
    segments = [{"start": 0.0, "end": 1.0, "text": "clear shaky"}]
    annotated = annotate_segments_with_flags(segments, words)
    assert annotated[0]["flagged"] is True
    assert annotated[0]["words"][0]["flagged"] is False
    assert annotated[0]["words"][1]["flagged"] is True


if __name__ == "__main__":
    test_pass1_records_confidence_and_pass2_only_hits_low_confidence_segment()
    print("PASS: test_pass1_records_confidence_and_pass2_only_hits_low_confidence_segment")
    test_both_models_use_int8_quantization()
    print("PASS: test_both_models_use_int8_quantization")
    test_pass2_falls_back_to_pass1_words_on_failure()
    print("PASS: test_pass2_falls_back_to_pass1_words_on_failure")
    test_flagging_now_actually_flags_low_confidence_words()
    print("PASS: test_flagging_now_actually_flags_low_confidence_words")
    print("\nALL TESTS PASSED")
