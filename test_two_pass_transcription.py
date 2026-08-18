"""
Tests for the two-pass transcription accuracy improvement (Ben's ask:
"pass one is 90% confident, calls out things it's not confident on, pass two
re-checks those, pass three is manual check").

Pass 1 = Whisper "small" transcribes the full clip, recording a per-word
confidence score (previously discarded -- flagging.py's confidence flagging
was silently dead code before this fix).

Pass 2 = any segment with a low-confidence word gets its audio re-cut and
re-decoded with the SAME small model, but using beam search (beam_size=5,
best_of=5) instead of the default greedy decode, and the improved words are
spliced back in.

IMPORTANT HISTORY: pass 2 originally loaded a larger "medium" Whisper model
instead of re-decoding with the same model. That OOM-crashed the whole Degas
service in production on 2026-08-18 -- the server has 3.7GB RAM total, and
"medium" needs roughly 4-5GB in CPU/FP32 mode on top of "small" already being
resident. The beam-search-on-the-same-model approach fixes this: pass 2 now
never loads a second model, so it cannot repeat that crash. Don't reintroduce
a second model here without re-checking actual server memory first.

Pass 3 = the existing manual Caption Review UI -- not automated, intentionally.

Whisper and ffmpeg are both mocked so this runs without real audio, a real
model download, or real transcription.

Run with: python -m pytest test_two_pass_transcription.py -v
(or just: python test_two_pass_transcription.py)
"""

import json
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import transcription
from flagging import LOW_CONFIDENCE_THRESHOLD, annotate_segments_with_flags


def _fake_pass1_result():
    # Segment 1: all high confidence -- should NOT trigger pass 2.
    # Segment 2: one low-confidence word -- SHOULD trigger pass 2.
    return {
        "segments": [
            {
                "start": 0.0, "end": 2.0, "text": "Hello there friend",
                "words": [
                    {"word": " Hello", "start": 0.0, "end": 0.5, "probability": 0.98},
                    {"word": " there", "start": 0.5, "end": 1.0, "probability": 0.95},
                    {"word": " friend", "start": 1.0, "end": 2.0, "probability": 0.92},
                ],
            },
            {
                "start": 2.0, "end": 4.0, "text": "Mumbled unclear word",
                "words": [
                    {"word": " Mumbled", "start": 2.0, "end": 2.5, "probability": 0.55},
                    {"word": " unclear", "start": 2.5, "end": 3.0, "probability": 0.60},
                    {"word": " word", "start": 3.0, "end": 4.0, "probability": 0.90},
                ],
            },
        ]
    }


def _fake_pass2_result():
    # The re-decoded (cut, padded) audio -- times are relative to the start
    # of the extracted clip, not the full video.
    return {
        "segments": [
            {
                "start": 0.0, "end": 2.0, "text": "Bundled up clear word",
                "words": [
                    {"word": " Bundled", "start": 0.1, "end": 0.6, "probability": 0.97},
                    {"word": " up", "start": 0.6, "end": 0.9, "probability": 0.96},
                    {"word": " clear", "start": 0.9, "end": 1.4, "probability": 0.94},
                    {"word": " word", "start": 1.4, "end": 1.9, "probability": 0.99},
                ],
            }
        ]
    }


class FakeWhisperModel:
    """Same model instance used for both pass 1 (whole clip, default decode)
    and pass 2 (short segment, beam search decode) -- exactly like production,
    where get_model() is called both times and returns the same object."""

    def __init__(self):
        self.calls = []

    def transcribe(self, path, **kwargs):
        self.calls.append({"path": path, "kwargs": kwargs})
        if len(self.calls) == 1:
            return _fake_pass1_result()
        return _fake_pass2_result()


def test_pass1_records_confidence_and_pass2_only_hits_low_confidence_segment():
    model = FakeWhisperModel()
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

    # Pass 2 should have run exactly once, only for the low-confidence segment (2.0-4.0)
    assert pass2_extract_calls == [(2.0, 4.0)], f"expected pass 2 only on the shaky segment, got {pass2_extract_calls}"

    # Only 2 total transcribe() calls: one full-clip pass 1, one segment-level pass 2.
    # No second model was ever loaded -- both calls used the same FakeWhisperModel.
    assert len(model.calls) == 2

    # Pass 2's call must actually request beam search, not the default greedy decode.
    pass2_kwargs = model.calls[1]["kwargs"]
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


def test_pass2_never_loads_a_second_model():
    # Regression test for the actual production incident: transcription.py
    # must not define/use any kind of second, larger model anywhere.
    source = open("transcription.py").read()
    assert 'load_model("medium")' not in source, "pass 2 must not load a second, larger model -- see test docstring"
    assert not hasattr(transcription, "get_pass2_model"), "pass 2 must reuse get_model(), not a second model loader"


def test_pass2_falls_back_to_pass1_words_on_failure():
    model = FakeWhisperModel()

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
    test_pass2_never_loads_a_second_model()
    print("PASS: test_pass2_never_loads_a_second_model")
    test_pass2_falls_back_to_pass1_words_on_failure()
    print("PASS: test_pass2_falls_back_to_pass1_words_on_failure")
    test_flagging_now_actually_flags_low_confidence_words()
    print("PASS: test_flagging_now_actually_flags_low_confidence_words")
    print("\nALL TESTS PASSED")
