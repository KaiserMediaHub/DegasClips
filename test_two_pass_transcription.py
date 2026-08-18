"""
Tests for the two-pass transcription accuracy improvement added 2026-08-18
(Ben's ask: "pass one is 90% confident, calls out things it's not confident
on, pass two re-checks those, pass three is manual check").

Pass 1 = Whisper "small" transcribes the full clip, now actually recording a
per-word confidence score (previously discarded -- flagging.py's confidence
flagging was silently dead code before this fix).
Pass 2 = any segment with a low-confidence word gets its audio re-cut and
re-transcribed with a larger model ("medium"), spliced back in.
Pass 3 = the existing manual Caption Review UI -- not automated, intentionally.

Whisper and ffmpeg are both mocked so this runs without real audio, a real
model download, or real transcription.

Run with: python -m pytest test_two_pass_transcription.py -v
(or just: python test_two_pass_transcription.py)
"""

import json
import os
import sys
from unittest.mock import patch, MagicMock

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
    # The re-transcribed (cut, padded) audio -- times are relative to the
    # start of the extracted clip, not the full video.
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
    def __init__(self, result):
        self._result = result

    def transcribe(self, path, word_timestamps=True):
        return self._result


def test_pass1_records_confidence_and_pass2_only_hits_low_confidence_segment():
    pass1_model = FakeWhisperModel(_fake_pass1_result())
    pass2_model = FakeWhisperModel(_fake_pass2_result())
    pass2_calls = []

    def fake_extract(video_path, start, end, pad=0.3):
        pass2_calls.append((start, end))
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
    assert pass2_calls == [(2.0, 4.0)], f"expected pass 2 only on the shaky segment, got {pass2_calls}"

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


def test_pass2_falls_back_to_pass1_words_on_failure():
    pass1_model = FakeWhisperModel(_fake_pass1_result())

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
    # This is the regression test for the original bug: flagging.py expected
    # a "confidence" key that transcription.py never wrote, so nothing was
    # ever flagged. Confirm the two now actually connect.
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
    test_pass2_falls_back_to_pass1_words_on_failure()
    print("PASS: test_pass2_falls_back_to_pass1_words_on_failure")
    test_flagging_now_actually_flags_low_confidence_words()
    print("PASS: test_flagging_now_actually_flags_low_confidence_words")
    print("\nALL TESTS PASSED")
