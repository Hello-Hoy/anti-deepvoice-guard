import json

from jhm.gptsovits import (
    build_list_lines,
    parse_reject,
    relocate_lines,
    remap_segments,
    seg_id,
    read_manifest,
    write_manifest,
)


def test_seg_id_zero_padded():
    assert seg_id(1) == "jhm_0001"
    assert seg_id(42) == "jhm_0042"


def test_remap_16k_to_32k_doubles_indices():
    segs = [{"start": 16000, "end": 24000, "sim": 0.9}]
    out = remap_segments(segs, sr_from=16000, sr_to=32000)
    assert out[0]["start"] == 32000
    assert out[0]["end"] == 48000
    assert out[0]["sim"] == 0.9  # 다른 키 보존


def test_parse_reject_ignores_comments_and_blanks():
    text = "# 헤더 주석\njhm_0003\n\n  jhm_0007  # 잡음\n#jhm_0009\n"
    assert parse_reject(text) == {"jhm_0003", "jhm_0007"}


def test_build_list_lines_format_and_skip_empty():
    entries = [
        {"relpath": "wavs/jhm_0001.wav", "text": "안녕하세요"},
        {"relpath": "wavs/jhm_0002.wav", "text": "   "},   # 빈 전사 → 제외
        {"relpath": "wavs/jhm_0003.wav", "text": " 반갑습니다 "},
    ]
    lines = build_list_lines(entries, speaker="jhm", lang="ko")
    assert lines == [
        "wavs/jhm_0001.wav|jhm|ko|안녕하세요",
        "wavs/jhm_0003.wav|jhm|ko|반갑습니다",
    ]


def test_relocate_lines_rewrites_first_field():
    lines = ["wavs/jhm_0001.wav|jhm|ko|안녕"]
    out = relocate_lines(lines, base="D:/gpt-sovits/jhm", sep="/")
    assert out == ["D:/gpt-sovits/jhm/wavs/jhm_0001.wav|jhm|ko|안녕"]


def test_manifest_roundtrip(tmp_path):
    entries = [{"id": "jhm_0001", "source": "a.m4a", "start_s": 1.0, "end_s": 3.0, "dur": 2.0, "sim": 0.88}]
    p = tmp_path / "segments.json"
    write_manifest(p, entries)
    assert read_manifest(p) == entries
