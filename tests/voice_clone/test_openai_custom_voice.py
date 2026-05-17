from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model_training.voice_clone.openai_custom_voice import (
    CONSENT_PHRASES,
    MAX_CUSTOM_VOICE_UPLOAD_BYTES,
    _escape_form_value,
    build_speech_payload,
    main,
    mime_type_for_audio,
    validate_custom_voice_audio,
    validate_own_voice_confirmation,
)


def test_validate_own_voice_confirmation_rejects_missing_confirmation() -> None:
    with pytest.raises(ValueError, match="본인 목소리"):
        validate_own_voice_confirmation(False)


def test_mime_type_for_audio_accepts_openai_custom_voice_formats(tmp_path: Path) -> None:
    wav_path = tmp_path / "sample.wav"
    wav_path.write_bytes(b"not real wav")

    assert mime_type_for_audio(wav_path) == "audio/x-wav"


def test_mime_type_for_audio_uses_audio_mp4_for_mp4_files(tmp_path: Path) -> None:
    mp4_path = tmp_path / "sample.mp4"
    mp4_path.write_bytes(b"not real mp4")

    assert mime_type_for_audio(mp4_path) == "audio/mp4"


def test_mime_type_for_audio_rejects_unknown_extension(tmp_path: Path) -> None:
    bad_path = tmp_path / "sample.txt"
    bad_path.write_text("not audio")

    with pytest.raises(ValueError, match="지원하지 않는 오디오 형식"):
        mime_type_for_audio(bad_path)


def test_build_speech_payload_uses_custom_voice_object() -> None:
    payload = build_speech_payload(
        voice_id="voice_123abc",
        text="테스트 문장입니다.",
        model="gpt-4o-mini-tts",
        language="ko",
        output_format="wav",
    )

    assert payload["voice"] == {"id": "voice_123abc"}
    assert payload["input"] == "테스트 문장입니다."
    assert payload["language"] == "ko"
    assert payload["response_format"] == "wav"
    assert "format" not in payload
    json.dumps(payload, ensure_ascii=False)


def test_validate_custom_voice_audio_rejects_files_over_openai_upload_limit(tmp_path: Path) -> None:
    large_path = tmp_path / "large.mp3"
    large_path.write_bytes(b"0" * (MAX_CUSTOM_VOICE_UPLOAD_BYTES + 1))

    with pytest.raises(ValueError, match="10MiB"):
        validate_custom_voice_audio(large_path, "sample recording")


def test_synthesize_requires_own_voice_confirmation_before_api_key_check(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        [
            "synthesize",
            "--voice-id",
            "voice_123abc",
            "--text",
            "테스트 문장입니다.",
            "--out",
            str(tmp_path / "out.wav"),
        ]
    )

    assert exit_code == 1
    assert "본인 목소리" in capsys.readouterr().err


def test_escape_form_value_rejects_header_newlines() -> None:
    with pytest.raises(ValueError, match="줄바꿈"):
        _escape_form_value("bad\nname")


def test_korean_consent_phrase_matches_openai_required_script() -> None:
    assert CONSENT_PHRASES["ko"] == (
        "나는 이 음성의 소유자이며 OpenAI가 이 음성을 사용하여 "
        "음성 합성 모델을 생성할 것을 허용합니다."
    )
