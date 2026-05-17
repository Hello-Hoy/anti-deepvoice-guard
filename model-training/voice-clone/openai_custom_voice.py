#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import uuid
import wave
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


CONSENT_PHRASES: dict[str, str] = {
    "de": "Ich bin der Eigentümer dieser Stimme und bin damit einverstanden, dass OpenAI diese Stimme zur Erstellung eines synthetischen Stimmmodells verwendet.",
    "en": "I am the owner of this voice and I consent to OpenAI using this voice to create a synthetic voice model.",
    "es": "Soy el propietario de esta voz y doy mi consentimiento para que OpenAI la utilice para crear un modelo de voz sintética.",
    "fr": "Je suis le propriétaire de cette voix et j'autorise OpenAI à utiliser cette voix pour créer un modèle de voix synthétique.",
    "hi": "मैं इस आवाज का मालिक हूं और मैं सिंथेटिक आवाज मॉडल बनाने के लिए OpenAI को इस आवाज का उपयोग करने की सहमति देता हूं",
    "id": "Saya adalah pemilik suara ini dan saya memberikan persetujuan kepada OpenAI untuk menggunakan suara ini guna membuat model suara sintetis.",
    "it": "Sono il proprietario di questa voce e acconsento che OpenAI la utilizzi per creare un modello di voce sintetica.",
    "ja": "私はこの音声の所有者であり、OpenAIがこの音声を使用して音声合成 モデルを作成することを承認します。",
    "ko": "나는 이 음성의 소유자이며 OpenAI가 이 음성을 사용하여 음성 합성 모델을 생성할 것을 허용합니다.",
    "nl": "Ik ben de eigenaar van deze stem en ik geef OpenAI toestemming om deze stem te gebruiken om een synthetisch stemmodel te maken.",
    "pl": "Jestem właścicielem tego głosu i wyrażam zgodę na wykorzystanie go przez OpenAI w celu utworzenia syntetycznego modelu głosu.",
    "pt": "Eu sou o proprietário desta voz e autorizo o OpenAI a usá-la para criar um modelo de voz sintética.",
    "ru": "Я являюсь владельцем этого голоса и даю согласие OpenAI на использование этого голоса для создания модели синтетического голоса.",
    "uk": "Я є власником цього голосу і даю згоду OpenAI використовувати цей голос для створення синтетичної голосової моделі.",
    "vi": "Tôi là chủ sở hữu giọng nói này và tôi đồng ý cho OpenAI sử dụng giọng nói này để tạo mô hình giọng nói tổng hợp.",
    "zh": "我是此声音的拥有者并授权OpenAI使用此声音创建语音合成模型",
}

SUPPORTED_AUDIO_MIME_TYPES: dict[str, str] = {
    ".aac": "audio/aac",
    ".flac": "audio/flac",
    ".m4a": "audio/mp4",
    ".mp3": "audio/mpeg",
    ".mp4": "audio/mp4",
    ".mpeg": "audio/mpeg",
    ".oga": "audio/ogg",
    ".ogg": "audio/ogg",
    ".wav": "audio/x-wav",
    ".webm": "audio/webm",
}

DEFAULT_MODEL = "gpt-4o-mini-tts"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_TEST_TEXT = "안녕하세요. 이것은 내 목소리 기반 커스텀 보이스 테스트입니다."
MAX_CUSTOM_VOICE_SAMPLE_SECONDS = 30.0
MAX_CUSTOM_VOICE_UPLOAD_BYTES = 10 * 1024 * 1024


class OpenAICustomVoiceError(RuntimeError):
    pass


def validate_own_voice_confirmation(confirmed: bool) -> None:
    if not confirmed:
        raise ValueError(
            "본인 목소리와 명시적 동의 녹음만 사용할 수 있습니다. "
            "계속하려면 --confirm-own-voice 를 지정하세요."
        )


def mime_type_for_audio(path: Path) -> str:
    mime_type = SUPPORTED_AUDIO_MIME_TYPES.get(path.suffix.lower())
    if mime_type is None:
        supported = ", ".join(sorted(SUPPORTED_AUDIO_MIME_TYPES))
        raise ValueError(f"지원하지 않는 오디오 형식입니다: {path.suffix} (지원: {supported})")
    return mime_type


def require_existing_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{label} 파일을 찾을 수 없습니다: {path}")
    if not path.is_file():
        raise ValueError(f"{label} 경로가 파일이 아닙니다: {path}")
    return path


def wav_duration_seconds(path: Path) -> float | None:
    if path.suffix.lower() != ".wav":
        return None
    try:
        with wave.open(str(path), "rb") as wav_file:
            frames = wav_file.getnframes()
            frame_rate = wav_file.getframerate()
    except wave.Error:
        return None
    if frame_rate <= 0:
        return None
    return frames / float(frame_rate)


def validate_custom_voice_audio(path: Path, label: str) -> None:
    require_existing_file(path, label)
    mime_type_for_audio(path)
    size = path.stat().st_size
    if size > MAX_CUSTOM_VOICE_UPLOAD_BYTES:
        raise ValueError(
            f"{label} 파일이 {size} bytes 입니다. "
            f"OpenAI Custom Voice 업로드는 10MiB 이하여야 합니다."
        )
    duration = wav_duration_seconds(path)
    if duration is not None and duration > MAX_CUSTOM_VOICE_SAMPLE_SECONDS:
        raise ValueError(
            f"{label} WAV 길이가 {duration:.2f}초입니다. "
            f"OpenAI Custom Voice 샘플은 {MAX_CUSTOM_VOICE_SAMPLE_SECONDS:.0f}초 이하여야 합니다."
        )


def build_speech_payload(
    *,
    voice_id: str,
    text: str,
    model: str = DEFAULT_MODEL,
    language: str = "ko",
    output_format: str = "wav",
) -> dict[str, object]:
    if not voice_id.strip():
        raise ValueError("voice_id 가 비어 있습니다.")
    if not text.strip():
        raise ValueError("합성할 text 가 비어 있습니다.")
    return {
        "model": model,
        "voice": {"id": voice_id},
        "input": text,
        "language": language,
        "response_format": output_format,
    }


def _escape_form_value(value: str) -> str:
    if "\r" in value or "\n" in value:
        raise ValueError("multipart header 값에는 줄바꿈을 포함할 수 없습니다.")
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _encode_multipart(
    *,
    fields: dict[str, str],
    files: dict[str, Path],
) -> tuple[bytes, str]:
    boundary = f"----anti-deepvoice-guard-{uuid.uuid4().hex}"
    chunks: list[bytes] = []

    for name, value in fields.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{_escape_form_value(name)}"\r\n\r\n'.encode(),
                str(value).encode("utf-8"),
                b"\r\n",
            ]
        )

    for name, path in files.items():
        mime_type = mime_type_for_audio(path)
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                (
                    f'Content-Disposition: form-data; name="{_escape_form_value(name)}"; '
                    f'filename="{_escape_form_value(path.name)}"\r\n'
                ).encode(),
                f"Content-Type: {mime_type}\r\n\r\n".encode(),
                path.read_bytes(),
                b"\r\n",
            ]
        )

    chunks.append(f"--{boundary}--\r\n".encode())
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


class OpenAICustomVoiceClient:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout_seconds: float = 120.0,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise OpenAICustomVoiceError(
                "OPENAI_API_KEY 환경변수가 필요합니다. 키를 채팅에 붙여넣지 말고 로컬 셸에 설정하세요."
            )
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def create_voice_consent(self, *, name: str, language: str, recording: Path) -> dict[str, object]:
        validate_custom_voice_audio(recording, "consent recording")
        if language not in CONSENT_PHRASES:
            raise ValueError(f"지원하지 않는 consent language 입니다: {language}")
        return self._post_multipart_json(
            "/audio/voice_consents",
            fields={"name": name, "language": language},
            files={"recording": recording},
        )

    def create_voice(self, *, name: str, audio_sample: Path, consent_id: str) -> dict[str, object]:
        validate_custom_voice_audio(audio_sample, "sample recording")
        if not consent_id.strip():
            raise ValueError("consent_id 가 비어 있습니다.")
        return self._post_multipart_json(
            "/audio/voices",
            fields={"name": name, "consent": consent_id},
            files={"audio_sample": audio_sample},
        )

    def create_speech(
        self,
        *,
        voice_id: str,
        text: str,
        out_path: Path,
        model: str = DEFAULT_MODEL,
        language: str = "ko",
        output_format: str = "wav",
    ) -> Path:
        payload = build_speech_payload(
            voice_id=voice_id,
            text=text,
            model=model,
            language=language,
            output_format=output_format,
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        audio_bytes = self._post_json_bytes("/audio/speech", payload)
        out_path.write_bytes(audio_bytes)
        return out_path

    def _headers(self, content_type: str | None = None) -> dict[str, str]:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        if content_type:
            headers["Content-Type"] = content_type
        return headers

    def _post_multipart_json(
        self,
        endpoint: str,
        *,
        fields: dict[str, str],
        files: dict[str, Path],
    ) -> dict[str, object]:
        body, content_type = _encode_multipart(fields=fields, files=files)
        response = self._request(endpoint, body=body, headers=self._headers(content_type))
        return _decode_json_response(response)

    def _post_json_bytes(self, endpoint: str, payload: dict[str, object]) -> bytes:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        return self._request(endpoint, body=body, headers=self._headers("application/json"))

    def _request(self, endpoint: str, *, body: bytes, headers: dict[str, str]) -> bytes:
        request = Request(
            f"{self.base_url}{endpoint}",
            data=body,
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                return response.read()
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise OpenAICustomVoiceError(f"OpenAI API 요청 실패 ({exc.code}): {detail}") from exc
        except URLError as exc:
            raise OpenAICustomVoiceError(f"OpenAI API 연결 실패: {exc.reason}") from exc


def _decode_json_response(response: bytes) -> dict[str, object]:
    try:
        payload = json.loads(response.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise OpenAICustomVoiceError("OpenAI API 응답이 JSON이 아닙니다.") from exc
    if not isinstance(payload, dict):
        raise OpenAICustomVoiceError("OpenAI API 응답 JSON이 객체가 아닙니다.")
    return payload


def _extract_id(payload: dict[str, object], label: str) -> str:
    value = payload.get("id")
    if not isinstance(value, str) or not value:
        raise OpenAICustomVoiceError(f"{label} 응답에서 id 를 찾지 못했습니다: {payload}")
    return value


def _print_json(payload: dict[str, object]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def create_and_test(args: argparse.Namespace) -> int:
    validate_own_voice_confirmation(args.confirm_own_voice)
    client = OpenAICustomVoiceClient(timeout_seconds=args.timeout)

    consent = client.create_voice_consent(
        name=args.consent_name,
        language=args.language,
        recording=args.consent_recording,
    )
    consent_id = _extract_id(consent, "voice consent")
    print(f"consent_id={consent_id}")

    voice = client.create_voice(
        name=args.voice_name,
        audio_sample=args.sample_recording,
        consent_id=consent_id,
    )
    voice_id = _extract_id(voice, "voice")
    print(f"voice_id={voice_id}")

    out_path = client.create_speech(
        voice_id=voice_id,
        text=args.text,
        out_path=args.out,
        model=args.model,
        language=args.language,
        output_format=args.format,
    )
    print(f"output={out_path}")
    print("disclosure=AI-generated synthetic voice; research use only")
    return 0


def create_consent(args: argparse.Namespace) -> int:
    validate_own_voice_confirmation(args.confirm_own_voice)
    payload = OpenAICustomVoiceClient(timeout_seconds=args.timeout).create_voice_consent(
        name=args.name,
        language=args.language,
        recording=args.recording,
    )
    _print_json(payload)
    return 0


def create_voice(args: argparse.Namespace) -> int:
    validate_own_voice_confirmation(args.confirm_own_voice)
    payload = OpenAICustomVoiceClient(timeout_seconds=args.timeout).create_voice(
        name=args.name,
        audio_sample=args.audio_sample,
        consent_id=args.consent_id,
    )
    _print_json(payload)
    return 0


def synthesize(args: argparse.Namespace) -> int:
    validate_own_voice_confirmation(args.confirm_own_voice)
    out_path = OpenAICustomVoiceClient(timeout_seconds=args.timeout).create_speech(
        voice_id=args.voice_id,
        text=args.text,
        out_path=args.out,
        model=args.model,
        language=args.language,
        output_format=args.format,
    )
    print(f"output={out_path}")
    print("disclosure=AI-generated synthetic voice; research use only")
    return 0


def print_consent_phrase(args: argparse.Namespace) -> int:
    phrase = CONSENT_PHRASES.get(args.language)
    if phrase is None:
        languages = ", ".join(sorted(CONSENT_PHRASES))
        raise ValueError(f"지원하지 않는 language 입니다: {args.language} (지원: {languages})")
    print(phrase)
    return 0


def _add_common_api_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--timeout", type=float, default=120.0, help="OpenAI API 요청 timeout 초")


def _add_confirmation_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--confirm-own-voice",
        action="store_true",
        help="본인 목소리와 명시적 동의 녹음만 사용한다는 확인",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="OpenAI Custom Voice 생성 및 테스트 CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    phrase_parser = subparsers.add_parser("consent-phrase", help="동의 녹음용 공식 문구 출력")
    phrase_parser.add_argument("--language", default="ko", choices=sorted(CONSENT_PHRASES))
    phrase_parser.set_defaults(func=print_consent_phrase)

    consent_parser = subparsers.add_parser("create-consent", help="동의 녹음 업로드")
    consent_parser.add_argument("--name", required=True, help="consent 이름")
    consent_parser.add_argument("--language", default="ko", choices=sorted(CONSENT_PHRASES))
    consent_parser.add_argument("--recording", required=True, type=Path, help="동의 문구만 읽은 녹음 파일")
    _add_confirmation_arg(consent_parser)
    _add_common_api_args(consent_parser)
    consent_parser.set_defaults(func=create_consent)

    voice_parser = subparsers.add_parser("create-voice", help="custom voice 생성")
    voice_parser.add_argument("--name", required=True, help="voice 이름")
    voice_parser.add_argument("--audio-sample", required=True, type=Path, help="복제 기준 샘플 녹음")
    voice_parser.add_argument("--consent-id", required=True, help="create-consent 응답 id")
    _add_confirmation_arg(voice_parser)
    _add_common_api_args(voice_parser)
    voice_parser.set_defaults(func=create_voice)

    synth_parser = subparsers.add_parser("synthesize", help="생성된 custom voice로 TTS 생성")
    synth_parser.add_argument("--voice-id", required=True, help="voice_... 형식의 custom voice id")
    synth_parser.add_argument("--text", required=True, help="합성할 문장")
    synth_parser.add_argument("--out", required=True, type=Path, help="출력 오디오 경로")
    synth_parser.add_argument("--language", default="ko", help="합성 언어 코드")
    synth_parser.add_argument("--model", default=DEFAULT_MODEL, help="TTS 모델")
    synth_parser.add_argument("--format", default="wav", help="출력 포맷")
    _add_confirmation_arg(synth_parser)
    _add_common_api_args(synth_parser)
    synth_parser.set_defaults(func=synthesize)

    all_parser = subparsers.add_parser("create-and-test", help="consent 업로드부터 테스트 음성 생성까지 실행")
    all_parser.add_argument("--consent-recording", required=True, type=Path, help="동의 문구만 읽은 녹음 파일")
    all_parser.add_argument("--sample-recording", required=True, type=Path, help="복제 기준 샘플 녹음")
    all_parser.add_argument("--voice-name", required=True, help="생성할 custom voice 이름")
    all_parser.add_argument("--consent-name", help="consent 이름. 기본값은 voice-name + _consent")
    all_parser.add_argument("--language", default="ko", choices=sorted(CONSENT_PHRASES))
    all_parser.add_argument("--text", default=DEFAULT_TEST_TEXT, help="테스트 합성 문장")
    all_parser.add_argument("--out", required=True, type=Path, help="테스트 출력 오디오 경로")
    all_parser.add_argument("--model", default=DEFAULT_MODEL, help="TTS 모델")
    all_parser.add_argument("--format", default="wav", help="출력 포맷")
    _add_confirmation_arg(all_parser)
    _add_common_api_args(all_parser)
    all_parser.set_defaults(func=create_and_test)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "consent_name", None) is None and getattr(args, "voice_name", None):
        args.consent_name = f"{args.voice_name}_consent"
    try:
        return args.func(args)
    except (OSError, ValueError, OpenAICustomVoiceError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
