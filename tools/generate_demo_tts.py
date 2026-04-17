#!/usr/bin/env python3
"""Demo 시나리오 #2, #4용 TTS 음성 생성.

각 시나리오의 transcript 텍스트를 그대로 읽어 edge-tts(한국어 Neural)로 mp3 생성 후,
ffmpeg로 16kHz mono WAV로 변환. 결과 파일은 demo_02.wav, demo_04.wav.
"""
import asyncio
import subprocess
from pathlib import Path

import edge_tts

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "android-app/app/src/main/assets/demo"

SCENARIOS = [
    ("demo_02", "ko-KR-SunHiNeural"),   # TTS 일상 (여성 음성)
    ("demo_04", "ko-KR-InJoonNeural"),  # TTS 피싱 (남성 음성)
]


async def synthesize(text: str, voice: str, out_mp3: Path) -> None:
    communicate = edge_tts.Communicate(text, voice, rate="+0%", volume="+0%")
    await communicate.save(str(out_mp3))


def to_16k_mono_wav(mp3: Path, wav: Path) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(mp3),
            "-ar", "16000",
            "-ac", "1",
            "-sample_fmt", "s16",
            str(wav),
        ],
        check=True,
        capture_output=True,
    )


async def main() -> None:
    for base, voice in SCENARIOS:
        transcript_path = ASSETS / f"{base}_transcript.txt"
        wav_path = ASSETS / f"{base}.wav"
        mp3_path = ASSETS / f"{base}.mp3"

        text = transcript_path.read_text(encoding="utf-8").strip()
        print(f"[{base}] voice={voice}, text={text[:40]}...")

        await synthesize(text, voice, mp3_path)
        to_16k_mono_wav(mp3_path, wav_path)
        mp3_path.unlink(missing_ok=True)

        size = wav_path.stat().st_size
        print(f"[{base}] wrote {wav_path.name} ({size:,} bytes)")


if __name__ == "__main__":
    asyncio.run(main())
