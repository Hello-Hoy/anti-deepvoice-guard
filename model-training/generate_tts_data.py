"""
Edge TTS를 사용한 fake 음성 데이터 자동 생성
다양한 TTS 음성으로 fake 샘플을 생성하여 fine-tuning 데이터로 사용
"""

import argparse
import asyncio
import random
import subprocess
import tempfile
from pathlib import Path

import edge_tts

VOICES = {
    "ko": [
        "ko-KR-SunHiNeural",
        "ko-KR-InJoonNeural",
        "ko-KR-HyunsuNeural",
    ],
    "en": [
        "en-US-JennyNeural",
        "en-US-GuyNeural",
        "en-US-AriaNeural",
        "en-US-DavisNeural",
    ],
}


def load_corpus(corpus_path: Path) -> dict[str, list[str]]:
    """텍스트 코퍼스 로드. 빈 줄/주석 제외, 언어별 분류."""
    lines = corpus_path.read_text(encoding="utf-8").strip().splitlines()
    texts = {"ko": [], "en": []}
    current_lang = "ko"
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            if "English" in line or "english" in line:
                current_lang = "en"
            else:
                current_lang = "ko"
            continue
        texts[current_lang].append(line)
    return texts


async def generate_one(text: str, voice: str, output_path: Path) -> bool:
    """단일 TTS 음성 생성 (MP3 → 16kHz mono WAV)"""
    try:
        communicate = edge_tts.Communicate(text, voice)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        await communicate.save(str(tmp_path))

        # ffmpeg로 16kHz mono WAV 변환
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(tmp_path),
                "-ar", "16000", "-ac", "1", "-f", "wav",
                str(output_path),
            ],
            capture_output=True,
            timeout=30,
        )
        tmp_path.unlink(missing_ok=True)

        if result.returncode != 0:
            print(f"  ffmpeg error: {result.stderr.decode()[:200]}")
            return False
        return True
    except Exception as e:
        print(f"  Error: {e}")
        return False


async def main_async(args):
    corpus = load_corpus(args.corpus)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    ko_texts = corpus["ko"]
    en_texts = corpus["en"]
    all_voices_ko = VOICES["ko"]
    all_voices_en = VOICES["en"]

    print(f"Corpus: {len(ko_texts)} Korean, {len(en_texts)} English texts")
    print(f"Voices: {len(all_voices_ko)} Korean, {len(all_voices_en)} English")
    print(f"Target: {args.count} samples")
    print()

    # 한국어 80%, 영어 20% 비율로 생성
    ko_count = int(args.count * 0.8)
    en_count = args.count - ko_count

    tasks = []
    idx = 0

    for i in range(ko_count):
        text = ko_texts[i % len(ko_texts)]
        voice = all_voices_ko[i % len(all_voices_ko)]
        out_path = output_dir / f"fake_ko_{idx:04d}.wav"
        tasks.append((text, voice, out_path))
        idx += 1

    for i in range(en_count):
        text = en_texts[i % len(en_texts)]
        voice = all_voices_en[i % len(all_voices_en)]
        out_path = output_dir / f"fake_en_{idx:04d}.wav"
        tasks.append((text, voice, out_path))
        idx += 1

    # 이미 존재하는 파일 스킵
    new_tasks = [(t, v, p) for t, v, p in tasks if not p.exists()]
    if len(new_tasks) < len(tasks):
        print(f"Skipping {len(tasks) - len(new_tasks)} existing files")
    print(f"Generating {len(new_tasks)} new files...")

    success = 0
    fail = 0
    # 동시 요청 제한 (rate limiting)
    semaphore = asyncio.Semaphore(args.concurrency)

    async def gen_with_limit(text, voice, path):
        async with semaphore:
            return await generate_one(text, voice, path)

    batch_size = 20
    for batch_start in range(0, len(new_tasks), batch_size):
        batch = new_tasks[batch_start : batch_start + batch_size]
        results = await asyncio.gather(
            *[gen_with_limit(t, v, p) for t, v, p in batch]
        )
        for ok, (_, voice, path) in zip(results, batch):
            if ok:
                success += 1
            else:
                fail += 1
        print(f"  Progress: {batch_start + len(batch)}/{len(new_tasks)} "
              f"(success={success}, fail={fail})")

    total_files = len(list(output_dir.glob("*.wav")))
    print(f"\nDone! {success} new files generated ({fail} failed)")
    print(f"Total files in {output_dir}: {total_files}")


def main():
    parser = argparse.ArgumentParser(description="Generate TTS fake audio data")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "text_corpus.txt",
        help="Path to text corpus file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "fake",
        help="Output directory for generated WAV files",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=150,
        help="Number of samples to generate (default: 150)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Max concurrent TTS requests (default: 5)",
    )
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
