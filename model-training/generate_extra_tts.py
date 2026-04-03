#!/usr/bin/env python3
"""
오탐 케이스(edge_ko_hyunsu, edge_ko_injoon) 전용 추가 fake 데이터 생성.
다양한 속도/텍스트 조합으로 이 음성들의 fake 패턴을 학습할 수 있도록 보강.

Usage:
  python generate_extra_tts.py --count 500
"""

import argparse
import asyncio
import random
import subprocess
import tempfile
from pathlib import Path

import edge_tts

SCRIPT_DIR = Path(__file__).resolve().parent

# 오탐 대상 음성 (한국어 고품질 TTS)
TARGET_VOICES = [
    "ko-KR-HyunsuMultilingualNeural",
    "ko-KR-InJoonNeural",
]

# 속도 변형 (더 다양하게)
RATE_VARIANTS = ["-30%", "-20%", "-10%", "+0%", "+10%", "+20%", "+30%"]


def load_texts(corpus_path: Path) -> list[str]:
    lines = corpus_path.read_text(encoding="utf-8").strip().splitlines()
    texts = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if all(ord(c) < 128 for c in line.replace(" ", "").replace(",", "").replace(".", "")):
            continue
        texts.append(line)
    return texts


async def edge_tts_generate(text: str, voice: str, output: Path, rate: str = "+0%") -> bool:
    try:
        comm = edge_tts.Communicate(text, voice, rate=rate)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        await comm.save(str(tmp_path))
        result = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-i", str(tmp_path), "-ar", "16000", "-ac", "1", "-f", "wav", str(output)],
            capture_output=True, timeout=30,
        )
        tmp_path.unlink(missing_ok=True)
        return result.returncode == 0
    except Exception:
        return False


async def main():
    parser = argparse.ArgumentParser(description="오탐 TTS 추가 fake 생성")
    parser.add_argument("--corpus", type=Path,
                        default=SCRIPT_DIR / "data" / "text_corpus_large.txt")
    parser.add_argument("--output-dir", type=Path,
                        default=SCRIPT_DIR / "data" / "fake")
    parser.add_argument("--count", type=int, default=500, help="생성할 샘플 수")
    parser.add_argument("--concurrency", type=int, default=5)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    texts = load_texts(args.corpus)
    print(f"텍스트: {len(texts)}문장")

    # 작업 목록 생성
    tasks = []
    idx = 0
    for voice in TARGET_VOICES:
        for rate in RATE_VARIANTS:
            for text in texts:
                tasks.append({"voice": voice, "rate": rate, "text": text, "idx": idx})
                idx += 1

    random.seed(42)
    random.shuffle(tasks)
    tasks = tasks[:args.count]
    for i, t in enumerate(tasks):
        t["idx"] = i

    # 기존 파일 스킵
    existing = set(f.name for f in args.output_dir.glob("fake_extra_*.wav"))
    new_tasks = [t for t in tasks if f"fake_extra_{t['idx']:05d}.wav" not in existing]
    if len(new_tasks) < len(tasks):
        print(f"기존 파일 스킵: {len(tasks) - len(new_tasks)}개")
    print(f"생성 대상: {len(new_tasks)}개 (음���: {', '.join(TARGET_VOICES)})\n")

    # 통계
    for v in TARGET_VOICES:
        cnt = sum(1 for t in new_tasks if t["voice"] == v)
        print(f"  {v}: {cnt}개")
    print()

    semaphore = asyncio.Semaphore(args.concurrency)
    success = 0
    fail = 0

    async def process(task):
        nonlocal success, fail
        out = args.output_dir / f"fake_extra_{task['idx']:05d}.wav"
        async with semaphore:
            ok = await edge_tts_generate(task["text"], task["voice"], out, task["rate"])
            if ok:
                success += 1
            else:
                fail += 1

    batch_size = 50
    for batch_start in range(0, len(new_tasks), batch_size):
        batch = new_tasks[batch_start:batch_start + batch_size]
        await asyncio.gather(*[process(t) for t in batch])
        total = batch_start + len(batch)
        print(f"  진행: {total}/{len(new_tasks)} (성공={success}, 실패={fail})")

    print(f"\n완료! 성공={success}, 실패={fail}")
    total_files = len(list(args.output_dir.glob("fake_extra_*.wav")))
    print(f"총 fake_extra 파일: {total_files}개")


if __name__ == "__main__":
    asyncio.run(main())
