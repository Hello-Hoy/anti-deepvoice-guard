#!/usr/bin/env python3
"""
다양한 TTS 엔진으로 fake 음성 데이터 대량 생성

엔진:
  1. Edge TTS - 한국어 3 음성 + Multilingual 음성
  2. gTTS (Google) - 한국어 1 음성

다양성 확보:
  - 여러 TTS 엔진/음성
  - 속도 변형 (±10%, ±20%)
  - 다양한 텍스트 (235+ 문장)

Usage:
  python generate_multi_tts.py --count 9000
  python generate_multi_tts.py --count 9000 --resume  # 이어서 생성
"""

import argparse
import asyncio
import random
import subprocess
import tempfile
from pathlib import Path

import edge_tts

# ─────────────── 음성 정의 ───────────────

EDGE_VOICES_KO = [
    "ko-KR-SunHiNeural",
    "ko-KR-InJoonNeural",
    "ko-KR-HyunsuMultilingualNeural",
]

# Multilingual 음성 (한국어 텍스트 발화 가능 - 다양한 억양/특성)
EDGE_VOICES_MULTI = [
    "en-US-AndrewMultilingualNeural",
    "en-US-AvaMultilingualNeural",
    "en-US-BrianMultilingualNeural",
    "en-US-EmmaMultilingualNeural",
    "en-AU-WilliamMultilingualNeural",
    "fr-FR-RemyMultilingualNeural",
    "fr-FR-VivienneMultilingualNeural",
    "de-DE-FlorianMultilingualNeural",
]

EDGE_VOICES_EN = [
    "en-US-JennyNeural",
    "en-US-GuyNeural",
    "en-US-AriaNeural",
    "en-US-DavisNeural",
]

# 속도 변형 (Edge TTS rate 파라미터)
RATE_VARIANTS = ["-20%", "-10%", "+0%", "+10%", "+20%"]


# ─────────────── 텍스트 로드 ───────────────

def load_corpus(path: Path) -> dict[str, list[str]]:
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    texts = {"ko": [], "en": []}
    lang = "ko"
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            lang = "en" if "English" in line or "english" in line else "ko"
            continue
        texts[lang].append(line)
    return texts


# ─────────────── TTS 생성 함수 ───────────────

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


def gtts_generate(text: str, output: Path, slow: bool = False) -> bool:
    try:
        from gtts import gTTS
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        tts = gTTS(text=text, lang="ko", slow=slow)
        tts.save(str(tmp_path))
        result = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-i", str(tmp_path), "-ar", "16000", "-ac", "1", "-f", "wav", str(output)],
            capture_output=True, timeout=30,
        )
        tmp_path.unlink(missing_ok=True)
        return result.returncode == 0
    except Exception:
        return False


# ─────────────── 작업 목록 생성 ───────────────

def build_task_list(texts: dict, target_count: int) -> list[dict]:
    """다양한 TTS 조합의 작업 목록 생성"""
    tasks = []
    ko_texts = texts["ko"]
    en_texts = texts["en"]
    idx = 0

    # 1. Edge TTS 한국어 음성 × 속도 변형 (주력)
    for voice in EDGE_VOICES_KO:
        for rate in RATE_VARIANTS:
            for text in ko_texts:
                tasks.append({
                    "engine": "edge", "text": text, "voice": voice,
                    "rate": rate, "idx": idx, "lang": "ko",
                })
                idx += 1

    # 2. Edge TTS Multilingual 음성 (한국어 텍스트)
    for voice in EDGE_VOICES_MULTI:
        for text in ko_texts:
            tasks.append({
                "engine": "edge", "text": text, "voice": voice,
                "rate": "+0%", "idx": idx, "lang": "ko",
            })
            idx += 1

    # 3. gTTS (Google) 한국어
    for text in ko_texts:
        tasks.append({
            "engine": "gtts", "text": text, "voice": "gtts_ko",
            "rate": "normal", "idx": idx, "lang": "ko",
        })
        idx += 1
    for text in ko_texts:
        tasks.append({
            "engine": "gtts", "text": text, "voice": "gtts_ko_slow",
            "rate": "slow", "idx": idx, "lang": "ko",
        })
        idx += 1

    # 4. Edge TTS 영어 음성 (영어 텍스트)
    for voice in EDGE_VOICES_EN:
        for text in en_texts:
            tasks.append({
                "engine": "edge", "text": text, "voice": voice,
                "rate": "+0%", "idx": idx, "lang": "en",
            })
            idx += 1

    # 셔플 후 target_count만큼 자르기
    random.seed(42)
    random.shuffle(tasks)

    if len(tasks) > target_count:
        tasks = tasks[:target_count]

    # idx 재할당
    for i, t in enumerate(tasks):
        t["idx"] = i

    return tasks


# ─────────────── 메인 생성 루프 ───────────────

async def generate_all(tasks: list[dict], output_dir: Path, concurrency: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(concurrency)

    # 이미 존재하는 파일 스킵
    existing = set(f.name for f in output_dir.glob("fake_gen_*.wav"))
    new_tasks = [t for t in tasks if f"fake_gen_{t['idx']:05d}.wav" not in existing]

    if len(new_tasks) < len(tasks):
        print(f"  기존 파일 스킵: {len(tasks) - len(new_tasks)}개")
    print(f"  생성 대상: {len(new_tasks)}개\n")

    success = 0
    fail = 0

    async def process_one(task):
        nonlocal success, fail
        out_path = output_dir / f"fake_gen_{task['idx']:05d}.wav"

        async with semaphore:
            if task["engine"] == "edge":
                ok = await edge_tts_generate(task["text"], task["voice"], out_path, task["rate"])
            elif task["engine"] == "gtts":
                # gTTS는 동기 함수 → executor에서 실행
                loop = asyncio.get_event_loop()
                slow = task["rate"] == "slow"
                ok = await loop.run_in_executor(None, gtts_generate, task["text"], out_path, slow)
            else:
                ok = False

            if ok:
                success += 1
            else:
                fail += 1

    # 배치 처리
    batch_size = 50
    for batch_start in range(0, len(new_tasks), batch_size):
        batch = new_tasks[batch_start:batch_start + batch_size]
        await asyncio.gather(*[process_one(t) for t in batch])
        total = batch_start + len(batch)
        print(f"  진행: {total}/{len(new_tasks)} (성공={success}, 실패={fail})")

    total_files = len(list(output_dir.glob("fake_gen_*.wav")))
    print(f"\n완료! 신규 생성={success}, 실패={fail}")
    print(f"총 fake_gen 파일: {total_files}개")
    return total_files


def main():
    parser = argparse.ArgumentParser(description="멀티 TTS fake 음성 대량 생성")
    parser.add_argument("--corpus", type=Path,
                        default=Path(__file__).resolve().parent / "data" / "text_corpus_large.txt")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "data" / "fake")
    parser.add_argument("--count", type=int, default=9000, help="생성할 샘플 수")
    parser.add_argument("--concurrency", type=int, default=10, help="동시 생성 수")
    args = parser.parse_args()

    texts = load_corpus(args.corpus)
    print(f"코퍼스: 한국어 {len(texts['ko'])}문장, 영어 {len(texts['en'])}문장")

    tasks = build_task_list(texts, args.count)

    # 통계
    engines = {}
    for t in tasks:
        key = f"{t['engine']}_{t['voice']}"
        engines[key] = engines.get(key, 0) + 1
    print(f"\n생성 계획: 총 {len(tasks)}개")
    print(f"  엔진별 분포:")
    for k, v in sorted(engines.items(), key=lambda x: -x[1])[:10]:
        print(f"    {k}: {v}개")
    if len(engines) > 10:
        print(f"    ... 외 {len(engines)-10}개 음성")
    print()

    asyncio.run(generate_all(tasks, args.output_dir, args.concurrency))


if __name__ == "__main__":
    main()
