#!/usr/bin/env python3
"""
XTTS v2 음성 클로닝으로 fake 데이터 생성

AIHub real 음성을 레퍼런스로 사용하여 특정인 목소리를 클로닝한 fake 데이터 생성.
실제 보이스피싱/딥보이스 위협을 시뮬레이션.

Usage:
  COQUI_TOS_AGREED=1 python generate_clone_data.py --count 3000
"""

import argparse
import os
import random
import subprocess
import time
from pathlib import Path

os.environ["COQUI_TOS_AGREED"] = "1"

import warnings
warnings.filterwarnings("ignore")

from TTS.api import TTS

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"


def load_texts(corpus_path: Path) -> list[str]:
    lines = corpus_path.read_text(encoding="utf-8").strip().splitlines()
    texts = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # 영어 섹션 건너뜀 (XTTS 한국어 클로닝)
        if all(ord(c) < 128 for c in line.replace(" ", "").replace(",", "").replace(".", "")):
            continue
        texts.append(line)
    return texts


def select_reference_voices(real_dir: Path, n: int = 30) -> list[Path]:
    """다양한 레퍼런스 음성 선택 (서로 다른 화자)"""
    all_real = sorted(real_dir.glob("*.wav"))
    # 서로 다른 데이터셋에서 골고루 선택
    groups = {}
    for f in all_real:
        prefix = f.name.split("_")[1] if "_" in f.name else "other"
        groups.setdefault(prefix, []).append(f)

    selected = []
    for prefix, files in groups.items():
        random.seed(42)
        sample = random.sample(files, min(n // len(groups) + 1, len(files)))
        selected.extend(sample)

    random.shuffle(selected)
    return selected[:n]


def convert_to_16k(src: Path, dst: Path) -> bool:
    result = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-i", str(src), "-ar", "16000", "-ac", "1", "-f", "wav", str(dst)],
        capture_output=True, timeout=30,
    )
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="XTTS v2 음성 클로닝 fake 데이터 생성")
    parser.add_argument("--corpus", type=Path, default=DATA_DIR / "text_corpus_large.txt")
    parser.add_argument("--real-dir", type=Path, default=DATA_DIR / "real")
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR / "fake")
    parser.add_argument("--count", type=int, default=3000)
    parser.add_argument("--ref-voices", type=int, default=30, help="레퍼런스 음성 수")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 텍스트 로드
    texts = load_texts(args.corpus)
    print(f"텍스트: {len(texts)}문장")

    # 레퍼런스 음성 선택
    refs = select_reference_voices(args.real_dir, args.ref_voices)
    print(f"레퍼런스 음성: {len(refs)}개")
    for r in refs[:5]:
        print(f"  {r.name}")
    print(f"  ...")

    # XTTS 모델 로드
    print("\nXTTS v2 모델 로드 중...")
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
    print("모델 준비 완료\n")

    # 작업 생성
    tasks = []
    for i in range(args.count):
        text = texts[i % len(texts)]
        ref = refs[i % len(refs)]
        tasks.append({"text": text, "ref": ref, "idx": i})

    # 기존 파일 스킵
    existing = set(f.name for f in args.output_dir.glob("fake_clone_*.wav"))
    new_tasks = [t for t in tasks if f"fake_clone_{t['idx']:05d}.wav" not in existing]
    if len(new_tasks) < len(tasks):
        print(f"기존 파일 스킵: {len(tasks) - len(new_tasks)}개")
    print(f"생성 대상: {len(new_tasks)}개\n")

    success = 0
    fail = 0
    t0 = time.time()

    for i, task in enumerate(new_tasks):
        tmp_path = args.output_dir / f"fake_clone_{task['idx']:05d}_tmp.wav"
        out_path = args.output_dir / f"fake_clone_{task['idx']:05d}.wav"

        try:
            tts.tts_to_file(
                text=task["text"],
                speaker_wav=str(task["ref"]),
                language="ko",
                file_path=str(tmp_path),
            )
            # 24kHz → 16kHz 변환
            if convert_to_16k(tmp_path, out_path):
                success += 1
            else:
                fail += 1
            tmp_path.unlink(missing_ok=True)
        except Exception as e:
            fail += 1
            tmp_path.unlink(missing_ok=True)
            if fail <= 3:
                print(f"  에러: {e}")

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(new_tasks) - i - 1) / rate / 60
            print(f"  진행: {i+1}/{len(new_tasks)} (성공={success}, 실패={fail}) "
                  f"[{rate:.1f}개/s, 남은시간: {eta:.0f}분]")

    elapsed = time.time() - t0
    print(f"\n완료! 성공={success}, 실패={fail}, 소요시간={elapsed/60:.1f}분")
    total = len(list(args.output_dir.glob("fake_clone_*.wav")))
    print(f"총 clone fake 파일: {total}개")


if __name__ == "__main__":
    main()
