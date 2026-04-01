"""
데이터셋 준비 스크립트
real/fake 오디오 스캔 → 16kHz mono WAV 변환 → train/val CSV 분할
"""

import argparse
import csv
import random
import subprocess
from pathlib import Path


def convert_to_wav16k(src: Path, dst: Path) -> bool:
    """ffmpeg로 16kHz mono WAV 변환"""
    if dst.exists():
        return True
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(src),
                "-ar", "16000", "-ac", "1", "-f", "wav",
                str(dst),
            ],
            capture_output=True,
            timeout=30,
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"  Failed to convert {src.name}: {e}")
        return False


def scan_and_prepare(data_dir: Path, test_samples_dir: Path | None) -> list[dict]:
    """오디오 파일 스캔, 변환, 메타데이터 반환"""
    real_dir = data_dir / "real"
    fake_dir = data_dir / "fake"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    # test-samples에서 real 데이터 복사 (아직 없으면)
    if test_samples_dir and test_samples_dir.exists():
        for ext in ("*.wav", "*.m4a", "*.mp3", "*.flac"):
            for src in (test_samples_dir / "real").glob(ext):
                dst = real_dir / src.with_suffix(".wav").name
                if not dst.exists():
                    print(f"  Converting {src.name} → {dst.name}")
                    convert_to_wav16k(src, dst)

    entries = []

    # Real 파일 스캔
    for wav in sorted(real_dir.glob("*.wav")):
        entries.append({
            "filepath": str(wav.resolve()),
            "label": 1,  # bonafide = 1 (ASVspoof convention)
            "category": "real",
        })

    # Fake 파일 스캔
    for wav in sorted(fake_dir.glob("*.wav")):
        entries.append({
            "filepath": str(wav.resolve()),
            "label": 0,  # spoof = 0 (ASVspoof convention)
            "category": "fake",
        })

    return entries


def split_and_save(entries: list[dict], data_dir: Path, val_ratio: float, seed: int):
    """train/val 분할 후 CSV 저장"""
    random.seed(seed)

    real_entries = [e for e in entries if e["category"] == "real"]
    fake_entries = [e for e in entries if e["category"] == "fake"]

    random.shuffle(real_entries)
    random.shuffle(fake_entries)

    # 각 카테고리별로 분할하여 비율 유지
    def split(items):
        n_val = max(1, int(len(items) * val_ratio))
        return items[n_val:], items[:n_val]

    real_train, real_val = split(real_entries)
    fake_train, fake_val = split(fake_entries)

    train = real_train + fake_train
    val = real_val + fake_val
    random.shuffle(train)
    random.shuffle(val)

    for name, data in [("train.csv", train), ("val.csv", val)]:
        path = data_dir / name
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["filepath", "label"])
            writer.writeheader()
            for entry in data:
                writer.writerow({"filepath": entry["filepath"], "label": entry["label"]})
        print(f"  {name}: {len(data)} samples")

    return train, val


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for AASIST fine-tuning")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
    )
    parser.add_argument(
        "--test-samples",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "test-samples",
        help="Path to test-samples directory (real recordings copied from here)",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=== Scanning and preparing audio files ===")
    entries = scan_and_prepare(args.data_dir, args.test_samples)

    real_count = sum(1 for e in entries if e["category"] == "real")
    fake_count = sum(1 for e in entries if e["category"] == "fake")
    print(f"\nTotal: {len(entries)} files (real={real_count}, fake={fake_count})")

    if real_count < 5:
        print(f"\n*** WARNING: real 데이터가 {real_count}개뿐입니다. ***")
        print("  더 나은 결과를 위해 data/real/ 에 핸드폰 녹음 20~30개를 추가하세요.")
        print("  추가 후 이 스크립트를 다시 실행하면 됩니다.\n")

    print("\n=== Splitting train/val ===")
    train, val = split_and_save(entries, args.data_dir, args.val_ratio, args.seed)

    # 통계
    print("\n=== Dataset Statistics ===")
    for name, data in [("Train", train), ("Val", val)]:
        real_n = sum(1 for e in data if e["label"] == 1)
        fake_n = sum(1 for e in data if e["label"] == 0)
        print(f"  {name}: {len(data)} total (real={real_n}, fake={fake_n})")


if __name__ == "__main__":
    main()
