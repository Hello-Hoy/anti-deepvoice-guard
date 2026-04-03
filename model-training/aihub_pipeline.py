#!/usr/bin/env python3
"""
AIHub 전체 파이프라인: 다운로드 → 전처리 → Fine-tuning

Usage:
  python aihub_pipeline.py --api-key YOUR_KEY
  python aihub_pipeline.py --api-key YOUR_KEY --skip-download   # 이미 다운받은 경우
  python aihub_pipeline.py --api-key YOUR_KEY --skip-process    # 이미 WAV 변환한 경우
  python aihub_pipeline.py --api-key YOUR_KEY --skip-train      # 학습만 건너뜀 (테스트용)
"""

import argparse
import csv
import os
import random
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

# ─────────────────────────── 경로 설정 ───────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
REAL_DIR = DATA_DIR / "real"
FAKE_DIR = DATA_DIR / "fake"
DOWNLOAD_DIR = DATA_DIR / "aihub_downloads"
EXTRACT_DIR = DATA_DIR / "aihub_extracted"
AIHUBSHELL = Path.home() / "aihubshell"

# ─────────────────────── 다운로드 대상 정의 ───────────────────────
# real 3종: 다양한 화자 / 환경 / 음질
# fake 1종: 다화자 TTS 합성음
DATASETS = [
    {
        "name": "real_537_common",
        "datasetkey": "537",
        "filekeys": "55154",          # TS_common_01.zip (21GB)
        "label": "real",
        "max_files": 3000,
        "desc": "화자인식용 common 음성 (21GB)",
    },
    {
        "name": "real_571_telephone",
        "datasetkey": "571",
        "filekeys": "69389",          # TS_D03.zip (19GB, 8kHz)
        "label": "real",
        "max_files": 3000,
        "desc": "저음질 전화망 D03 (19GB)",
    },
    {
        "name": "real_123_kspon01",
        "datasetkey": "123",
        "filekeys": "50211",          # KsponSpeech_01.zip (14GB)
        "label": "real",
        "max_files": 3000,
        "desc": "KsponSpeech_01 (14GB)",
    },
    {
        "name": "fake_542_ts4",
        "datasetkey": "542",
        "filekeys": "61733",          # TS4.zip (45GB)
        "label": "fake",
        "max_files": 9000,
        "desc": "다화자 음성합성 TS4 (45GB)",
    },
]

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".pcm", ".ogg"}


# ─────────────────────────── 유틸 ───────────────────────────────

def run(cmd: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    return subprocess.run(cmd, cwd=cwd, check=check, text=True,
                          capture_output=False)


def count_files(directory: Path, exts: set[str]) -> int:
    return sum(1 for f in directory.rglob("*") if f.suffix.lower() in exts)


def find_audio_files(directory: Path, exts: set[str]) -> list[Path]:
    return [f for f in directory.rglob("*") if f.suffix.lower() in exts]


def convert_to_wav16k(src: Path, dst: Path) -> bool:
    """ffmpeg로 16kHz mono WAV 변환. PCM(.pcm) 파일은 s16le로 처리."""
    if dst.exists():
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)

    if src.suffix.lower() == ".pcm":
        # KsponSpeech 등: 16kHz 16-bit mono raw PCM
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "s16le", "-ar", "16000", "-ac", "1",
            "-i", str(src),
            "-ar", "16000", "-ac", "1", str(dst),
        ]
    else:
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(src),
            "-ar", "16000", "-ac", "1", str(dst),
        ]

    result = subprocess.run(cmd, capture_output=True, timeout=60)
    return result.returncode == 0


# ─────────────────────────── 다운로드 ───────────────────────────

def download_dataset(ds: dict, api_key: str) -> Path:
    """aihubshell로 다운로드. 이미 zip이 있으면 스킵."""
    dl_dir = DOWNLOAD_DIR / ds["name"]
    dl_dir.mkdir(parents=True, exist_ok=True)

    # 이미 zip 파일이 있는지 확인 (aihubshell은 중첩 디렉토리 안에 zip을 생성)
    existing_zips = list(dl_dir.rglob("*.zip"))
    if existing_zips:
        print(f"  [SKIP] 이미 다운로드됨: {existing_zips[0].name}")
        return dl_dir

    print(f"\n{'='*60}")
    print(f"다운로드: {ds['desc']}")
    print(f"{'='*60}")
    run(
        [
            str(AIHUBSHELL),
            "-mode", "d",
            "-datasetkey", ds["datasetkey"],
            "-filekey", ds["filekeys"],
            "-aihubapikey", api_key,
        ],
        cwd=dl_dir,
    )
    return dl_dir


# ──────────────────────── 압축 해제 ─────────────────────────────

def extract_dataset(dl_dir: Path, ds_name: str) -> Path:
    """zip 파일 압축 해제. 이미 해제된 경우 스킵."""
    ext_dir = EXTRACT_DIR / ds_name
    done_marker = ext_dir / ".extracted"

    if done_marker.exists():
        print(f"  [SKIP] 이미 압축 해제됨: {ext_dir}")
        return ext_dir

    ext_dir.mkdir(parents=True, exist_ok=True)
    zips = list(dl_dir.rglob("*.zip"))
    if not zips:
        raise FileNotFoundError(f"zip 파일 없음: {dl_dir}")

    for zp in zips:
        print(f"  압축 해제: {zp.name} ({zp.stat().st_size / 1e9:.1f}GB) → {ext_dir}")
        # aihubshell은 2GB 분할 zip을 단순 연결 → 7z가 가장 잘 처리
        run(["7z", "x", str(zp), f"-o{ext_dir}", "-y"], check=False)

    done_marker.touch()
    return ext_dir


# ─────────────────────── 오디오 샘플링+변환 ──────────────────────

def process_dataset(ext_dir: Path, ds: dict) -> int:
    """오디오 파일 샘플링 → 16kHz WAV 변환 → real/fake 디렉토리에 저장"""
    target_dir = REAL_DIR if ds["label"] == "real" else FAKE_DIR
    prefix = ds["name"]
    done_marker = target_dir / f".done_{prefix}"

    if done_marker.exists():
        existing = list(target_dir.glob(f"{prefix}_*.wav"))
        print(f"  [SKIP] 이미 처리됨 ({len(existing)}개): {prefix}")
        return len(existing)

    print(f"\n오디오 파일 스캔 중: {ext_dir} ...")
    all_audio = find_audio_files(ext_dir, AUDIO_EXTS)
    print(f"  발견: {len(all_audio)}개")

    if not all_audio:
        print(f"  [WARNING] 오디오 파일 없음: {ext_dir}")
        return 0

    # 랜덤 샘플링
    max_n = ds["max_files"]
    random.seed(42)
    sampled = random.sample(all_audio, min(max_n, len(all_audio)))
    print(f"  샘플링: {len(sampled)}개 (max={max_n})")

    converted = 0
    failed = 0
    for i, src in enumerate(sampled):
        dst_name = f"{prefix}_{i:05d}.wav"
        dst = target_dir / dst_name

        ok = convert_to_wav16k(src, dst)
        if ok:
            converted += 1
        else:
            failed += 1

        if (i + 1) % 500 == 0:
            print(f"  진행: {i+1}/{len(sampled)} (성공={converted}, 실패={failed})")

    print(f"  완료: 변환 성공={converted}, 실패={failed}")
    done_marker.touch()

    # 압축 해제 디렉토리 삭제 (공간 확보)
    print(f"  임시 디렉토리 삭제: {ext_dir}")
    shutil.rmtree(ext_dir, ignore_errors=True)

    return converted


# ───────────────────────── CSV 재구성 ────────────────────────────

def rebuild_csv(val_ratio: float = 0.15, seed: int = 42):
    """real/fake WAV 파일 전체 스캔 → train.csv / val.csv 재구성"""
    print("\n=== CSV 재구성 ===")
    random.seed(seed)

    real_files = sorted(REAL_DIR.glob("*.wav"))
    fake_files = sorted(FAKE_DIR.glob("*.wav"))
    print(f"  Real: {len(real_files)}개, Fake: {len(fake_files)}개")

    def split(files):
        files = list(files)
        random.shuffle(files)
        n_val = max(1, int(len(files) * val_ratio))
        return files[n_val:], files[:n_val]

    real_train, real_val = split(real_files)
    fake_train, fake_val = split(fake_files)

    train = [(f, 1) for f in real_train] + [(f, 0) for f in fake_train]
    val = [(f, 1) for f in real_val] + [(f, 0) for f in fake_val]
    random.shuffle(train)
    random.shuffle(val)

    for name, data in [("train.csv", train), ("val.csv", val)]:
        path = DATA_DIR / name
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["filepath", "label"])
            writer.writeheader()
            for filepath, label in data:
                writer.writerow({"filepath": str(filepath.resolve()), "label": label})
        real_n = sum(1 for _, l in data if l == 1)
        fake_n = sum(1 for _, l in data if l == 0)
        print(f"  {name}: {len(data)}개 (real={real_n}, fake={fake_n})")


# ─────────────────────────── 학습 ────────────────────────────────

def run_training(args):
    """fine_tune.py 실행"""
    print("\n=== Fine-tuning 시작 ===")
    train_csv = DATA_DIR / "train.csv"
    val_csv = DATA_DIR / "val.csv"

    # 데이터 수에 따른 파라미터 자동 조정
    with open(train_csv) as f:
        n_train = sum(1 for _ in f) - 1  # header 제외

    # 대규모 데이터: batch 32, epoch 30, freeze 5
    batch_size = 32 if n_train > 5000 else 16
    epochs = 30 if n_train > 5000 else 15
    freeze_epochs = 5 if n_train > 5000 else 3

    print(f"  학습 데이터: {n_train}개")
    print(f"  batch_size={batch_size}, epochs={epochs}, freeze_epochs={freeze_epochs}")

    cmd = [
        sys.executable, str(SCRIPT_DIR / "fine_tune.py"),
        "--train-csv", str(train_csv),
        "--val-csv", str(val_csv),
        "--output-dir", str(SCRIPT_DIR / "weights" / "finetuned"),
        "--batch-size", str(batch_size),
        "--epochs", str(epochs),
        "--freeze-epochs", str(freeze_epochs),
        "--lr", "1e-4",
        "--weight-decay", "1e-4",
        "--device", "auto",
    ]
    run(cmd, cwd=SCRIPT_DIR)


# ─────────────────────────── 메인 ────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AIHub 다운로드 → 전처리 → Fine-tuning 파이프라인")
    parser.add_argument("--api-key", required=True, help="AIHub API 키")
    parser.add_argument("--skip-download", action="store_true", help="다운로드 단계 건너뜀")
    parser.add_argument("--skip-process", action="store_true", help="WAV 변환 단계 건너뜀")
    parser.add_argument("--skip-train", action="store_true", help="학습 단계 건너뜀")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="검증 데이터 비율")
    args = parser.parse_args()

    REAL_DIR.mkdir(parents=True, exist_ok=True)
    FAKE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("AIHub 파이프라인 시작")
    print(f"  다운로드 디렉토리: {DOWNLOAD_DIR}")
    print(f"  출력 디렉토리:     {DATA_DIR}")
    print(f"{'='*60}\n")

    for ds in DATASETS:
        # 1) 다운로드
        if not args.skip_download:
            dl_dir = download_dataset(ds, args.api_key)
        else:
            dl_dir = DOWNLOAD_DIR / ds["name"]
            print(f"[SKIP] 다운로드 건너뜀: {ds['name']}")

        # 2) 압축 해제 + 변환
        if not args.skip_process:
            ext_dir = extract_dataset(dl_dir, ds["name"])
            n = process_dataset(ext_dir, ds)
            print(f"  → {ds['label']} 데이터 추가: {n}개")
        else:
            print(f"[SKIP] 변환 건너뜀: {ds['name']}")

    # 3) CSV 재구성
    if not args.skip_process:
        rebuild_csv(val_ratio=args.val_ratio)

    # 4) Fine-tuning
    if not args.skip_train:
        run_training(args)
    else:
        print("\n[SKIP] 학습 건너뜀")

    print("\n파이프라인 완료!")


if __name__ == "__main__":
    main()
