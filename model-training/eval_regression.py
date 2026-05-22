"""
AASIST 회귀 평가 스크립트
===========================
Baseline 및 회귀 검증용 — deployed 모델 성능을 CSV + GPT-SoVITS 클립으로 측정.

Label convention (ASVspoof / fine_tune.py 동일):
  0 = spoof (fake),  1 = bonafide (real)
fakeScore = softmax(logits)[0]   (spoof 확률)

torchaudio 2.11 호환성 노트:
  torchaudio 2.11은 기본 backend로 TorchCodec을 요구하지만 환경에 설치되지 않을 수 있음.
  soundfile 패키지로 대체하여 WAV 파일을 로드하도록 torchaudio.load를 monkey-patch.
  torchaudio.functional.resample은 여전히 사용 (tensor 연산만 수행하므로 문제없음).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# torchaudio.load monkey-patch (soundfile 기반, TorchCodec 불필요)
# ---------------------------------------------------------------------------
try:
    import torchaudio as _ta
    _ta.load  # probe the attribute
    import soundfile as _sf

    def _sf_load(path, frame_offset=0, num_frames=-1,
                 normalize=True, channels_first=True,
                 format=None, backend=None, **kwargs):
        """soundfile 기반 torchaudio.load 대체 구현.
        WAV/FLAC/OGG를 지원. torchaudio 2.11+ 의 TorchCodec 의존성 우회.
        """
        data, sr = _sf.read(str(path), dtype="float32", always_2d=True)
        # data: (samples, channels) -> (channels, samples)
        data = data.T
        waveform = torch.from_numpy(data.copy())
        if frame_offset > 0:
            waveform = waveform[:, frame_offset:]
        if num_frames > 0:
            waveform = waveform[:, :num_frames]
        return waveform, sr

    _ta.load = _sf_load
    print("[patch] torchaudio.load -> soundfile backend (TorchCodec 우회)")
except Exception as _e:
    print(f"[WARN] torchaudio patch skipped: {_e}")

import torchaudio  # noqa: E402  (re-import after patch)

# model-training/ 디렉터리를 sys.path에 추가
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from fine_tune import AudioDataset, get_device  # noqa: E402
from models.AASIST import Model  # noqa: E402


# ---------------------------------------------------------------------------
# 기본 경로 상수
# ---------------------------------------------------------------------------
_WEIGHTS_DEFAULT = str(
    _SCRIPT_DIR / "weights" / "finetuned_v4_stable_2026-05-11" / "aasist_best.pth"
)
_CONFIG_DEFAULT = str(_SCRIPT_DIR / "configs" / "aasist.json")
_DATA_DIR = _SCRIPT_DIR / "data"

_DEFAULT_CSVS = [
    str(_DATA_DIR / "test_iphone_heldout.csv"),
    str(_DATA_DIR / "test_speaker_holdout.csv"),
    str(_DATA_DIR / "test_fake_regression.csv"),
]

_REPO_ROOT = _SCRIPT_DIR.parent
_GPT_DEMO_DIR = str(_REPO_ROOT / "GPT-SoVITS" / "outputs" / "jhm" / "demo")
_GPT_DATASET_DIR = str(
    _REPO_ROOT / "GPT-SoVITS" / "training_data" / "jhm" / "dataset" / "wavs"
)


# ---------------------------------------------------------------------------
# 모델 로드
# ---------------------------------------------------------------------------
def load_model(checkpoint_path: str, config_path: str, device: torch.device) -> Model:
    with open(config_path) as f:
        config = json.load(f)
    model = Model(config["model_config"])
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    result = model.load_state_dict(ckpt, strict=False)
    print(
        f"[model] loaded {checkpoint_path}\n"
        f"        missing={len(result.missing_keys)}, "
        f"unexpected={len(result.unexpected_keys)}"
    )
    model = model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# 오디오 -> 16kHz mono 텐서 (soundfile 기반)
# ---------------------------------------------------------------------------
def load_audio_tensor(path: str) -> torch.Tensor:
    """파일을 16kHz mono float32 텐서로 로드.
    soundfile (monkey-patched torchaudio.load) + torchaudio.functional.resample 사용.
    """
    waveform, sr = torchaudio.load(path)   # (C, T) float32
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.squeeze(0)  # (T,)

    # 볼륨 정규화 (fine_tune.py AudioDataset 동일 로직)
    peak = waveform.abs().max()
    if peak > 0 and peak < 0.05:
        gain = min(0.3 / peak, 10.0)
        waveform = waveform * gain

    return waveform


# ---------------------------------------------------------------------------
# 슬라이딩 윈도우 fakeScore (50% overlap, 앱 DemoAnalysisPipeline 방식과 동일)
# ---------------------------------------------------------------------------
def score_file(
    model: Model, path: str, device: torch.device, nb_samp: int = 64600
) -> float:
    """fakeScore = softmax(logits)[0] 의 sliding-window 평균."""
    try:
        waveform = load_audio_tensor(path)
    except Exception as e:
        print(f"  [WARN] load failed: {path}: {e}")
        return float("nan")

    n = waveform.shape[0]
    hop = nb_samp // 2

    if n <= nb_samp:
        windows = [F.pad(waveform, (0, nb_samp - n))]
    else:
        windows = []
        start = 0
        while start + nb_samp <= n:
            windows.append(waveform[start : start + nb_samp])
            start += hop
        if start < n:
            chunk = waveform[start:]
            windows.append(F.pad(chunk, (0, nb_samp - chunk.shape[0])))

    scores = []
    try:
        with torch.no_grad():
            for chunk in windows:
                inp = chunk.unsqueeze(0).to(device)  # (1, nb_samp)
                _, logits, _ = model(inp)             # 3-tuple: fine_tune.py 컨벤션
                probs = torch.softmax(logits, dim=1)
                scores.append(probs[0, 0].item())    # index 0 = spoof prob
    except Exception as e:
        print(f"  [WARN] inference failed: {path}: {e}")
        return float("nan")

    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# CSV 평가 (AudioDataset + DataLoader 재사용)
# ---------------------------------------------------------------------------
def assess_csv(
    model: Model,
    csv_path: str,
    device: torch.device,
    batch_size: int = 16,
) -> dict:
    """CSV 파일 전체 평가. returns dict with acc, per-class stats."""
    from torch.utils.data import DataLoader

    dataset = AudioDataset(csv_path, nb_samp=64600, augment=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    real_correct = real_total = fake_correct = fake_total = 0

    with torch.no_grad():
        for waveforms, labels in loader:
            waveforms = waveforms.to(device)
            _, logits, _ = model(waveforms)
            preds = logits.argmax(dim=1)  # 0=fake, 1=real

            for pred, label in zip(preds.cpu().tolist(), labels.tolist()):
                if label == 1:  # real
                    real_total += 1
                    if pred == 1:
                        real_correct += 1
                else:           # fake
                    fake_total += 1
                    if pred == 0:
                        fake_correct += 1

    n = real_total + fake_total
    overall_correct = real_correct + fake_correct
    return {
        "n": n,
        "acc": overall_correct / n if n > 0 else 0.0,
        "real_correct": real_correct,
        "real_total": real_total,
        "real_acc": real_correct / real_total if real_total > 0 else None,
        "fake_correct": fake_correct,
        "fake_total": fake_total,
        "fake_acc": fake_correct / fake_total if fake_total > 0 else None,
    }


# ---------------------------------------------------------------------------
# GPT-SoVITS 폴더 평가
# ---------------------------------------------------------------------------
def assess_gptsovits_dir(
    model: Model,
    dir_path: str,
    device: torch.device,
    max_files=None,
    named: bool = False,
) -> dict:
    """디렉터리의 wav 파일들을 평가.
    named=True 이면 개별 파일 결과도 즉시 출력.
    """
    p = Path(dir_path)
    wavs = sorted(p.glob("*.wav"))
    if not wavs:
        wavs = sorted(p.glob("**/*.wav"))
    if max_files is not None:
        wavs = wavs[:max_files]

    scores = {}
    for wav in wavs:
        fs = score_file(model, str(wav), device)
        scores[wav.name] = fs
        if named:
            tag = "FAKE" if (not np.isnan(fs) and fs > 0.5) else "real"
            print(f"    {wav.name:30s}  fakeScore={fs:.4f}  [{tag}]")

    valid = [v for v in scores.values() if not np.isnan(v)]
    mean_fs = float(np.mean(valid)) if valid else float("nan")
    n_det = sum(1 for v in valid if v > 0.5)
    det_rate = n_det / len(valid) if valid else float("nan")
    return {
        "n": len(wavs),
        "n_valid": len(valid),
        "n_detected": n_det,
        "mean_fakeScore": mean_fs,
        "detected_rate": det_rate,
        "per_file": scores,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="AASIST regression assessment -- CSV accuracy + GPT-SoVITS fake detection rate"
    )
    parser.add_argument("--model", default=_WEIGHTS_DEFAULT, help="checkpoint path")
    parser.add_argument("--config", default=_CONFIG_DEFAULT, help="AASIST config JSON")
    parser.add_argument(
        "--csvs",
        nargs="+",
        default=[c for c in _DEFAULT_CSVS if Path(c).exists()],
        help="CSV files to evaluate (filepath,label)",
    )
    parser.add_argument(
        "--gptsovits-dirs",
        nargs="+",
        default=[d for d in [_GPT_DEMO_DIR, _GPT_DATASET_DIR] if Path(d).exists()],
        help="GPT-SoVITS wav directories",
    )
    parser.add_argument(
        "--gptsovits-sample",
        type=int,
        default=60,
        help="max dataset-dir files to score",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    # 체크포인트 fallback
    model_path = args.model
    if not Path(model_path).exists():
        fallback = str(
            _SCRIPT_DIR / "weights" / "restore_pre_gptsovits" / "aasist_best.pth"
        )
        print(f"[WARN] model not found: {model_path}")
        if Path(fallback).exists():
            model_path = fallback
            print(f"[INFO] fallback to: {model_path}")
        else:
            print("[ERROR] no checkpoint found; aborting")
            sys.exit(1)

    device = get_device(args.device)
    print(f"\n{'='*60}")
    print("AASIST REGRESSION ASSESSMENT")
    print(f"{'='*60}")
    print(f"  model  : {model_path}")
    print(f"  config : {args.config}")
    print(f"  device : {device}")
    print()

    model = load_model(model_path, args.config, device)

    # -------------------------------------------------------------------
    # 1. CSV 평가
    # -------------------------------------------------------------------
    print(f"\n{'─'*60}")
    print("CSV ACCURACY")
    print(f"{'─'*60}")

    csv_results = {}
    for csv_path in args.csvs:
        if not Path(csv_path).exists():
            print(f"  [SKIP] not found: {csv_path}")
            continue
        name = Path(csv_path).stem
        print(f"\n  [{name}]")
        try:
            r = assess_csv(model, csv_path, device, batch_size=args.batch_size)
        except Exception as e:
            print(f"  [ERROR] {e}")
            csv_results[name] = None
            continue

        csv_results[name] = r
        print(f"    overall : acc={r['acc']:.4f}  n={r['n']}")
        if r["real_total"] > 0:
            print(f"    real    : {r['real_correct']}/{r['real_total']} = {r['real_acc']:.4f}")
        if r["fake_total"] > 0:
            print(f"    fake    : {r['fake_correct']}/{r['fake_total']} = {r['fake_acc']:.4f}")

    # -------------------------------------------------------------------
    # 2. GPT-SoVITS 평가
    # -------------------------------------------------------------------
    print(f"\n{'─'*60}")
    print("GPT-SoVITS FAKE DETECTION")
    print(f"{'─'*60}")

    gptsovits_results = {}
    for i, dir_path in enumerate(args.gptsovits_dirs or []):
        if not Path(dir_path).exists():
            print(f"  [SKIP] not found: {dir_path}")
            continue

        is_demo = i == 0
        cap = None if is_demo else args.gptsovits_sample
        label = "demo" if is_demo else f"dataset (first {cap})"

        print(f"\n  [{label}]  {dir_path}")
        r = assess_gptsovits_dir(model, dir_path, device, max_files=cap, named=is_demo)
        gptsovits_results[dir_path] = r

        print(
            f"    n_scored  : {r['n_valid']}/{r['n']}\n"
            f"    mean fakeScore          : {r['mean_fakeScore']:.4f}\n"
            f"    detected-as-fake (>0.5) : "
            f"{r['n_detected']}/{r['n_valid']} = {r['detected_rate']:.4f}"
        )

    # -------------------------------------------------------------------
    # 최종 요약 블록
    # -------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("BASELINE SUMMARY")
    print(f"{'='*60}")
    print(f"  model: {model_path}\n")

    print("  [CSV Accuracy]")
    for name, r in csv_results.items():
        if r is None:
            print(f"    {name}: ERROR")
            continue
        real_str = (
            f"real {r['real_correct']}/{r['real_total']}={r['real_acc']:.4f}"
            if r["real_total"] > 0 else ""
        )
        fake_str = (
            f"fake {r['fake_correct']}/{r['fake_total']}={r['fake_acc']:.4f}"
            if r["fake_total"] > 0 else ""
        )
        parts = [p for p in [real_str, fake_str] if p]
        print(
            f"    {name}: overall_acc={r['acc']:.4f} n={r['n']}"
            + (f"  |  {' | '.join(parts)}" if parts else "")
        )

    print("\n  [GPT-SoVITS Detection]")
    for i, (dir_path, r) in enumerate(gptsovits_results.items()):
        lbl = "demo" if i == 0 else "dataset-sample"
        print(
            f"    {lbl} ({Path(dir_path).name}): "
            f"mean_fakeScore={r['mean_fakeScore']:.4f}  "
            f"detected={r['n_detected']}/{r['n_valid']} ({r['detected_rate']:.4f})"
        )
        if i == 0:
            for fname, fs in r["per_file"].items():
                tag = "FAKE" if (not np.isnan(fs) and fs > 0.5) else "real"
                print(f"      {fname}: fakeScore={fs:.4f} [{tag}]")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
