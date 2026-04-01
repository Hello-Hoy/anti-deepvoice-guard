"""
공식 AASIST 가중치 다운로드 스크립트
clovaai/aasist (ASVspoof 2019 LA 학습)
"""

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

WEIGHTS = {
    "AASIST": {
        "url": "https://raw.githubusercontent.com/clovaai/aasist/main/models/weights/AASIST.pth",
        "filename": "AASIST.pth",
        "expected_size": 1_281_532,
    },
    "AASIST-L": {
        "url": "https://raw.githubusercontent.com/clovaai/aasist/main/models/weights/AASIST-L.pth",
        "filename": "AASIST-L.pth",
        "expected_size": 426_428,
    },
}


def download(url: str, dest: Path) -> None:
    print(f"Downloading {url}")
    print(f"  -> {dest}")
    urllib.request.urlretrieve(url, dest)
    size = dest.stat().st_size
    print(f"  Done ({size:,} bytes)")


def verify_loadable(path: Path, config_path: Path) -> None:
    """가중치가 AASIST 모델에 로드 가능한지 검증"""
    import json

    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from models.AASIST import Model

    with open(config_path) as f:
        config = json.load(f)

    model = Model(config["model_config"])
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    result = model.load_state_dict(ckpt, strict=False)

    if result.unexpected_keys:
        print(f"  WARNING unexpected keys: {result.unexpected_keys}")
    if result.missing_keys:
        print(f"  INFO missing keys (normal): {result.missing_keys}")

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Model loaded OK — {param_count:,} parameters")


def main():
    parser = argparse.ArgumentParser(description="Download official AASIST weights")
    parser.add_argument(
        "--model",
        choices=["AASIST", "AASIST-L", "all"],
        default="all",
        help="Which model to download (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "weights" / "official",
        help="Output directory",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        default=True,
        help="Verify weights load into model (default: True)",
    )
    parser.add_argument(
        "--no-verify",
        action="store_false",
        dest="verify",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    models = list(WEIGHTS.keys()) if args.model == "all" else [args.model]
    config_dir = Path(__file__).resolve().parent / "configs"

    for name in models:
        info = WEIGHTS[name]
        dest = args.output_dir / info["filename"]

        if dest.exists() and dest.stat().st_size == info["expected_size"]:
            print(f"[{name}] Already exists, skipping download")
        else:
            download(info["url"], dest)

        actual_size = dest.stat().st_size
        if actual_size != info["expected_size"]:
            print(
                f"  WARNING: size mismatch (expected {info['expected_size']:,}, got {actual_size:,})"
            )

        if args.verify:
            config_name = "aasist.json" if name == "AASIST" else "aasist_l.json"
            config_path = config_dir / config_name
            print(f"[{name}] Verifying model load with {config_path.name}...")
            verify_loadable(dest, config_path)

    print("\nAll done!")


if __name__ == "__main__":
    main()
