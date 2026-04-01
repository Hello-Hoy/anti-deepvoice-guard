"""AASIST 모델 추론 래퍼.

AASIST 모델을 로드하고 오디오 세그먼트에 대한 deepfake 탐지 추론을 수행한다.
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

# 프로젝트 내 model-training/models/ 참조
_MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "model-training"
sys.path.insert(0, str(_MODELS_DIR))

from models.AASIST import Model

SAMPLE_RATE = 16000
NB_SAMP = 64600  # ~4초 @ 16kHz (AASIST 공식 입력 크기)

# 기본 config 경로
DEFAULT_CONFIG_PATH = str(_MODELS_DIR / "configs" / "aasist.json")


@dataclass
class DetectionResult:
    fake_score: float
    real_score: float
    confidence: float
    model_version: str
    inference_ms: float


def pad_audio(x: np.ndarray, max_len: int = NB_SAMP) -> np.ndarray:
    """오디오를 고정 길이로 패딩/크롭."""
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len) + 1
    padded = np.tile(x, num_repeats)[:max_len]
    return padded


class AASISTDetector:
    """AASIST 기반 deepfake 음성 탐지기."""

    def __init__(self, model_path: str, config_path: str = None, device: str = "cpu"):
        self.device = torch.device(device)
        self.model_version = "aasist-v1"

        if config_path is None:
            config_path = DEFAULT_CONFIG_PATH

        with open(config_path, "r") as f:
            config = json.load(f)

        model_config = config["model_config"]
        self.model = Model(model_config)

        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        else:
            self.model.load_state_dict(checkpoint, strict=False)

        self.model.to(self.device)
        self.model.eval()

    def detect(self, audio: np.ndarray) -> DetectionResult:
        """오디오 세그먼트에 대해 deepfake 탐지 수행.

        Args:
            audio: 1D numpy array, 16kHz mono PCM float32

        Returns:
            DetectionResult with fake/real scores
        """
        # 전처리: 패딩/크롭
        audio = pad_audio(audio.astype(np.float32), NB_SAMP)
        x = torch.FloatTensor(audio).unsqueeze(0).to(self.device)

        start = perf_counter()
        with torch.no_grad():
            _, output, _ = self.model(x, alpha=0, Freq_aug=False)
            probs = torch.softmax(output, dim=1)

        elapsed_ms = (perf_counter() - start) * 1000

        # ASVspoof convention: index 0 = spoof, index 1 = bonafide
        fake_score = probs[0][0].item()
        real_score = probs[0][1].item()
        confidence = max(real_score, fake_score)

        return DetectionResult(
            fake_score=fake_score,
            real_score=real_score,
            confidence=confidence,
            model_version=self.model_version,
            inference_ms=round(elapsed_ms, 1),
        )
