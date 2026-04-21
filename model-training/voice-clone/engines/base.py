from __future__ import annotations

from abc import ABC, abstractmethod
import os
from pathlib import Path
import subprocess
from typing import Any


DEFAULT_WEIGHTS_BASE = Path(__file__).resolve().parents[2] / "weights" / "voice-clone"
_CHECKPOINT_CANDIDATES = (
    "best.ckpt",
    "best.pt",
    "best.pth",
    "last.ckpt",
    "last.pt",
    "last.pth",
    "model.ckpt",
    "model.pt",
    "model.pth",
    "checkpoint.ckpt",
    "checkpoint.pt",
    "checkpoint.pth",
)
_CHECKPOINT_SUFFIXES = (".ckpt", ".pt", ".pth", ".bin", ".onnx")


class VoiceCloneEngine(ABC):
    name: str
    train_sample_rate: int
    supports_language: list[str]
    license: str
    phase: str
    hardware_pref: str

    @abstractmethod
    def health_check(self) -> tuple[bool, str]:
        """의존성, 가상환경, 하드웨어, 체크포인트 준비 상태를 검증한다."""

    @abstractmethod
    def prepare_data(self, manifest_path: Path, out_dir: Path) -> Path:
        """공통 manifest를 엔진 고유 포맷으로 변환하고 결과 디렉토리를 반환한다."""

    @abstractmethod
    def train(
        self,
        data_dir: Path,
        ckpt_out: Path,
        epochs: int = 10,
        **kwargs: Any,
    ) -> Path:
        """파인튜닝을 수행하고 결과 체크포인트 경로를 반환한다."""

    @abstractmethod
    def synthesize(
        self,
        text: str,
        ckpt: Path | None,
        ref_wav: Path | None = None,
        language: str = "ko",
        out_path: Path = Path("out.wav"),
        sample_rate: int | None = None,
    ) -> Path:
        """텍스트를 합성해 출력 파일 경로를 반환한다."""


def run_in_venv(
    venv_python: Path,
    script: Path,
    args: list[str],
    env: dict[str, str] | None = None,
    timeout: int = 3600,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """엔진별 외부 venv를 분리 실행한다."""
    if not venv_python.exists():
        raise FileNotFoundError(f"Virtualenv python not found: {venv_python}")
    if not script.exists():
        raise FileNotFoundError(f"Script not found: {script}")

    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    try:
        return subprocess.run(
            [str(venv_python), str(script), *args],
            capture_output=True,
            text=True,
            check=False,
            env=merged_env,
            timeout=timeout,
            cwd=str(cwd) if cwd is not None else None,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(
            f"Timed out after {timeout}s while running {script.name} in {venv_python}"
        ) from exc


def resolve_checkpoint(
    speaker: str,
    engine: str,
    base: Path = DEFAULT_WEIGHTS_BASE,
) -> Path | None:
    """표준 가중치 디렉토리에서 엔진 체크포인트를 찾는다."""
    speaker_candidates = [base / f"user-{speaker}" / engine, base / speaker / engine]

    for engine_dir in speaker_candidates:
        if not engine_dir.exists():
            continue
        if engine_dir.is_file():
            return engine_dir

        for filename in _CHECKPOINT_CANDIDATES:
            candidate = engine_dir / filename
            if candidate.exists():
                return candidate

        recursive_files = sorted(
            path
            for path in engine_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in _CHECKPOINT_SUFFIXES
        )
        if recursive_files:
            return recursive_files[0]
        return engine_dir

    return None
