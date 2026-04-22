#!/usr/bin/env python3
"""Interactive/non-interactive WAV recording helper with quality gates."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import soundfile as sf


SAMPLE_RATE = 48_000
CHANNELS = 1
BIT_DEPTH = "PCM_24"
MIN_SNR_DB = 25.0
MAX_TRUE_PEAK_DBFS = -1.0
CLIP_THRESHOLD = 0.999
VOICE_CLONE_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_ROOT = VOICE_CLONE_DIR / "data" / "raw"
DEFAULT_SCRIPT_PATH = Path(__file__).resolve().parent / "recording_script_ko.txt"


@dataclass(slots=True)
class RecordingAssessment:
    duration_sec: float
    snr_db: float
    noise_floor_dbfs: float
    peak_dbfs: float
    true_peak_dbfs: float
    clipping: bool

    @property
    def accepted(self) -> bool:
        return self.snr_db >= MIN_SNR_DB and not self.clipping and self.true_peak_dbfs <= MAX_TRUE_PEAK_DBFS


def _import_sounddevice() -> Any:
    try:
        import sounddevice as sd
    except ModuleNotFoundError as exc:
        raise RuntimeError("sounddevice is required for recording. Install it in .venv-cloning.") from exc
    return sd


def load_script_lines(script_path: Path) -> list[str]:
    return [line.strip() for line in script_path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]


def _dbfs(value: float) -> float:
    if value <= 0.0:
        return float("-inf")
    return float(20.0 * math.log10(value))


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)) + 1e-12))


def measure_noise_floor(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 1e-8
    frame = max(int(SAMPLE_RATE * 0.05), 1)
    usable = (audio.size // frame) * frame
    if usable < frame:
        return max(_rms(audio), 1e-8)
    levels = np.sqrt(np.mean(np.square(audio[:usable].reshape(-1, frame), dtype=np.float64), axis=1))
    return max(float(np.percentile(levels, 20)), 1e-8)


def true_peak_dbfs(audio: np.ndarray) -> float:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    try:
        import librosa

        oversampled = librosa.resample(np.asarray(audio, dtype=np.float32), orig_sr=SAMPLE_RATE, target_sr=SAMPLE_RATE * 4)
        if oversampled.size:
            peak = max(peak, float(np.max(np.abs(oversampled))))
    except Exception:
        pass
    return _dbfs(peak)


def assess_recording(audio: np.ndarray, noise_floor: float | None = None) -> RecordingAssessment:
    duration_sec = audio.size / SAMPLE_RATE if SAMPLE_RATE else 0.0
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    floor = max(noise_floor if noise_floor is not None else measure_noise_floor(audio), 1e-8)
    speech = max(_rms(audio), 1e-8)
    snr_db = 20.0 * math.log10(speech / floor)
    clipping = bool(np.sum(np.abs(audio) >= CLIP_THRESHOLD) >= 3)
    return RecordingAssessment(
        duration_sec=duration_sec,
        snr_db=float(snr_db),
        noise_floor_dbfs=_dbfs(floor),
        peak_dbfs=_dbfs(peak),
        true_peak_dbfs=true_peak_dbfs(audio),
        clipping=clipping,
    )


def record_audio(device: int | None, duration_sec: float | None) -> np.ndarray:
    sd = _import_sounddevice()
    if duration_sec is None:
        import queue
        q: "queue.Queue[np.ndarray]" = queue.Queue()

        def _callback(indata, frames, time_info, status):  # noqa: ARG001
            q.put(indata.copy())

        input("Press Enter to start recording, then Enter again to stop.")
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="float32", device=device, callback=_callback):
            input("Recording... press Enter to stop.")
        chunks: list[np.ndarray] = []
        while not q.empty():
            chunks.append(q.get())
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks).reshape(-1)
    audio = sd.rec(int(SAMPLE_RATE * duration_sec), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="float32", device=device)
    sd.wait()
    return np.asarray(audio, dtype=np.float32).reshape(-1)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _assessment_dict(assessment: RecordingAssessment) -> dict[str, Any]:
    return {
        "duration_sec": round(assessment.duration_sec, 3),
        "snr_db": round(assessment.snr_db, 3),
        "noise_floor_dbfs": round(assessment.noise_floor_dbfs, 3),
        "peak_dbfs": round(assessment.peak_dbfs, 3),
        "true_peak_dbfs": round(assessment.true_peak_dbfs, 3),
        "clipping": assessment.clipping,
        "accepted": assessment.accepted,
    }


def _print_assessment(assessment: RecordingAssessment) -> None:
    print(
        f"SNR={assessment.snr_db:.1f}dB noise_floor={assessment.noise_floor_dbfs:.1f}dBFS "
        f"peak={assessment.peak_dbfs:.1f}dBFS true_peak={assessment.true_peak_dbfs:.1f}dBFS "
        f"clipping={assessment.clipping}"
    )
    if not assessment.accepted:
        print("Rejected: SNR must be >=35dB, clipping must be false, and true peak must be <= -1dBFS.")


def run_session(args: argparse.Namespace) -> int:
    script_lines = load_script_lines(Path(args.script))
    session_name = args.session or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    session_dir = RAW_DATA_ROOT / args.speaker / session_name
    session_dir.mkdir(parents=True, exist_ok=True)
    noise_floor = None
    log: dict[str, Any] = {
        "speaker": args.speaker,
        "session": session_name,
        "script_path": str(Path(args.script).resolve()),
        "sample_rate": SAMPLE_RATE,
        "clips": [],
    }

    if args.noise_seconds > 0:
        print(f"Recording {args.noise_seconds:.1f}s room tone for noise floor.")
        noise_audio = record_audio(args.device, args.noise_seconds)
        noise_floor = measure_noise_floor(noise_audio)
        log["noise_floor_dbfs"] = round(_dbfs(noise_floor), 3)

    exit_code = 0
    for index, sentence in enumerate(script_lines, start=1):
        while True:
            print(f"[{index}/{len(script_lines)}] {sentence}")
            audio = record_audio(args.device, args.duration)
            assessment = assess_recording(audio, noise_floor)
            _print_assessment(assessment)
            clip_path = session_dir / f"clip_{index:03d}.wav"
            if assessment.accepted or args.non_interactive:
                sf.write(clip_path, audio, SAMPLE_RATE, subtype=BIT_DEPTH)
                entry = {"index": index, "text": sentence, "path": clip_path.name, **_assessment_dict(assessment)}
                log["clips"].append(entry)
                write_json(session_dir / "session_log.json", log)
                if not assessment.accepted:
                    exit_code = 2
                break
            answer = input("Re-record this clip? [Y/n] ").strip().lower()
            if answer in {"n", "no"}:
                sf.write(clip_path, audio, SAMPLE_RATE, subtype=BIT_DEPTH)
                log["clips"].append({"index": index, "text": sentence, "path": clip_path.name, **_assessment_dict(assessment)})
                write_json(session_dir / "session_log.json", log)
                exit_code = 2
                break
    print(session_dir)
    return exit_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Voice-clone recording helper")
    parser.add_argument("--speaker", required=True)
    parser.add_argument("--script", type=Path, default=DEFAULT_SCRIPT_PATH)
    parser.add_argument("--session")
    parser.add_argument("--device", type=int)
    parser.add_argument("--duration", type=float, help="Non-interactive fixed clip duration in seconds")
    parser.add_argument("--noise-seconds", type=float, default=3.0)
    parser.add_argument("--non-interactive", action="store_true", help="Do not prompt; log rejection and exit with code 2.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.non_interactive and args.duration is None:
        raise SystemExit("--non-interactive requires --duration")
    try:
        return run_session(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
