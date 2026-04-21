#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import threading
import time
from typing import Any

import numpy as np
import soundfile as sf


SAMPLE_RATE = 48_000
CHANNELS = 1
BIT_DEPTH = "PCM_24"
PHASE0_DURATION_SEC = 10.0
MIN_DURATION_SEC = 2.0
MAX_DURATION_SEC = 20.0
CLIPPING_THRESHOLD_DBFS = -3.0
LOW_LEVEL_THRESHOLD_DBFS = -24.0
SNR_WARNING_THRESHOLD_DB = 20.0


class Ansi:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    CYAN = "\033[36m"


@dataclass(slots=True)
class RecordingAssessment:
    duration_sec: float
    peak_dbfs: float
    snr_db: float
    warning_codes: list[str]
    warning_messages: list[str]


@dataclass(slots=True)
class MeterState:
    peak_linear: float = 0.0
    start_time: float = 0.0
    stream_status: str = ""
    lock: threading.Lock | None = None

    def __post_init__(self) -> None:
        if self.lock is None:
            self.lock = threading.Lock()

    def update_peak(self, peak_linear: float) -> None:
        assert self.lock is not None
        with self.lock:
            self.peak_linear = max(self.peak_linear * 0.92, peak_linear)

    def set_status(self, stream_status: str) -> None:
        assert self.lock is not None
        with self.lock:
            self.stream_status = stream_status

    def snapshot(self) -> tuple[float, float, str]:
        assert self.lock is not None
        with self.lock:
            elapsed = max(time.monotonic() - self.start_time, 0.0)
            return self.peak_linear, elapsed, self.stream_status


REPO_ROOT = Path(__file__).resolve().parents[3]
VOICE_CLONE_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SCRIPT_PATH = SCRIPT_DIR / "recording_script_ko.txt"
RAW_DATA_ROOT = VOICE_CLONE_DIR / "data" / "raw"
PHASE0_OUTPUT_ROOT = REPO_ROOT / "test-samples" / "voice-clone-raw"


def _import_sounddevice():
    try:
        import sounddevice as sd
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on runtime environment
        raise RuntimeError(
            "sounddevice 패키지가 없습니다. `.venv-cloning` 환경에서 실행하거나 `pip install sounddevice`를 진행하세요."
        ) from exc
    return sd


def colorize(text: str, color: str, enabled: bool = True) -> str:
    if not enabled:
        return text
    return f"{color}{text}{Ansi.RESET}"


def clear_screen() -> None:
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def load_script_lines(script_path: Path) -> list[str]:
    if not script_path.exists():
        raise FileNotFoundError(f"대본 파일을 찾을 수 없습니다: {script_path}")

    lines = [line.strip() for line in script_path.read_text(encoding="utf-8").splitlines()]
    return [line for line in lines if line]


def compute_peak_dbfs(audio: np.ndarray) -> float:
    if audio.size == 0:
        return float("-inf")

    peak = float(np.max(np.abs(audio)))
    if peak <= 0.0:
        return float("-inf")
    if peak >= 1.0:
        return 0.0
    return float(20.0 * math.log10(peak))


def estimate_snr_db(audio: np.ndarray, sample_rate: int) -> float:
    if audio.size == 0:
        return 0.0

    frame_length = max(int(sample_rate * 0.05), 1)
    if audio.size < frame_length * 4:
        return 0.0

    usable = (audio.size // frame_length) * frame_length
    frames = audio[:usable].reshape(-1, frame_length)
    rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))
    rms = np.clip(rms, 1e-8, None)

    noise_floor = float(np.mean(np.percentile(rms, [10, 20])))
    speech_floor = float(np.mean(np.percentile(rms, [80, 90])))
    if speech_floor <= noise_floor:
        return 0.0
    return float(20.0 * math.log10(speech_floor / max(noise_floor, 1e-8)))


def assess_recording(audio: np.ndarray, sample_rate: int) -> RecordingAssessment:
    duration_sec = float(audio.size / sample_rate) if sample_rate > 0 else 0.0
    peak_dbfs = compute_peak_dbfs(audio)
    snr_db = estimate_snr_db(audio, sample_rate)

    warning_codes: list[str] = []
    warning_messages: list[str] = []

    if peak_dbfs > CLIPPING_THRESHOLD_DBFS:
        warning_codes.append("clipping_risk")
        warning_messages.append("⚠ 클리핑 위험: 입력 레벨이 높습니다. 마이크와 거리를 조금 늘리고 재녹음을 권장합니다.")

    if peak_dbfs < LOW_LEVEL_THRESHOLD_DBFS:
        warning_codes.append("low_volume")
        warning_messages.append("⚠ 볼륨 낮음: 입력 레벨이 너무 낮습니다. 입력 게인이나 마이크 위치를 조정하세요.")

    if snr_db < SNR_WARNING_THRESHOLD_DB:
        warning_codes.append("noise_detected")
        warning_messages.append("⚠ 잡음 감지: 주변 소음이 많아 보입니다. 더 조용한 환경에서 다시 녹음하세요.")

    if duration_sec < MIN_DURATION_SEC or duration_sec > MAX_DURATION_SEC:
        warning_codes.append("duration_out_of_range")
        warning_messages.append("⚠ 길이 경고: 권장 길이는 2초 이상 20초 이하입니다.")

    return RecordingAssessment(
        duration_sec=duration_sec,
        peak_dbfs=peak_dbfs,
        snr_db=snr_db,
        warning_codes=warning_codes,
        warning_messages=warning_messages,
    )


def format_db(value: float) -> str:
    if math.isinf(value):
        return "-inf dB"
    return f"{value:.1f} dB"


def prompt(text: str) -> str:
    return input(text).strip().lower()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(path)


def build_empty_clips(script_lines: list[str]) -> list[dict[str, Any]]:
    clips: list[dict[str, Any]] = []
    for index, text in enumerate(script_lines, start=1):
        clips.append(
            {
                "index": index,
                "clip_name": f"clip_{index:03d}.wav",
                "text": text,
                "relative_path": None,
                "duration_sec": None,
                "peak_dbfs": None,
                "snr_db": None,
                "warning_codes": [],
                "warning_messages": [],
                "rec_status": "pending",
                "updated_at": None,
            }
        )
    return clips


def find_resume_start_index(log_entries: list[dict[str, Any]], total_lines: int) -> int:
    completed_statuses = {"recorded", "skipped"}
    by_index = {int(entry["index"]): entry for entry in log_entries if "index" in entry}
    for index in range(1, total_lines + 1):
        entry = by_index.get(index)
        if entry is None or entry.get("rec_status") not in completed_statuses:
            return index - 1
    return total_lines


def summarize_progress(clips: list[dict[str, Any]]) -> tuple[int, int, int]:
    recorded = sum(1 for clip in clips if clip["rec_status"] == "recorded")
    skipped = sum(1 for clip in clips if clip["rec_status"] == "skipped")
    pending = len(clips) - recorded - skipped
    return recorded, skipped, pending


def update_clip_entry(
    log_data: dict[str, Any],
    clip_index: int,
    *,
    assessment: RecordingAssessment | None,
    rec_status: str,
    saved_path: Path | None,
) -> None:
    clip_entry = log_data["clips"][clip_index]
    clip_entry["relative_path"] = saved_path.name if saved_path else None
    clip_entry["duration_sec"] = None if assessment is None else round(assessment.duration_sec, 3)
    clip_entry["peak_dbfs"] = None if assessment is None else round(assessment.peak_dbfs, 2)
    clip_entry["snr_db"] = None if assessment is None else round(assessment.snr_db, 2)
    clip_entry["warning_codes"] = [] if assessment is None else assessment.warning_codes
    clip_entry["warning_messages"] = [] if assessment is None else assessment.warning_messages
    clip_entry["rec_status"] = rec_status
    clip_entry["updated_at"] = datetime.now().isoformat(timespec="seconds")
    log_data["updated_at"] = clip_entry["updated_at"]


def build_meter_line(peak_linear: float, elapsed_sec: float, status: str, *, total_sec: float | None) -> str:
    filled = max(0, min(int(peak_linear * 32), 32))
    bar = "#" * filled + "-" * (32 - filled)
    peak_dbfs = compute_peak_dbfs(np.array([peak_linear], dtype=np.float32))
    elapsed_text = f"{elapsed_sec:5.1f}s"
    if total_sec is not None:
        elapsed_text = f"{elapsed_sec:5.1f}s / {total_sec:.1f}s"
    status_text = f" status={status}" if status else ""
    return f"\r레벨 [{bar}] peak {format_db(peak_dbfs):>8}  {elapsed_text}{status_text}"


def meter_worker(stop_event: threading.Event, state: MeterState, *, total_sec: float | None) -> None:
    while not stop_event.is_set():
        peak_linear, elapsed_sec, status = state.snapshot()
        sys.stdout.write(build_meter_line(peak_linear, elapsed_sec, status, total_sec=total_sec))
        sys.stdout.flush()
        time.sleep(0.12)
    peak_linear, elapsed_sec, status = state.snapshot()
    sys.stdout.write(build_meter_line(peak_linear, elapsed_sec, status, total_sec=total_sec))
    sys.stdout.write("\n")
    sys.stdout.flush()


def record_audio(
    *,
    device_index: int | None,
    auto_stop_sec: float | None,
) -> np.ndarray:
    sd = _import_sounddevice()
    chunks: list[np.ndarray] = []
    meter_state = MeterState(start_time=time.monotonic())
    stop_event = threading.Event()

    def callback(indata: np.ndarray, _frames: int, _time_info: Any, status: Any) -> None:
        mono = np.asarray(indata[:, 0], dtype=np.float32).copy()
        chunks.append(mono)
        meter_state.update_peak(float(np.max(np.abs(mono))) if mono.size else 0.0)
        if status:
            meter_state.set_status(str(status))

    meter_thread = threading.Thread(
        target=meter_worker,
        args=(stop_event, meter_state),
        kwargs={"total_sec": auto_stop_sec},
        daemon=True,
    )

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            device=device_index,
            callback=callback,
        ):
            meter_thread.start()
            if auto_stop_sec is None:
                input("녹음 중입니다. [Enter]를 누르면 종료합니다. ")
            else:
                deadline = time.monotonic() + auto_stop_sec
                while time.monotonic() < deadline:
                    time.sleep(0.05)
    except KeyboardInterrupt:
        raise
    except Exception as exc:  # pragma: no cover - hardware dependent
        raise RuntimeError(format_audio_error(exc)) from exc
    finally:
        stop_event.set()
        meter_thread.join(timeout=1.0)

    if not chunks:
        return np.empty(0, dtype=np.float32)

    return np.concatenate(chunks)


def list_input_devices() -> list[dict[str, Any]]:
    sd = _import_sounddevice()
    devices = sd.query_devices()
    result: list[dict[str, Any]] = []
    for index, device in enumerate(devices):
        if int(device.get("max_input_channels", 0)) <= 0:
            continue
        result.append(
            {
                "index": index,
                "name": str(device.get("name", "Unknown device")),
                "channels": int(device.get("max_input_channels", 0)),
                "default_samplerate": float(device.get("default_samplerate", 0.0)),
            }
        )
    return result


def get_default_input_device_index() -> int | None:
    sd = _import_sounddevice()
    default_device = sd.default.device
    if isinstance(default_device, (list, tuple)) and default_device:
        input_index = default_device[0]
        return None if input_index in (None, -1) else int(input_index)
    if isinstance(default_device, int):
        return None if default_device == -1 else int(default_device)
    return None


def print_device_table(devices: list[dict[str, Any]]) -> None:
    if not devices:
        print("사용 가능한 입력 장치를 찾지 못했습니다.")
        return

    default_index = get_default_input_device_index()
    print("사용 가능한 입력 장치")
    print("-" * 72)
    for device in devices:
        marker = "*" if device["index"] == default_index else " "
        print(
            f"{marker} {device['index']:>2} | {device['channels']}ch | "
            f"{int(device['default_samplerate']):>6} Hz | {device['name']}"
        )
    print("-" * 72)
    print("* = 현재 기본 입력 장치")


def resolve_device_index(device_arg: str | None) -> int | None:
    devices = list_input_devices()
    if not devices:
        raise RuntimeError("사용 가능한 입력 장치를 찾지 못했습니다.")

    if device_arg is not None:
        if device_arg.isdigit():
            candidate = int(device_arg)
            if any(device["index"] == candidate for device in devices):
                return candidate
            raise ValueError(f"입력 장치 {candidate} 을(를) 찾지 못했습니다.")

        lowered = device_arg.lower()
        for device in devices:
            if lowered in device["name"].lower():
                return int(device["index"])
        raise ValueError(f"이름으로 일치하는 입력 장치를 찾지 못했습니다: {device_arg}")

    default_index = get_default_input_device_index()
    print_device_table(devices)
    if default_index is not None:
        answer = prompt(f"입력 장치 선택 [Enter={default_index}, 숫자 입력]: ")
    else:
        answer = prompt("입력 장치 선택 [숫자 입력, Enter=첫 번째 장치]: ")

    if not answer:
        if default_index is not None:
            return default_index
        return int(devices[0]["index"])

    if not answer.isdigit():
        raise ValueError("장치 번호는 숫자로 입력해야 합니다.")

    selected = int(answer)
    if not any(device["index"] == selected for device in devices):
        raise ValueError(f"입력 장치 {selected} 을(를) 찾지 못했습니다.")
    return selected


def get_device_name(device_index: int | None) -> str:
    if device_index is None:
        return "default"
    devices = list_input_devices()
    for device in devices:
        if device["index"] == device_index:
            return str(device["name"])
    return f"device_{device_index}"


def format_audio_error(exc: Exception) -> str:
    message = str(exc)
    if sys.platform == "darwin":
        return (
            f"{message}\n"
            "macOS에서 마이크 권한이 없을 수 있습니다. "
            "시스템 설정 > 개인정보 보호 및 보안 > 마이크에서 실행 중인 터미널 앱의 권한을 허용한 뒤 다시 시도하세요."
        )
    return message


def sanitize_session_name(session_name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in session_name)


def latest_session_dir(speaker_root: Path) -> Path | None:
    candidates = sorted(
        [path for path in speaker_root.glob("session_*") if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def prepare_session(
    *,
    speaker: str,
    session_arg: str | None,
    resume: bool,
    script_path: Path,
    script_lines: list[str],
    device_index: int | None,
) -> tuple[Path, dict[str, Any]]:
    speaker_root = RAW_DATA_ROOT / speaker
    speaker_root.mkdir(parents=True, exist_ok=True)

    if resume:
        if session_arg:
            session_dir = speaker_root / sanitize_session_name(session_arg)
        else:
            session_dir = latest_session_dir(speaker_root)
            if session_dir is None:
                raise FileNotFoundError("이어할 세션이 없습니다. 먼저 새 세션으로 녹음을 시작하세요.")

        log_path = session_dir / "session_log.json"
        if not log_path.exists():
            raise FileNotFoundError(f"session_log.json 을 찾을 수 없습니다: {log_path}")

        log_data = json.loads(log_path.read_text(encoding="utf-8"))
        if len(log_data.get("clips", [])) != len(script_lines):
            raise ValueError("기존 세션의 clip 수와 현재 대본 길이가 다릅니다. 동일한 대본으로 재개하세요.")
        log_data["device"] = {
            "index": device_index,
            "name": get_device_name(device_index),
        }
        log_data["updated_at"] = datetime.now().isoformat(timespec="seconds")
        return session_dir, log_data

    session_name = sanitize_session_name(session_arg or f"session_{now_stamp()}")
    session_dir = speaker_root / session_name
    if session_dir.exists():
        raise FileExistsError(f"세션 디렉토리가 이미 존재합니다: {session_dir}")
    session_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().isoformat(timespec="seconds")
    log_data = {
        "speaker": speaker,
        "session_name": session_name,
        "created_at": timestamp,
        "updated_at": timestamp,
        "script_path": str(script_path.resolve()),
        "sample_rate": SAMPLE_RATE,
        "channels": CHANNELS,
        "bit_depth": BIT_DEPTH,
        "device": {
            "index": device_index,
            "name": get_device_name(device_index),
        },
        "clips": build_empty_clips(script_lines),
    }
    return session_dir, log_data


def save_session_log(session_dir: Path, log_data: dict[str, Any]) -> None:
    atomic_write_json(session_dir / "session_log.json", log_data)


def render_sentence_screen(
    *,
    session_dir: Path,
    speaker: str,
    sentence_index: int,
    total_lines: int,
    sentence: str,
    clips: list[dict[str, Any]],
    color_enabled: bool,
) -> None:
    clear_screen()
    recorded, skipped, pending = summarize_progress(clips)
    print(colorize("Interactive Recording Helper", Ansi.BOLD + Ansi.CYAN, color_enabled))
    print(f"speaker: {speaker}")
    print(f"session: {session_dir.name}")
    print(f"progress: recorded={recorded}, skipped={skipped}, pending={pending}")
    print("-" * 72)
    print(colorize(f"[{sentence_index + 1}/{total_lines}] {sentence}", Ansi.BOLD, color_enabled))
    print("-" * 72)

    clip = clips[sentence_index]
    status = clip.get("rec_status", "pending")
    if status != "pending":
        status_text = f"현재 상태: {status}"
        if clip.get("duration_sec") is not None:
            status_text += (
                f" | duration={clip['duration_sec']}s"
                f" | peak={clip.get('peak_dbfs')} dB"
            )
        print(status_text)
    print("[Enter] 녹음 시작   r 재녹음   s 스킵   b 이전 문장   q 저장 후 종료")


def print_assessment(assessment: RecordingAssessment, *, color_enabled: bool) -> None:
    print(
        "저장 완료 | "
        f"duration={assessment.duration_sec:.2f}s | "
        f"peak={format_db(assessment.peak_dbfs)} | "
        f"SNR={assessment.snr_db:.1f} dB"
    )
    if assessment.warning_messages:
        for message in assessment.warning_messages:
            print(colorize(message, Ansi.YELLOW, color_enabled))
    else:
        print(colorize("입력 레벨과 길이가 권장 범위 안에 있습니다.", Ansi.GREEN, color_enabled))


def handle_phase0(args: argparse.Namespace, device_index: int | None) -> int:
    clear_screen()
    print("Phase 0 빠른 레퍼런스 녹음을 시작합니다.")
    if sys.stdin.isatty():
        print("준비가 되면 [Enter]를 눌러 10초 녹음을 시작하세요.")
        input("> ")
    else:
        # 비인터랙티브(stdin 없음) 환경: Claude Code '!' bash 등에서 실행 시 Enter 대기 불가
        print("비인터랙티브 모드 감지. 5초 카운트다운 후 10초 녹음을 시작합니다.")
        print("지금 문장을 준비하세요. 예시: \"안녕하세요, 저는 이효희입니다. 오늘 날씨가 참 좋네요.\"")
        for remaining in range(5, 0, -1):
            print(f"  {remaining}초 후 시작...")
            time.sleep(1)
        print("🔴 녹음 시작! 10초 동안 말씀하세요.")

    audio = record_audio(device_index=device_index, auto_stop_sec=PHASE0_DURATION_SEC)
    assessment = assess_recording(audio, SAMPLE_RATE)

    PHASE0_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    output_path = PHASE0_OUTPUT_ROOT / f"phase0_ref_{now_stamp()}.wav"
    sf.write(output_path, audio, SAMPLE_RATE, subtype=BIT_DEPTH)

    print_assessment(assessment, color_enabled=not args.no_color)
    print(f"저장 경로: {output_path}")
    return 0


def handle_interactive_session(args: argparse.Namespace, device_index: int | None) -> int:
    script_path = Path(args.script or DEFAULT_SCRIPT_PATH)
    script_lines = load_script_lines(script_path)

    session_dir, log_data = prepare_session(
        speaker=args.speaker,
        session_arg=args.session,
        resume=args.resume,
        script_path=script_path,
        script_lines=script_lines,
        device_index=device_index,
    )
    save_session_log(session_dir, log_data)

    clips = log_data["clips"]
    current_index = find_resume_start_index(clips, len(script_lines)) if args.resume else 0

    try:
        while current_index < len(script_lines):
            render_sentence_screen(
                session_dir=session_dir,
                speaker=args.speaker,
                sentence_index=current_index,
                total_lines=len(script_lines),
                sentence=script_lines[current_index],
                clips=clips,
                color_enabled=not args.no_color,
            )
            command = prompt("명령 입력 [Enter/r/s/b/q]: ")
            if command not in ("", "r", "s", "b", "q"):
                continue
            if command == "q":
                break
            if command == "b":
                current_index = max(0, current_index - 1)
                continue
            if command == "s":
                update_clip_entry(log_data, current_index, assessment=None, rec_status="skipped", saved_path=None)
                save_session_log(session_dir, log_data)
                current_index += 1
                continue

            clear_screen()
            print(colorize(f"[{current_index + 1}/{len(script_lines)}] 녹음 준비", Ansi.BOLD, not args.no_color))
            print(script_lines[current_index])
            print()
            audio = record_audio(device_index=device_index, auto_stop_sec=None)

            clip_path = session_dir / f"clip_{current_index + 1:03d}.wav"
            sf.write(clip_path, audio, SAMPLE_RATE, subtype=BIT_DEPTH)
            assessment = assess_recording(audio, SAMPLE_RATE)
            update_clip_entry(log_data, current_index, assessment=assessment, rec_status="recorded", saved_path=clip_path)
            save_session_log(session_dir, log_data)

            print_assessment(assessment, color_enabled=not args.no_color)
            followup = prompt("[Enter] 다음 문장 / r 재녹음 / b 이전 문장 / q 종료: ")
            if followup == "r":
                continue
            if followup == "b":
                current_index = max(0, current_index - 1)
                continue
            if followup == "q":
                break
            current_index += 1
    except KeyboardInterrupt:
        save_session_log(session_dir, log_data)
        print("\nCtrl+C 감지: 현재까지의 session_log.json 을 저장하고 종료합니다.")
        return 130

    save_session_log(session_dir, log_data)
    recorded, skipped, pending = summarize_progress(clips)
    print(f"세션 저장 완료: {session_dir}")
    print(f"recorded={recorded}, skipped={skipped}, pending={pending}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase 0 / Phase A 녹음을 위한 인터랙티브 WAV 녹음 도우미",
    )
    parser.add_argument("--speaker", help="화자 이름. raw 데이터 저장 경로에 사용됩니다.")
    parser.add_argument(
        "--script",
        default=str(DEFAULT_SCRIPT_PATH),
        help="녹음 대본 경로. 기본값은 recording_script_ko.txt 입니다.",
    )
    parser.add_argument("--device", help="입력 장치 index 또는 이름 일부")
    parser.add_argument("--session", help="세션 디렉토리 이름 지정")
    parser.add_argument("--resume", action="store_true", help="기존 session_log.json 기준으로 이어서 녹음")
    parser.add_argument("--phase0", action="store_true", help="대본을 무시하고 10초 레퍼런스 1회만 녹음")
    parser.add_argument("--list-devices", action="store_true", help="사용 가능한 입력 장치 목록만 출력하고 종료")
    parser.add_argument("--no-color", action="store_true", help="ANSI 색상 출력 비활성화")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.list_devices:
        devices = list_input_devices()
        print_device_table(devices)
        return 0

    if not args.speaker:
        parser.error("--speaker 는 필수입니다. 예: --speaker hyohee")

    try:
        device_index = resolve_device_index(args.device)
        if args.phase0:
            return handle_phase0(args, device_index)
        return handle_interactive_session(args, device_index)
    except Exception as exc:
        print(colorize(f"오류: {exc}", Ansi.RED, not args.no_color), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
