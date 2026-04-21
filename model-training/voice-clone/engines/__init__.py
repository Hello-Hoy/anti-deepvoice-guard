from __future__ import annotations

from importlib import import_module
from typing import Callable

from .base import VoiceCloneEngine


_ENGINE_REGISTRY: dict[str, type[VoiceCloneEngine]] = {}
_ENGINE_MODULES: dict[str, str] = {
    "xtts": "model_training.voice_clone.engines.xtts",
    "cosyvoice2": "model_training.voice_clone.engines.cosyvoice2",
    "gpt_sovits": "model_training.voice_clone.engines.gpt_sovits",
    "qwen3_tts": "model_training.voice_clone.engines.qwen3_tts",
    "indextts2": "model_training.voice_clone.engines.indextts2",
    "styletts2": "model_training.voice_clone.engines.styletts2",
    "rvc_postvc": "model_training.voice_clone.engines.rvc_postvc",
}


def register(name: str, cls: type[VoiceCloneEngine] | None = None) -> Callable[[type[VoiceCloneEngine]], type[VoiceCloneEngine]] | type[VoiceCloneEngine]:
    """엔진 클래스를 이름으로 등록하는 데코레이터."""

    def decorator(engine_cls: type[VoiceCloneEngine]) -> type[VoiceCloneEngine]:
        if not issubclass(engine_cls, VoiceCloneEngine):
            raise TypeError(
                f"Registered engine '{name}' must inherit VoiceCloneEngine, got {engine_cls!r}"
            )
        engine_cls.name = name
        _ENGINE_REGISTRY[name] = engine_cls
        return engine_cls

    if cls is not None:
        return decorator(cls)
    return decorator


def _import_engine_module(name: str) -> None:
    module_name = _ENGINE_MODULES.get(name)
    if module_name is None:
        known = ", ".join(_known_engines()) or "(none)"
        raise ValueError(f"Unknown voice clone engine '{name}'. Known engines: {known}")

    try:
        import_module(module_name)
    except Exception as exc:  # pragma: no cover - exercised through integration use
        raise RuntimeError(
            f"Failed to import voice clone engine '{name}' from module '{module_name}': {exc}"
        ) from exc


def _known_engines() -> list[str]:
    return sorted(set(_ENGINE_MODULES) | set(_ENGINE_REGISTRY))


def load_engine(name: str) -> VoiceCloneEngine:
    """필요할 때만 엔진 모듈을 import하고 인스턴스를 반환한다."""
    engine_cls = _ENGINE_REGISTRY.get(name)
    if engine_cls is None:
        _import_engine_module(name)
        engine_cls = _ENGINE_REGISTRY.get(name)
    if engine_cls is None:
        raise RuntimeError(
            f"Voice clone engine '{name}' was imported but did not call register()."
        )
    return engine_cls()


def list_engines() -> list[str]:
    """설치 여부와 무관하게 이 저장소가 지원하는 엔진 목록을 반환한다."""
    return sorted(_ENGINE_MODULES)


def list_available_engines() -> list[str]:
    """health_check를 통과하는 엔진만 반환한다."""
    available: list[str] = []
    for name in list_engines():
        try:
            engine = load_engine(name)
            ok, _message = engine.health_check()
        except Exception:
            continue
        if ok:
            available.append(name)
    return available


__all__ = [
    "VoiceCloneEngine",
    "list_available_engines",
    "list_engines",
    "load_engine",
    "register",
]
