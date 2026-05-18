from __future__ import annotations

import json
from pathlib import Path


class Manifest:
    """resumable 상태 + 산출 기록. 처리완료 소스키/클립/에러를 JSON으로 영속."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.done_keys: set[str] = set()
        self.clips: list[dict] = []
        self.errors: list[dict] = []
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.done_keys = set(data.get("done_keys", []))
            self.clips = data.get("clips", [])
            self.errors = data.get("errors", [])

    def done(self, src_key: str) -> bool:
        return src_key in self.done_keys

    def mark_done(self, src_key: str, n_clips: int) -> None:
        self.done_keys.add(src_key)

    def add_clip(self, record: dict) -> None:
        self.clips.append(record)

    def add_error(self, src: str, msg: str) -> None:
        self.errors.append({"src": src, "msg": msg})

    def stats(self) -> dict:
        total = sum(c.get("dur", 0.0) for c in self.clips)
        return {
            "n_clips": len(self.clips),
            "total_seconds": total,
            "n_errors": len(self.errors),
            "n_done_sources": len(self.done_keys),
        }

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "done_keys": sorted(self.done_keys),
            "clips": self.clips,
            "errors": self.errors,
            "stats": self.stats(),
        }
        self.path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
