#!/usr/bin/env python3
"""Small Flask MOS/deception test for real vs cloned clips."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
from typing import Any


@dataclass
class Clip:
    clip_id: str
    engine: str
    label: str
    path: Path
    reference: Path | None


def wilson_lower_bound(successes: int, n: int, z: float = 1.96) -> float:
    if n == 0:
        return 0.0
    phat = successes / n
    denom = 1 + z * z / n
    centre = phat + z * z / (2 * n)
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)
    return max(0.0, (centre - margin) / denom)


def _take_wavs(path: Path, n: int) -> list[Path]:
    return sorted(path.rglob("*.wav"))[:n] if path.exists() else []


def build_pool(real_dir: Path, gpt_dir: Path, openvoice_dir: Path, ref_wav: Path | None, seed: int) -> list[Clip]:
    clips: list[Clip] = []
    for label, engine, directory in (
        ("real", "real", real_dir),
        ("ai", "gpt_sovits", gpt_dir),
        ("ai", "openvoice2", openvoice_dir),
    ):
        for idx, path in enumerate(_take_wavs(directory, 10), start=1):
            clips.append(Clip(f"{engine}_{idx:02d}", engine, label, path.resolve(), ref_wav.resolve() if ref_wav else None))
    random.Random(seed).shuffle(clips)
    return clips


HTML = """<!doctype html>
<meta charset="utf-8">
<title>Human MOS Tool</title>
<h1>Human MOS Tool</h1>
<p>Clip {{index}} / {{total}}</p>
{% if ref %}<p>Reference</p><audio src="/media/ref/{{clip_id}}" controls></audio>{% endif %}
<p>Sample</p><audio id="player" src="/media/clip/{{clip_id}}" controls></audio>
<form method="post">
  <label><input type="radio" name="real_vs_ai" value="real" required> Real</label>
  <label><input type="radio" name="real_vs_ai" value="ai" required> AI</label><br>
  <label>Naturalness <input type="number" name="naturalness" min="1" max="5" required></label><br>
  <label>Speaker similarity <input type="number" name="similarity" min="1" max="5" required></label><br>
  <input type="hidden" name="plays" id="plays" value="0">
  <button type="submit">Next</button>
</form>
<script>
let plays = 0;
const player = document.getElementById("player");
player.addEventListener("play", () => {
  plays += 1;
  document.getElementById("plays").value = plays;
  if (plays >= 2) {
    player.onended = () => { player.controls = false; };
  }
});
</script>
"""


def run_flask(args: argparse.Namespace, pool: list[Clip]) -> int:
    try:
        from flask import Flask, Response, redirect, render_template_string, request, send_file, session, url_for
    except ModuleNotFoundError as exc:
        raise RuntimeError("Flask is required: pip install flask") from exc

    app = Flask(__name__)
    app.secret_key = args.secret
    responses: list[dict[str, Any]] = []
    by_id = {clip.clip_id: clip for clip in pool}

    @app.route("/", methods=["GET", "POST"])
    def index() -> Any:
        current = int(session.get("index", 0))
        if request.method == "POST" and current < len(pool):
            clip = pool[current]
            responses.append(
                {
                    "listener": args.listener,
                    "clip_id": clip.clip_id,
                    "engine": clip.engine,
                    "label": clip.label,
                    "guess": request.form["real_vs_ai"],
                    "naturalness": int(request.form["naturalness"]),
                    "similarity": int(request.form["similarity"]),
                    "plays": min(int(request.form.get("plays", "0")), 2),
                }
            )
            session["index"] = current + 1
            return redirect(url_for("index"))
        current = int(session.get("index", 0))
        if current >= len(pool):
            write_outputs(args.out_dir, args.listener, responses)
            return "Done. Results written."
        clip = pool[current]
        return render_template_string(HTML, index=current + 1, total=len(pool), clip_id=clip.clip_id, ref=clip.reference)

    @app.route("/media/clip/<clip_id>")
    def media_clip(clip_id: str) -> Any:
        return send_file(by_id[clip_id].path)

    @app.route("/media/ref/<clip_id>")
    def media_ref(clip_id: str) -> Any:
        ref = by_id[clip_id].reference
        if ref is None:
            return Response("no reference", status=404)
        return send_file(ref)

    app.run(host=args.host, port=args.port, debug=False)
    return 0


def write_outputs(out_dir: Path, listener: str, responses: list[dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{listener}.csv"
    if responses:
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(responses[0].keys()))
            writer.writeheader()
            writer.writerows(responses)
    aggregate: dict[str, Any] = {}
    for engine in sorted({row["engine"] for row in responses if row["engine"] != "real"}):
        ai_rows = [row for row in responses if row["engine"] == engine]
        deceived = sum(1 for row in ai_rows if row["guess"] == "real")
        aggregate[engine] = {
            "n": len(ai_rows),
            "deception_rate": deceived / len(ai_rows) if ai_rows else 0.0,
            "wilson_95_lower": wilson_lower_bound(deceived, len(ai_rows)),
            "naturalness_mean": sum(row["naturalness"] for row in ai_rows) / len(ai_rows) if ai_rows else 0.0,
            "similarity_mean": sum(row["similarity"] for row in ai_rows) / len(ai_rows) if ai_rows else 0.0,
        }
    (out_dir / "aggregate.json").write_text(json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Human MOS/deception test UI")
    parser.add_argument("--real-dir", type=Path, required=True)
    parser.add_argument("--gpt-sovits-dir", type=Path, required=True)
    parser.add_argument("--openvoice2-dir", type=Path, required=True)
    parser.add_argument("--ref-wav", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("reports/human_mos"))
    parser.add_argument("--listener", default="listener_001")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--secret", default="local-mos-session")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pool = build_pool(args.real_dir, args.gpt_sovits_dir, args.openvoice2_dir, args.ref_wav, args.seed)
    if len(pool) != 30:
        print(f"warning: expected 30 clips, found {len(pool)}")
    return run_flask(args, pool)


if __name__ == "__main__":
    raise SystemExit(main())
