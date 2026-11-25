from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from driveaudit_eval.research.consistency.verification_benchmark.common import read_json, write_json


def run_scenes(scenes: list[dict[str, Any]], worker: Callable[[dict[str, Any]], dict[str, Any]], output: str | Path,
               resume: bool = True) -> list[dict[str, Any]]:
    output = Path(output); results = []
    for scene in scenes:
        cache = output / scene["pair_id"] / "result.json"
        if resume and cache.is_file():
            result = read_json(cache); result["resumed"] = True
        else:
            try: result = {"scene_id": scene["pair_id"], "status": "completed", **worker(scene), "resumed": False}
            except Exception as error: result = {"scene_id": scene["pair_id"], "status": "failed", "failure_reason": f"{type(error).__name__}: {error}"}
            write_json(cache, result)
        results.append(result)
    write_json(output / "index.json", {"scene_count": len(scenes), "results": results, "per_scene_failure_isolation": True})
    return results

