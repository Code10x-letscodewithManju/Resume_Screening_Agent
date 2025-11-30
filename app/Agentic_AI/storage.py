import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from .config import LOG_DIR


RUNS_LOG = LOG_DIR / "runs.jsonl"


def log_run(
    jd_json: Dict[str, Any],
    weights: Dict[str, float],
    candidates: List[Dict[str, Any]],
) -> None:
    """
    Append a run entry to runs.jsonl for audit / debugging.
    Each line: {"timestamp": ..., "jd": ..., "weights": ..., "candidates": [...]}
    """
    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "jd": jd_json,
        "weights": weights,
        "candidates": candidates,
    }
    with RUNS_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
