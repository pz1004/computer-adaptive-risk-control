"""Small utilities shared by reproducibility scripts."""
from __future__ import annotations
import json
import subprocess
from pathlib import Path

import numpy as np
from scipy.stats import beta


def clopper_pearson(successes: int, trials: int, confidence: float = 0.95) -> list[float]:
    """Two-sided Clopper-Pearson interval for a binomial rate."""
    if trials <= 0:
        return [float("nan"), float("nan")]
    alpha = 1.0 - confidence
    lo = 0.0 if successes == 0 else beta.ppf(alpha / 2.0, successes, trials - successes + 1)
    hi = 1.0 if successes == trials else beta.ppf(1.0 - alpha / 2.0, successes + 1, trials - successes)
    return [float(lo), float(hi)]


def mean_or_none(values) -> float | None:
    """Return a JSON-friendly mean, or None for empty selections."""
    if not values:
        return None
    return float(np.mean(values))


def git_sha() -> str | None:
    """Best-effort current git commit hash for result provenance."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return proc.stdout.strip()


def write_json(path: str | Path, payload: dict) -> Path:
    """Write indented JSON, creating parent directories."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out
