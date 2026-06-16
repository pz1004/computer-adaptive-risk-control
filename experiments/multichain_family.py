"""Reusable pre-specified threshold families for multi-chain CARC experiments."""
from __future__ import annotations

import numpy as np


def default_offset_schedules(num_thresholded_exits: int) -> list[tuple[str, np.ndarray]]:
    """Return deterministic per-exit offset schedules for a multi-chain family."""
    if num_thresholded_exits < 1:
        raise ValueError("num_thresholded_exits must be positive")
    if num_thresholded_exits == 4:
        return [
            ("global", np.array([0.00, 0.00, 0.00, 0.00])),
            ("early_strict", np.array([0.16, 0.10, -0.04, -0.08])),
            ("early_very_strict", np.array([0.24, 0.14, -0.08, -0.12])),
            ("mid_strict", np.array([0.10, 0.18, 0.02, -0.10])),
            ("late_permissive", np.array([0.12, 0.08, -0.12, -0.14])),
            ("balanced_staggered", np.array([0.06, 0.12, -0.02, -0.06])),
        ]

    positions = np.linspace(0.0, 1.0, num_thresholded_exits)
    mid_bump = np.exp(-0.5 * ((positions - 0.35) / 0.25) ** 2)
    return [
        ("global", np.zeros(num_thresholded_exits)),
        ("early_strict", np.linspace(0.16, -0.08, num_thresholded_exits)),
        ("early_very_strict", np.linspace(0.24, -0.12, num_thresholded_exits)),
        ("mid_strict", np.linspace(0.06, -0.10, num_thresholded_exits) + 0.10 * mid_bump),
        ("late_permissive", np.linspace(0.12, -0.14, num_thresholded_exits)),
        ("balanced_staggered", np.linspace(0.08, -0.06, num_thresholded_exits)),
    ]


def build_multichain_thresholds(
    num_exits: int,
    levels_per_chain: int,
    threshold_low: float,
    threshold_high: float,
) -> tuple[np.ndarray, list[list[int]], list[str]]:
    """
    Build threshold vectors and fixed-sequence chain orders.

    Configurations are appended from cheap to expensive within each chain; each
    returned chain order is expensive to cheap, which is what fixed-sequence
    testing expects.
    """
    if num_exits < 2:
        raise ValueError("num_exits must be at least 2")
    if levels_per_chain < 2:
        raise ValueError("levels_per_chain must be at least 2")
    if threshold_low >= threshold_high:
        raise ValueError("threshold_low must be less than threshold_high")

    base = np.linspace(threshold_low, threshold_high, levels_per_chain)
    schedules = default_offset_schedules(num_exits - 1)
    rows: list[np.ndarray] = []
    chains: list[list[int]] = []
    names: list[str] = []
    for name, offsets in schedules:
        start = len(rows)
        for level in base:
            rows.append(np.clip(level + offsets, 0.0, np.nextafter(1.0, 0.0)))
        end = len(rows)
        chains.append(list(range(end - 1, start - 1, -1)))
        names.append(name)
    return np.vstack(rows), chains, names
