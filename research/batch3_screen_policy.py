"""Adaptive early stop / prune rules for Batch 3 screening only."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ScreenStopConfig:
    warmup_epochs: int = 18
    improve_patience: int = 13  # must exceed plateau patience
    plateau_patience: int = 10
    prune_min_epoch: int = 30
    prune_mae_gap: float = 0.05
    prune_flat_window: int = 20
    prune_flat_max_gain: float = 0.008


def screening_leader_mae(state: dict) -> float | None:
    """Best val_insp_mae among completed screens in this batch."""
    vals = [
        float(s["val_insp_mae"])
        for s in state.get("screening", {}).values()
        if s.get("done") and s.get("val_insp_mae") is not None
    ]
    return min(vals) if vals else None


def should_prune_vs_leader(
    epoch: int,
    val_mae: float,
    leader_mae: float,
    history_vals: list[float],
    cfg: ScreenStopConfig,
) -> tuple[bool, str]:
    if epoch < cfg.prune_min_epoch or leader_mae is None:
        return False, ""
    if val_mae <= leader_mae + cfg.prune_mae_gap:
        return False, ""
    recent = history_vals[-cfg.prune_flat_window :]
    if len(recent) < cfg.prune_flat_window // 2:
        return False, ""
    gain = min(recent) - max(recent[: len(recent) // 2] or recent)
    flat = gain >= -cfg.prune_flat_max_gain  # no meaningful drop in window
    if flat:
        return True, (
            f"prune: val={val_mae:.4f} > leader+{cfg.prune_mae_gap} "
            f"(leader={leader_mae:.4f}) and flat last {len(recent)} epochs"
        )
    return False, ""


def check_screen_stop(
    epoch: int,
    stale: int,
    val_mae: float,
    history_vals: list[float],
    *,
    cfg: ScreenStopConfig,
    leader_mae: float | None,
) -> tuple[bool, str]:
    if epoch < cfg.warmup_epochs:
        return False, ""
    if stale >= cfg.improve_patience:
        return True, f"no_improve_{cfg.improve_patience}ep_after_warmup"
    if leader_mae is not None:
        ok, reason = should_prune_vs_leader(epoch, val_mae, leader_mae, history_vals, cfg)
        if ok:
            return True, reason
    return False, ""