"""Batch 3 experiment state — screening vs CV, resumable."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from batch2_common import OUT

STATE_PATH = OUT / "batch3_state.json"
CKPT_CV_PATH = OUT / "batch3_checkpoint.json"
PROMOTE_DELTA = 0.005
PROMOTE_HARD_DELTA = 0.005
TOP_K_SCREEN = 3
CV_FOLDS = 5
CV_SEED_DEFAULT = [42]
CV_MULTI_SEED = [42, 0, 1]

# Controls always get 1-seed GroupKFold when CV is approved
CV_CONTROLS = frozenset({"e1_baseline", "e2_multitask"})

# Primary hypotheses reported at end of Phase 1 (aug variants listed separately)
PRIMARY_SCREEN_KEYS = (
    "e1_baseline",
    "e2_multitask",
    "e3_aug_none",
    "e4_residual",
    "e5_hybrid",
)


def load_state() -> dict[str, Any]:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "protocol": "screen_holdout_then_cv",
        "screen_split": {"val_frac": 0.2, "seed": 42},
        "screening": {},
        "cv_queue": [],
        "cv_results": {},
        "promotions": {},
        "cv_plan": {},
        "baseline_audit": {},
        "approval": {"cv_approved": False},
    }


def save_state(state: dict) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2, default=float), encoding="utf-8")


def screening_done(state: dict, key: str) -> bool:
    s = state.get("screening", {}).get(key, {})
    return bool(s.get("done"))


def record_screening(
    state: dict,
    key: str,
    *,
    val_mae: float,
    hard_mae: float | None = None,
    best_epoch: int,
    recipe: dict,
    promote: bool | None = None,
    reason: str = "",
    extra: dict | None = None,
) -> None:
    row = {
        "result_type": "SCREENING",
        "split": f"holdout val_frac={state['screen_split']['val_frac']} "
                 f"split_seed={state['screen_split']['seed']}",
        "train_seed": state["screen_split"]["seed"],
        "done": True,
        "val_insp_mae": val_mae,
        "hard50_10_mae": hard_mae,
        "best_epoch": best_epoch,
        "recipe": recipe,
        "promote_to_cv": promote,
        "promote_reason": reason,
    }
    if extra:
        row.update(extra)
    state.setdefault("screening", {})[key] = row
    save_state(state)


def patch_screening_meta(state: dict, key: str, meta: dict) -> None:
    if key in state.get("screening", {}):
        state["screening"][key].update(meta)
        save_state(state)


def decide_promotions(state: dict, *, cv_seeds: list[int] | None = None) -> list[str]:
    """After screening, set cv_queue (does not run training)."""
    screening = state.get("screening", {})
    base = screening.get("e1_baseline", {})
    base_mae = float(base.get("val_insp_mae", 999.0))
    base_hard = float(base.get("hard50_10_mae", 999.0))

    promoted: set[str] = set()
    promotions: dict[str, dict[str, Any]] = {}

    for key in CV_CONTROLS:
        if screening_done(state, key):
            promoted.add(key)
            promotions[key] = {"promote": True, "reason": "control (always CV)"}

    for key, sc in screening.items():
        if key in CV_CONTROLS or key.startswith("e1_batch2"):
            continue
        mae = float(sc.get("val_insp_mae", 999.0))
        hard = sc.get("hard50_10_mae")
        hard_f = float(hard) if hard is not None else None
        reasons: list[str] = []
        if mae < base_mae - PROMOTE_DELTA:
            reasons.append(f"overall Δ={base_mae - mae:.4f} (≥{PROMOTE_DELTA})")
        if hard_f is not None and base_hard < 900 and hard_f < base_hard - PROMOTE_HARD_DELTA:
            reasons.append(f"hard50_10 Δ={base_hard - hard_f:.4f} (≥{PROMOTE_HARD_DELTA})")
        if reasons:
            promoted.add(key)
            promotions[key] = {"promote": True, "reason": "; ".join(reasons)}
        else:
            promotions[key] = {
                "promote": False,
                "reason": f"screen Δ overall {mae - base_mae:+.4f} (threshold {PROMOTE_DELTA})",
            }

    ranked = sorted(
        (
            (float(sc["val_insp_mae"]), key)
            for key, sc in screening.items()
            if key not in CV_CONTROLS
            and not key.startswith("e1_batch2")
            and key not in promoted
        ),
        key=lambda t: t[0],
    )
    for rank, (_, key) in enumerate(ranked[:TOP_K_SCREEN], start=1):
        promoted.add(key)
        promotions[key] = {
            "promote": True,
            "reason": f"top-{TOP_K_SCREEN} screening (#{rank} by val MAE, exploratory CV)",
        }

    for key, sc in screening.items():
        if key.startswith("e1_batch2"):
            sc["promote_to_cv"] = False
            sc["promote_reason"] = "baseline audit only"
            continue
        sc["promote_to_cv"] = key in promoted
        sc["promote_reason"] = promotions.get(key, {}).get("reason", "")

    ordered: list[str] = []
    for k in ("e1_baseline", "e2_multitask"):
        if k in promoted:
            ordered.append(k)
    for k in sorted(promoted):
        if k not in ordered:
            ordered.append(k)

    seeds = cv_seeds or CV_SEED_DEFAULT
    state["promotions"] = promotions
    state["cv_queue"] = ordered
    state["cv_plan"] = {
        "result_type": "PLAN",
        "n_configs": len(ordered),
        "configs": ordered,
        "n_folds": CV_FOLDS,
        "seeds_per_config": seeds,
        "estimated_training_runs": len(ordered) * CV_FOLDS * len(seeds),
        "note": "Phase 2 runs only after --run-cv-approved; multi-seed via --multi-seed-final <key>",
    }
    save_state(state)
    return ordered


def write_screening_report(state: dict | None = None) -> str:
    state = state or load_state()
    lines = [
        "BATCH 3 — PHASE 1 SCREENING REPORT",
        "=" * 60,
        f"Split: {state.get('screen_split')}",
        "",
        "Primary experiment holdout insp MAE (SCREENING):",
    ]
    for key in PRIMARY_SCREEN_KEYS:
        sc = state.get("screening", {}).get(key, {})
        if not sc:
            lines.append(f"  {key}: (not finished)")
            continue
        lines.append(
            f"  {key}: val={sc.get('val_insp_mae'):.4f} "
            f"hard50_10={sc.get('hard50_10_mae', 'NA')} promote={sc.get('promote_to_cv')}"
        )
    lines.extend(["", "Other screens:"])
    for key, sc in sorted(state.get("screening", {}).items()):
        if key in PRIMARY_SCREEN_KEYS or key.startswith("e1_batch2"):
            continue
        lines.append(f"  {key}: val={sc.get('val_insp_mae')} promote={sc.get('promote_to_cv')}")

    lines.extend(["", "Promotions:"])
    for key in state.get("cv_queue", []):
        p = state.get("promotions", {}).get(key, {})
        reason = p.get("reason") or state.get("screening", {}).get(key, {}).get("promote_reason", "")
        lines.append(f"  {key}: {reason}")

    plan = state.get("cv_plan", {})
    lines.extend([
        "",
        "Planned Phase 2 (not started until approved):",
        f"  configs={plan.get('configs')}",
        f"  estimated_training_runs={plan.get('estimated_training_runs')}",
        f"  seeds={plan.get('seeds_per_config')}",
        "",
        "Next: review batch3_baseline_audit.json, then:",
        "  python -u research/batch3_run.py --run-cv-approved",
    ])
    text = "\n".join(lines)
    (OUT / "batch3_screening_report.txt").write_text(text, encoding="utf-8")
    return text


def record_cv(state: dict, key: str, cv_payload: dict) -> None:
    cv_payload["result_type"] = "GROUP_KFOLD_CV"
    state.setdefault("cv_results", {})[key] = cv_payload
    save_state(state)


def set_cv_approved(state: dict, approved: bool = True) -> None:
    state.setdefault("approval", {})["cv_approved"] = approved
    save_state(state)