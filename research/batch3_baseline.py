"""Batch 3 baseline equivalence vs Batch 2 best-known (document only + optional replay)."""
from __future__ import annotations

import json
from typing import Any

import numpy as np

from batch2_common import OUT, breath_split
from batch3_common import BATCH2_OOF, TrainRecipe, train_recipe_on_split, set_global_seed, reshape_sequences, SEQ_COLS_RICH
from batch3_state import load_state, save_state, screening_done

# Batch 2 E3 bilstm_rich on E2 split (same protocol: val_frac=0.2, SEED=42)
BATCH2_HOLDOUT = 0.23893257975578308
BATCH2_HOLDOUT_HARD = 0.26460716128349304
BATCH2_FINALIST_RECIPE = {
    "model": "BiLSTMReg",
    "cols": "SEQ_COLS_RICH",
    "hidden": 256,
    "layers": 3,
    "loss": "mse",
    "schedule": "plateau",
    "y_scale": True,
    "lr": 3e-3,
    "epochs": 90,
    "hard_weight": 1.0,
    "amp": False,
    "split": {"val_frac": 0.2, "seed": 42},
    "train_seed": 42,
}
E1_BATCH3_RECIPE = {
    "loss": "mse",
    "y_scale": True,
    "lr": 3e-3,
    "epochs": 75,
    "amp_default": True,
    "cols": "SEQ_COLS_RICH",
}


def document_baseline_audit(state: dict | None = None) -> dict[str, Any]:
    """Compare e1 screening to Batch 2 references; no extra training."""
    state = state or load_state()
    e1 = state.get("screening", {}).get("e1_baseline", {})
    audit = {
        "batch2_reference": {
            "groupkfold_oof_mae": BATCH2_OOF,
            "holdout_insp_mae": BATCH2_HOLDOUT,
            "holdout_hard50_10_mae": BATCH2_HOLDOUT_HARD,
            "recipe": BATCH2_FINALIST_RECIPE,
            "source": "batch2_results.json E3 best + E5 kfold_nn",
        },
        "batch3_e1_screen": {
            "val_insp_mae": e1.get("val_insp_mae"),
            "hard50_10_mae": e1.get("hard50_10_mae"),
            "best_epoch": e1.get("best_epoch"),
            "recipe": e1.get("recipe") or E1_BATCH3_RECIPE,
            "split": state.get("screen_split"),
            "train_amp": True,
        },
        "known_deltas_vs_batch2_holdout": {
            "holdout_mae": (
                float(e1["val_insp_mae"]) - BATCH2_HOLDOUT if e1.get("val_insp_mae") is not None else None
            ),
            "hard50_10_mae": (
                float(e1["hard50_10_mae"]) - BATCH2_HOLDOUT_HARD
                if e1.get("hard50_10_mae") is not None
                else None
            ),
        },
        "fairness_notes": [
            "Holdout split should match Batch 2 if val_frac=0.2 and split_seed=42 (breath_split).",
            "Batch 3 e1 uses epochs=75 and AMP; Batch 2 finalist holdout used epochs=90 and no AMP.",
            "OOF 0.2506 is GroupKFold on full train; screening holdout is a single 80/20 split.",
            "Use batch3 replay (no AMP, 90 ep) below for apples-to-apples holdout check.",
        ],
    }
    state["baseline_audit"] = audit
    save_state(state)
    (OUT / "batch3_baseline_audit.json").write_text(
        json.dumps(audit, indent=2, default=float), encoding="utf-8"
    )
    return audit


def replay_batch2_holdout(feat_df, state: dict | None = None) -> dict[str, Any]:
    """Optional: same split as screening, Batch 2 recipe, AMP off (skipped if cached)."""
    state = state or load_state()
    key = "e1_batch2_recipe_replay"
    if screening_done(state, key):
        return state["screening"][key]
    split = state.get("screen_split", {"val_frac": 0.2, "seed": 42})
    X, y, u, R, C, bids = reshape_sequences(feat_df, SEQ_COLS_RICH)
    tr_ids, va_ids = breath_split(
        np.unique(bids), val_frac=split["val_frac"], seed=split["seed"]
    )
    tr_idx = np.flatnonzero(np.isin(bids, list(tr_ids)))
    va_idx = np.flatnonzero(np.isin(bids, list(va_ids)))
    recipe = TrainRecipe(
        name="batch2_replay", epochs=90, lr=3e-3, loss="mse", y_scale=True, use_amp=False,
    )
    out = train_recipe_on_split(
        X, y, u, R, C, tr_idx, va_idx, recipe, seed=42, label=f"SCREEN:{key}",
    )
    from batch3_state import record_screening
    record_screening(
        state,
        key,
        val_mae=out["val_mae"],
        hard_mae=out["hard_mae"],
        best_epoch=out["best_epoch"],
        recipe={"replay_batch2_finalist": True, **BATCH2_FINALIST_RECIPE},
    )
    st = load_state()
    st["screening"][key]["result_type"] = "BASELINE_REPLAY"
    st["screening"][key]["delta_vs_batch2_holdout"] = float(out["val_mae"]) - BATCH2_HOLDOUT
    save_state(st)
    document_baseline_audit(st)
    return st["screening"][key]