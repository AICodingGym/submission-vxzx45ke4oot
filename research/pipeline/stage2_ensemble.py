"""Stage 2: better_than_median vs mean / median / inverse-MAE weights.

Published rule used with AmbrosM's neural stack (CPMP / public VPP notebooks):
if max-min spread is small, mean; else median. We also try inlier-mean.
All comparisons use our own aligned predictions — never external CSVs.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pipe_common import PIPE, insp_mae, insp_mae_hard, update_track


def better_than_median(preds: np.ndarray, spread_lim: float = 0.45) -> np.ndarray:
    """preds: (n_models, ...). Mean if models agree, median if they disagree."""
    spread = preds.max(axis=0) - preds.min(axis=0)
    return np.where(spread < spread_lim, preds.mean(axis=0), np.median(preds, axis=0))


def inlier_mean(preds: np.ndarray, radius: float = 0.45) -> np.ndarray:
    """Mean of models within `radius` of the median; median if none."""
    med = np.median(preds, axis=0, keepdims=True)
    w = (np.abs(preds - med) <= radius).astype(np.float64)
    denom = np.maximum(w.sum(axis=0, keepdims=True), 1.0)
    m = (preds * w).sum(axis=0, keepdims=True) / denom
    none = w.sum(axis=0) == 0
    out = np.squeeze(m, axis=0)
    med_s = np.squeeze(med, axis=0)
    return np.where(none, med_s, out)


def inv_mae_weights(preds: np.ndarray, y: np.ndarray, u_out: np.ndarray) -> np.ndarray:
    """Train-only weights: 1 / insp MAE, normalized. preds (M, N, 80)."""
    w = []
    for i in range(preds.shape[0]):
        mae = max(insp_mae(y, preds[i], u_out), 1e-6)
        w.append(1.0 / mae)
    w = np.array(w, dtype=np.float64)
    w = w / w.sum()
    return w


def weighted_mean(preds: np.ndarray, w: np.ndarray) -> np.ndarray:
    w = w.reshape([-1] + [1] * (preds.ndim - 1))
    return (preds * w).sum(axis=0)


def eval_ensembles(preds: np.ndarray, y: np.ndarray, u_out: np.ndarray, R=None, C=None) -> dict:
    """preds (n_models, n_breaths, 80). Labels only for scoring, not for btm itself."""
    out = {}
    methods = {
        "mean": preds.mean(axis=0),
        "median": np.median(preds, axis=0),
        "btm_0.25": better_than_median(preds, 0.25),
        "btm_0.45": better_than_median(preds, 0.45),
        "btm_0.70": better_than_median(preds, 0.70),
        "inlier_mean_0.45": inlier_mean(preds, 0.45),
    }
    w = inv_mae_weights(preds, y, u_out)
    methods["inv_mae_weighted"] = weighted_mean(preds, w)
    for name, pred in methods.items():
        rec = {"insp_mae": insp_mae(y, pred, u_out), "weights": None}
        if R is not None:
            rec["hard50_10"] = insp_mae_hard(y, pred, u_out, R, C)
        out[name] = rec
    out["inv_mae_weighted"]["weights"] = w.tolist()
    singles = [insp_mae(y, preds[i], u_out) for i in range(preds.shape[0])]
    out["members"] = singles
    best_name = min(
        (k for k in methods),
        key=lambda k: out[k]["insp_mae"],
    )
    out["best"] = best_name
    out["best_mae"] = out[best_name]["insp_mae"]
    return out


def main():
    path = PIPE / "ensemble_inputs.npz"
    if not path.exists():
        print("No ensemble_inputs.npz yet — stage 1 must write aligned preds.")
        return
    pack = np.load(path)
    preds = pack["preds"]
    y, u = pack["y"], pack["u_out"]
    R = pack["R"] if "R" in pack.files else None
    C = pack["C"] if "C" in pack.files else None
    res = eval_ensembles(preds, y, u, R, C)
    (PIPE / "stage2_ensemble.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    update_track(
        "neural_ensemble",
        {"status": "done", "protocol": "aligned holdout/OOF", **{k: v for k, v in res.items() if k != "inv_mae_weighted"}},
    )
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != "weights"}) for k, v in res.items()}, indent=2, default=float))


if __name__ == "__main__":
    main()
