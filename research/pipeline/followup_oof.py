"""After PF tiny failed: ensemble e1+e3 OOF, tight PI on e3, lattice snap.

Prior models are the strongest neural; PF architecture is recorded as underfit.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inverse_control"))

from inv_common import snap_unique
from pipe_common import DATA, PIPE, RESEARCH, insp_mae, insp_mae_hard, update_track
from stage2_ensemble import eval_ensembles
from stage3_pi import run_overlay


def lattice_snap_eval(pred, y, u, R, C, uniq):
    snapped = snap_unique(pred.reshape(-1), uniq).reshape(pred.shape)
    return {
        "raw": insp_mae(y, pred, u),
        "snapped": insp_mae(y, snapped, u),
        "hard_raw": insp_mae_hard(y, pred, u, R, C),
        "hard_snapped": insp_mae_hard(y, snapped, u, R, C),
    }, snapped


def main():
    PIPE.mkdir(exist_ok=True)
    e3 = np.load(RESEARCH / "batch3_oof_e3_aug_uin.npz")
    e1 = np.load(RESEARCH / "batch3_oof_e1_baseline.npz")
    y, u, R, C, bids = e3["y"], e3["u_out"], e3["R"], e3["C"], e3["breath_id"]
    preds = np.stack([e1["oof"], e3["oof"]], axis=0)
    ens = eval_ensembles(preds, y, u, R, C)
    (PIPE / "stage2_e1e3_ensemble.json").write_text(json.dumps(ens, indent=2, default=float), encoding="utf-8")
    update_track(
        "neural_ensemble",
        {
            "status": "done",
            "protocol": "GroupKFold OOF aligned e1_baseline + e3_aug_uin",
            "members": {"e1": ens["members"][0], "e3": ens["members"][1]},
            "best": ens["best"],
            "best_mae": ens["best_mae"],
            "methods": {k: v["insp_mae"] for k, v in ens.items() if isinstance(v, dict) and "insp_mae" in v},
        },
    )
    print("ENSEMBLE", json.dumps({k: ens[k] if k in ("best", "best_mae", "members") else ens[k].get("insp_mae") if isinstance(ens[k], dict) and "insp_mae" in ens[k] else ens[k] for k in ens}, indent=2, default=float))

    # export e3 as overlay input
    e3_path = PIPE / "e3_oof_as_pred.npz"
    np.savez_compressed(e3_path, breath_id=bids, pred=e3["oof"], y=y, u_out=u, R=R, C=C)

    df = pd.read_csv(DATA / "train.csv", usecols=["pressure"])
    uniq = np.sort(df["pressure"].unique().astype(np.float64))
    snap_e3, _ = lattice_snap_eval(e3["oof"], y, u, R, C, uniq)
    best_ens_name = ens["best"]
    # rebuild best ensemble pred for snap
    from stage2_ensemble import better_than_median, inlier_mean, weighted_mean, inv_mae_weights
    w = inv_mae_weights(preds, y, u)
    ens_pred = {
        "mean": preds.mean(0),
        "median": np.median(preds, 0),
        "btm_0.45": better_than_median(preds, 0.45),
        "inlier_mean_0.45": inlier_mean(preds, 0.45),
        "inv_mae_weighted": weighted_mean(preds, w),
    }.get(best_ens_name, preds.mean(0))
    snap_ens, _ = lattice_snap_eval(ens_pred, y, u, R, C, uniq)
    snap_rep = {"e3": snap_e3, "ensemble_best": {"method": best_ens_name, **snap_ens}}
    (PIPE / "lattice_snap_oof.json").write_text(json.dumps(snap_rep, indent=2, default=float), encoding="utf-8")
    print("SNAP", json.dumps(snap_rep, indent=2, default=float))

    print("=== TIGHT PI on e3 OOF (400 breath gate) ===", flush=True)
    rec400 = run_overlay(str(e3_path), p0_k=80, max_breaths=400, tag="pi_e3_400")
    print("PI400", rec400.get("hybrid_insp_mae"), "vs nn", rec400.get("nn_insp_mae"), "trusted", rec400.get("mae_trusted"))
    if rec400.get("mae_trusted", 9) < 0.05 and rec400.get("delta_vs_nn", -1) > -0.002:
        print("=== TIGHT PI full e3 OOF ===", flush=True)
        rec = run_overlay(str(e3_path), p0_k=80, max_breaths=None, tag="pi_e3_full")
        update_track("inverse_pi_neural_fallback", {**rec, "neural": "e3_aug_uin OOF"})
        update_track("full_hybrid", {**rec, "neural": "e3_aug_uin OOF", "ensemble_mae": ens["best_mae"]})
    else:
        note = "tight PI did not beat e3 on 400-breath sample (false positives or no coverage)"
        update_track("inverse_pi_neural_fallback", {**rec400, "neural": "e3_aug_uin OOF", "note": note})
        update_track(
            "full_hybrid",
            {
                "status": "nn_only",
                "note": note,
                "e3_oof": float(ens["members"][1]),
                "ensemble_best": ens["best"],
                "ensemble_mae": ens["best_mae"],
                "pi400": rec400,
            },
        )
        print(note)


if __name__ == "__main__":
    main()
