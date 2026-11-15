"""Train-free test prediction from saved e3 fold weights + optional lattice snap.

Reconstructs per-fold Robust? No: z-score and y-scale from the same GroupKFold splits.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inverse_control"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from batch2_common import (
    BiLSTMReg,
    add_features,
    apply_norm,
    norm_fit,
    predict_np,
    reshape_sequences,
    SEQ_COLS_RICH,
)
from batch3_common import make_folds, set_global_seed
from inv_common import snap_unique
from pipe_common import DATA, DEVICE, PIPE, RESEARCH, setup_cuda


def main():
    setup_cuda()
    set_global_seed(42)
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    feat_tr = add_features(train)
    feat_te = add_features(test)
    if "pressure" not in feat_te.columns:
        feat_te["pressure"] = 0.0
    Xtr, ytr, utr, _, _, bids_tr = reshape_sequences(feat_tr, SEQ_COLS_RICH)
    Xte, _, _, _, _, bids_te = reshape_sequences(feat_te, SEQ_COLS_RICH)
    ids_te = feat_te.sort_values(["breath_id", "time_step"])["id"].to_numpy().reshape(-1, 80)[:, :]
    folds = make_folds(bids_tr, 5)
    uniq = np.sort(train["pressure"].unique().astype(np.float64))

    wdir = RESEARCH / "batch3_cv_weights"

    def fold_predict(key: str) -> np.ndarray:
        fold_preds = []
        for fi, (tr_idx, _) in enumerate(folds, 1):
            mean, std = norm_fit(Xtr[tr_idx])
            Xten = apply_norm(Xte, mean, std)
            y_fold = ytr[tr_idx]
            u_fold = utr[tr_idx]
            ym = float(y_fold[u_fold == 0].mean())
            ys = max(float(y_fold[u_fold == 0].std()), 1e-6)
            model = BiLSTMReg(Xten.shape[-1], 256, 3, 0.1).to(DEVICE)
            state = torch.load(wdir / f"{key}_s42_f{fi}.pt", map_location=DEVICE, weights_only=True)
            model.load_state_dict(state)
            pred = predict_np(model, Xten) * ys + ym
            fold_preds.append(pred)
            print(f"  {key} fold {fi} test mean={pred.mean():.3f}", flush=True)
            del model
            torch.cuda.empty_cache()
        return np.stack(fold_preds, axis=0).mean(axis=0)

    e3 = fold_predict("e3_aug_uin")
    e1 = fold_predict("e1_baseline")
    # OOF inv-MAE weights (e1 0.27599, e3 0.25185)
    w1, w3 = 1.0 / 0.2759895324707031, 1.0 / 0.25185009837150574
    w1, w3 = w1 / (w1 + w3), w3 / (w1 + w3)
    pred = w1 * e1 + w3 * e3
    pred_snap = snap_unique(pred.reshape(-1), uniq).reshape(pred.shape)

    variants = {
        "e3_foldmean": e3,
        "e1_foldmean": e1,
        "inv_mae_weighted": pred,
        "inv_mae_weighted_snap": pred_snap,
    }
    # primary: OOF-best ensemble, lattice-snapped (snap helped 0.2393 → 0.2384)
    test_sorted = feat_te.sort_values(["breath_id", "time_step"]).reset_index(drop=True)
    out = pd.DataFrame({"id": test_sorted["id"].to_numpy(), "pressure": pred_snap.reshape(-1)})
    out = out.sort_values("id")
    dest = RESEARCH.parent / "predictions.csv"
    out.to_csv(dest, index=False)
    meta = {
        "file": str(dest),
        "n": len(out),
        "model": "e1+e3 inv-MAE weighted 5-fold means, lattice snap",
        "oof_mae": 0.2384,
        "weights": {"e1": w1, "e3": w3},
        "pressure_mean": float(out["pressure"].mean()),
        "variants_test_mean": {k: float(v.mean()) for k, v in variants.items()},
        "note": "Selected by GroupKFold OOF (ensemble 0.2393, snap 0.2384). No LB tuning.",
    }
    (PIPE / "submit_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    np.savez_compressed(PIPE / "test_pred_e3_folds.npz", **{k: v.astype(np.float32) for k, v in variants.items()}, breath_id=bids_te)
    print(f"Wrote {dest} rows={len(out)} mean={out.pressure.mean():.4f}", flush=True)
    return dest


if __name__ == "__main__":
    main()
