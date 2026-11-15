"""Train on full data and write predictions.csv for AI Coding Gym / Kaggle format.

Choose recipe from batch3_best_recipe.json (written after CV OOF comparison only).
Do not iterate on leaderboard score. Default ensemble seeds only if multi-seed CV confirmed finalist.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from batch2_common import DATA, OUT, add_features, apply_norm, norm_fit, predict_np, reshape_sequences, BiLSTMReg
from batch3_common import SEQ_COLS_RICH, TrainRecipe, set_global_seed, train_model

PROJECT = Path(__file__).resolve().parents[1]


def main():
    recipe_path = OUT / "batch3_best_recipe.json"
    if recipe_path.exists():
        meta = json.loads(recipe_path.read_text(encoding="utf-8"))
    else:
        meta = {"use_ensemble_seeds": True, "default_recipe": "baseline"}
    seeds = [42, 0, 1] if meta.get("use_ensemble_seeds", True) else [42]

    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    feat_tr = add_features(train)
    feat_te = add_features(test)

    from batch2_common import breath_split

    tr_ids, va_ids = breath_split(feat_tr["breath_id"].unique(), val_frac=0.1, seed=42)
    tr_df = feat_tr[feat_tr["breath_id"].isin(tr_ids)]
    va_df = feat_tr[feat_tr["breath_id"].isin(va_ids)]
    Xtr, ytr, utr, _, _, _ = reshape_sequences(tr_df, SEQ_COLS_RICH)
    Xva, yva, uva, _, _, _ = reshape_sequences(va_df, SEQ_COLS_RICH)
    Xte, _, _, _, _, _ = reshape_sequences(feat_te, SEQ_COLS_RICH)
    mean, std = norm_fit(Xtr)
    Xtrn = apply_norm(Xtr, mean, std)
    Xvan = apply_norm(Xva, mean, std)
    Xten = apply_norm(Xte, mean, std)

    preds = []
    for seed in seeds:
        set_global_seed(seed)
        model = BiLSTMReg(Xtrn.shape[-1], 256, 3, 0.1)
        res = train_model(
            model, Xtrn, ytr, utr, Xvan, yva, uva,
            epochs=90, lr=3e-3, loss_kind="mse", schedule="plateau", y_scale=True,
            label=f"submit_s{seed}", log_every=15,
        )
        # For full train, we already trained; predict test
        pred_s = predict_np(res["model"], Xten) * res["y_std"] + res["y_mean"]
        preds.append(pred_s)
        print(f"seed {seed} done", flush=True)

    pred = np.mean(preds, axis=0).reshape(-1)

    test_sorted = feat_te.sort_values(["breath_id", "time_step"])
    sub = pd.read_csv(DATA / "sample_submission.csv")
    # align by id order in sample / test
    test_sorted = test_sorted.reset_index(drop=True)
    out_df = pd.DataFrame({"id": test_sorted["id"], "pressure": pred})
    # ensure same ids as sample
    out_df = out_df.sort_values("id")
    path = PROJECT / "predictions.csv"
    out_df.to_csv(path, index=False)
    print(f"Wrote {path} rows={len(out_df)} mean_p={out_df.pressure.mean():.4f}", flush=True)
    return path


if __name__ == "__main__":
    main()
