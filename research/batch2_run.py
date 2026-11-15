"""
Batch 2 runner — see batch2_hypotheses.md.
Does not start Batch 3.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold

from batch2_common import (
    DEVICE,
    FEATS_TREE,
    OUT,
    SEED,
    SEQ_COLS_BASE,
    SEQ_COLS_RICH,
    AttnBiLSTM,
    BiLSTMReg,
    ConvBiLSTM,
    add_features,
    apply_norm,
    breath_split,
    insp_mae,
    insp_mae_by_rc,
    norm_fit,
    predict_np,
    reshape_sequences,
    train_model,
)
from batch2_common import DATA

lines: list[str] = []


def log(msg: str = "") -> None:
    print(msg, flush=True)
    lines.append(msg)


# =========================================================================
# E1 — Memorization diagnosis
# =========================================================================
def e1_memorization(feat_all: pd.DataFrame):
    log("=" * 72)
    log("E1: MEMORIZATION DIAGNOSIS")
    log("H: target scaling and/or LR mismatch block near-zero train error on 500 breaths.")
    log("=" * 72)

    rng = np.random.default_rng(0)
    ids = rng.choice(feat_all["breath_id"].unique(), size=500, replace=False)
    sub = feat_all[feat_all["breath_id"].isin(ids)]
    X, y, u, R, C, _ = reshape_sequences(sub, SEQ_COLS_BASE)
    mean, std = norm_fit(X)
    Xn = apply_norm(X, mean, std)

    probes = []
    configs = [
        # label, y_scale, loss, lr, epochs, hidden
        ("raw_mae_lr1e3", False, "mae", 1e-3, 120, 256),
        ("raw_mae_lr3e3", False, "mae", 3e-3, 120, 256),
        ("raw_mse_lr1e3", False, "mse", 1e-3, 120, 256),
        ("scaled_mae_lr1e3", True, "mae", 1e-3, 120, 256),
        ("scaled_mse_lr1e3", True, "mse", 1e-3, 120, 256),
        ("scaled_mse_lr3e3", True, "mse", 3e-3, 120, 256),
        ("scaled_huber_lr2e3", True, "huber", 2e-3, 120, 256),
        ("scaled_mse_lr1e2_wide", True, "mse", 1e-2, 150, 384),
    ]
    for label, y_scale, loss, lr, epochs, hidden in configs:
        model = BiLSTMReg(Xn.shape[-1], hidden=hidden, layers=3, dropout=0.0)
        # train=val same set for pure memorization
        res = train_model(
            model, Xn, y, u, Xn, y, u,
            R_tr=R, C_tr=C, R_va=R, C_va=C,
            epochs=epochs, lr=lr, batch_size=64, loss_kind=loss,
            # no plateau for pure memorization — keep LR up so we can actually fit
            schedule="cosine", y_scale=y_scale, patience=8,
            label=label, log_every=30, min_lr=1e-6,
        )
        probes.append({
            "label": label, "y_scale": y_scale, "loss": loss, "lr": lr,
            "train_mae": res["best_train"], "val_mae_same": res["best_val"],
            "best_epoch": res["best_epoch"],
        })
        log(f"E1 {label}: best_train_insp_mae={res['best_train']:.4f} ep={res['best_epoch']}")

    best = min(probes, key=lambda d: d["train_mae"])
    log(f"E1 winner: {best['label']} train_mae={best['train_mae']:.4f}")
    memorized = best["train_mae"] < 0.05
    log(f"E1 near-zero memorization achieved? {memorized}")
    return {"probes": probes, "winner": best, "memorized": memorized}


# =========================================================================
# E4 — Hard case analysis (run early; no GPU needed long)
# =========================================================================
def e4_hard_cases(train_raw: pd.DataFrame, feat_all: pd.DataFrame):
    log()
    log("=" * 72)
    log("E4: HARD CASES — R=50, C=10")
    log("H: stiff/resistive regime has sharper ΔP, distinct u_in→P maps, late accumulation.")
    log("=" * 72)

    meta = train_raw.groupby("breath_id").agg(
        R=("R", "first"), C=("C", "first"),
    ).reset_index()
    hard_ids = set(meta[(meta.R == 50) & (meta.C == 10)]["breath_id"])
    easy_ids = set(meta[(meta.R == 5) & (meta.C == 50)]["breath_id"])
    log(f"n hard R50C10 breaths={len(hard_ids)} easy R5C50={len(easy_ids)}")

    def breath_stats(ids, name):
        sub = feat_all[feat_all["breath_id"].isin(ids)]
        insp = sub[sub["u_out"] == 0]
        rows = {
            "name": name,
            "u_in_mean": insp["u_in"].mean(),
            "u_in_max": insp["u_in"].max(),
            "u_in_std": insp["u_in"].std(),
            "du_abs_mean": insp["du"].abs().mean(),
            "pressure_mean": insp["pressure"].mean(),
            "pressure_max": insp["pressure"].max(),
            "pressure_std": insp["pressure"].std(),
            "dp_abs_mean": insp.groupby("breath_id")["pressure"].diff().abs().mean(),
            "vol_over_c_mean": insp["vol_over_c"].mean(),
            "vol_r_over_c_mean": insp["vol_r_over_c"].mean(),
            "corr_uin_p": insp["u_in"].corr(insp["pressure"]),
            "corr_volc_p": insp["vol_over_c"].corr(insp["pressure"]),
            "corr_area_p": insp["area_uin"].corr(insp["pressure"]),
        }
        # late vs early insp
        tmp = insp.copy()
        tmp["istep"] = tmp.groupby("breath_id").cumcount()
        early = tmp[tmp["istep"] <= 5]["pressure"]
        late = tmp[tmp["istep"] >= 20]["pressure"]
        rows["p_early_mean"] = early.mean()
        rows["p_late_mean"] = late.mean()
        rows["p_late_minus_early"] = rows["p_late_mean"] - rows["p_early_mean"]
        return rows

    hard_s = breath_stats(hard_ids, "R50C10")
    easy_s = breath_stats(easy_ids, "R5C50")
    mid_ids = set(meta[(meta.R == 20) & (meta.C == 20)]["breath_id"])
    mid_s = breath_stats(mid_ids, "R20C20")
    for s in (hard_s, mid_s, easy_s):
        log(
            f"{s['name']}: u_in_mean={s['u_in_mean']:.2f} |dp|={s['dp_abs_mean']:.3f} "
            f"Pmean={s['pressure_mean']:.2f} Plate-Pearly={s['p_late_minus_early']:.2f} "
            f"corr(u,P)={s['corr_uin_p']:.3f} corr(V/C,P)={s['corr_volc_p']:.3f} "
            f"vol_r/c={s['vol_r_over_c_mean']:.2f}"
        )

    # Linear slope pressure ~ vol_over_c within regime (physics check)
    def fit_slope(ids):
        sub = feat_all[feat_all["breath_id"].isin(ids) & (feat_all["u_out"] == 0)]
        x = sub["vol_over_c"].to_numpy()
        y = sub["pressure"].to_numpy()
        x = np.column_stack([x, np.ones_like(x)])
        coef, *_ = np.linalg.lstsq(x, y, rcond=None)
        pred = x @ coef
        return float(coef[0]), float(np.mean(np.abs(y - pred)))

    for name, ids in [("R50C10", hard_ids), ("R20C20", mid_ids), ("R5C50", easy_ids)]:
        slope, mae = fit_slope(ids)
        log(f"P ~ a*(cumsum_u/C)+b  [{name}]: a={slope:.4f} resid_MAE={mae:.3f}")

    # Repeated control patterns within hard regime
    rng = np.random.default_rng(1)
    sample_hard = rng.choice(list(hard_ids), size=min(3000, len(hard_ids)), replace=False)
    fps = {}
    for bid in sample_hard:
        g = feat_all[feat_all["breath_id"] == bid].sort_values("time_step")
        key = tuple(np.round(g["u_in"].to_numpy()[:30], 1))
        fps.setdefault(key, []).append(bid)
    multi = {k: v for k, v in fps.items() if len(v) >= 2}
    log(f"Hard-regime coarse u_in fingerprints with >=2 breaths (n=3000 sample): {len(multi)}")
    if multi:
        cons = []
        for bids in list(multi.values())[:50]:
            ps = []
            for b in bids:
                g = feat_all[feat_all["breath_id"] == b]
                ps.append(g.loc[g.u_out == 0, "pressure"].to_numpy())
            L = min(map(len, ps))
            arr = np.stack([p[:L] for p in ps])
            cons.append(np.mean(np.abs(arr - arr.mean(0))))
        log(f"Within-fingerprint pressure MAE to mean: mean={np.mean(cons):.3f} n={len(cons)}")

    # Late-breath error structure using a quick LGBM on subsample for residuals
    log("Fitting quick LGBM on 15k breaths to inspect late-step hard residuals...")
    tr_ids, va_ids = breath_split(feat_all["breath_id"].unique(), val_frac=0.2, seed=SEED)
    # subsample train for speed
    rng = np.random.default_rng(2)
    tr_list = list(tr_ids)
    tr_sub = set(rng.choice(tr_list, size=min(15000, len(tr_list)), replace=False).tolist())
    tr = feat_all[feat_all["breath_id"].isin(tr_sub)]
    va = feat_all[feat_all["breath_id"].isin(va_ids)]
    tr_insp = tr[tr["u_out"] == 0]
    dtrain = lgb.Dataset(tr_insp[FEATS_TREE], label=tr_insp["pressure"])
    dvalid = lgb.Dataset(va.loc[va.u_out == 0, FEATS_TREE], label=va.loc[va.u_out == 0, "pressure"])
    booster = lgb.train(
        {"objective": "regression_l1", "learning_rate": 0.1, "num_leaves": 96,
         "verbosity": -1, "seed": SEED, "n_jobs": 4},
        dtrain, num_boost_round=300,
        valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    va = va.copy()
    va["pred"] = booster.predict(va[FEATS_TREE])
    va["err"] = va["pressure"] - va["pred"]
    va["abs_err"] = va["err"].abs()
    insp = va[va["u_out"] == 0].copy()
    insp["istep"] = insp.groupby("breath_id").cumcount()
    hard = insp[(insp.R == 50) & (insp.C == 10)]
    other = insp[~((insp.R == 50) & (insp.C == 10))]
    log("Late-step (|istep>=20) MAE: "
        f"hard={hard.loc[hard.istep>=20,'abs_err'].mean():.3f} "
        f"other={other.loc[other.istep>=20,'abs_err'].mean():.3f}")
    log("Early-step (|istep<=5) MAE: "
        f"hard={hard.loc[hard.istep<=5,'abs_err'].mean():.3f} "
        f"other={other.loc[other.istep<=5,'abs_err'].mean():.3f}")
    # signed bias late
    log(f"Hard late signed bias (pred-true): "
        f"{(-hard.loc[hard.istep>=20,'err']).mean():.3f}  "
        f"(positive => under-prediction of pressure)")

    return {
        "hard": hard_s, "easy": easy_s, "mid": mid_s,
        "n_hard": len(hard_ids), "n_fingerprint_dups": len(multi),
    }


# =========================================================================
# E2 — Push sequence model
# =========================================================================
def e2_push_sequence(feat_all: pd.DataFrame, e1_winner: dict):
    log()
    log("=" * 72)
    log("E2: PUSH SEQUENCE MODEL (schedule / loss / size)")
    log("H: underfit; plateau sched + better loss/scale/size beats 0.375.")
    log("=" * 72)

    tr_ids, va_ids = breath_split(feat_all["breath_id"].unique(), 0.2, SEED)
    tr = feat_all[feat_all["breath_id"].isin(tr_ids)]
    va = feat_all[feat_all["breath_id"].isin(va_ids)]
    Xtr, ytr, utr, Rtr, Ctr, _ = reshape_sequences(tr, SEQ_COLS_BASE)
    Xva, yva, uva, Rva, Cva, _ = reshape_sequences(va, SEQ_COLS_BASE)
    mean, std = norm_fit(Xtr)
    Xtrn, Xvan = apply_norm(Xtr, mean, std), apply_norm(Xva, mean, std)

    # Use E1-informed defaults
    y_scale = bool(e1_winner.get("y_scale", True))
    prefer_loss = e1_winner.get("loss", "mae")
    prefer_lr = float(e1_winner.get("lr", 1e-3))
    log(f"E2 using E1-informed y_scale={y_scale} loss_hint={prefer_loss} lr_hint={prefer_lr}")

    screens = []
    # Compact but decisive screen; epochs generous with plateau early-stop via LR decay.
    configs = [
        dict(label="baseline_cos_mae_noscale_h256", loss="mae", sched="cosine", y_scale=False,
             hidden=256, layers=3, lr=8e-4, epochs=80, hard_w=1.0),
        dict(label="plateau_mae_scale", loss="mae", sched="plateau", y_scale=True,
             hidden=256, layers=3, lr=1e-3, epochs=90, hard_w=1.0),
        dict(label="plateau_mse_scale", loss="mse", sched="plateau", y_scale=True,
             hidden=256, layers=3, lr=1e-3, epochs=90, hard_w=1.0),
        dict(label="plateau_huber_scale", loss="huber", sched="plateau", y_scale=True,
             hidden=256, layers=3, lr=1e-3, epochs=90, hard_w=1.0),
        dict(label="plateau_mse_scale_lr3e3", loss="mse", sched="plateau", y_scale=True,
             hidden=256, layers=3, lr=3e-3, epochs=90, hard_w=1.0),
        dict(label="plateau_mae_scale_hardw3", loss="mae", sched="plateau", y_scale=True,
             hidden=256, layers=3, lr=1e-3, epochs=90, hard_w=3.0),
    ]
    # Prefer putting E1 winner recipe first if mse+scale
    results = []
    for cfg in configs:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model = BiLSTMReg(Xtrn.shape[-1], hidden=cfg["hidden"], layers=cfg["layers"], dropout=0.1)
        res = train_model(
            model, Xtrn, ytr, utr, Xvan, yva, uva,
            R_tr=Rtr, C_tr=Ctr, R_va=Rva, C_va=Cva,
            epochs=cfg["epochs"], lr=cfg["lr"], loss_kind=cfg["loss"],
            schedule=cfg["sched"], y_scale=cfg["y_scale"],
            hard_weight=cfg["hard_w"], patience=10, label=cfg["label"], log_every=10,
            batch_size=256,
        )
        row = {
            **cfg,
            "best_val": res["best_val"],
            "best_train": res["best_train"],
            "best_epoch": res["best_epoch"],
            "best_hard": res["best_hard"],
        }
        results.append(row)
        # keep only best full res in memory; drop model weights for losers
        screens.append({"cfg": cfg["label"], "res": res})
        log(
            f"E2 {cfg['label']}: val={res['best_val']:.4f} train={res['best_train']:.4f} "
            f"hard={res['best_hard']:.4f} ep={res['best_epoch']}"
        )
        # free non-best models
        if len(screens) >= 2:
            best_so_far = min(results, key=lambda d: d["best_val"])["label"]
            for s in screens[:-1]:
                if s["cfg"] != best_so_far and s.get("res") is not None:
                    s["res"]["model"] = None
                    s["res"]["pred_va"] = s["res"]["pred_va"]  # keep preds
                    if "state" in s["res"]:
                        pass

    best_row = min(results, key=lambda d: d["best_val"])
    best_res = next(s["res"] for s in screens if s["cfg"] == best_row["label"])
    log(f"E2 best screen: {best_row['label']} val={best_row['best_val']:.4f} "
        f"(prior best-known 0.3747)")
    return {
        "results": results,
        "best": best_row,
        "best_res": best_res,
        "split": {"tr_ids": tr_ids, "va_ids": va_ids},
        "arrays": {
            "Xtrn": Xtrn, "Xvan": Xvan, "ytr": ytr, "yva": yva,
            "utr": utr, "uva": uva, "Rtr": Rtr, "Rva": Rva, "Ctr": Ctr, "Cva": Cva,
            "mean": mean, "std": std, "tr": tr, "va": va,
        },
    }


# =========================================================================
# E3 — Architecture / input changes
# =========================================================================
def e3_architecture(feat_all: pd.DataFrame, e2: dict):
    log()
    log("=" * 72)
    log("E3: ARCHITECTURE / INPUT CHANGES")
    log("H: conv/attention + richer hist feats help overall and especially R=50,C=10.")
    log("=" * 72)

    tr_ids, va_ids = e2["split"]["tr_ids"], e2["split"]["va_ids"]
    tr = feat_all[feat_all["breath_id"].isin(tr_ids)]
    va = feat_all[feat_all["breath_id"].isin(va_ids)]
    best_cfg = e2["best"]

    def prep(cols):
        Xtr, ytr, utr, Rtr, Ctr, _ = reshape_sequences(tr, cols)
        Xva, yva, uva, Rva, Cva, _ = reshape_sequences(va, cols)
        mean, std = norm_fit(Xtr)
        return (apply_norm(Xtr, mean, std), apply_norm(Xva, mean, std),
                ytr, yva, utr, uva, Rtr, Rva, Ctr, Cva)

    results = []
    # Carry forward best training recipe from E2
    y_scale = best_cfg["y_scale"]
    loss = best_cfg["loss"]
    lr = best_cfg["lr"]
    hard_w = best_cfg.get("hard_w", 1.0)

    variants = [
        ("bilstm_rich_feats", "bilstm", SEQ_COLS_RICH, 256, 3),
        ("conv_bilstm_rich", "conv", SEQ_COLS_RICH, 256, 3),
        ("attn_bilstm_rich", "attn", SEQ_COLS_RICH, 256, 3),
        ("conv_bilstm_rich_hardw3", "conv", SEQ_COLS_RICH, 256, 3),
    ]
    best_pack = None
    for label, kind, cols, hidden, layers in variants:
        Xtrn, Xvan, ytr, yva, utr, uva, Rtr, Rva, Ctr, Cva = prep(cols)
        hw = 3.0 if "hardw3" in label else hard_w
        if kind == "bilstm":
            model = BiLSTMReg(Xtrn.shape[-1], hidden=hidden, layers=layers, dropout=0.1)
        elif kind == "conv":
            model = ConvBiLSTM(Xtrn.shape[-1], hidden=hidden, layers=layers, dropout=0.1)
        else:
            model = AttnBiLSTM(Xtrn.shape[-1], hidden=hidden, layers=layers, dropout=0.1)
        res = train_model(
            model, Xtrn, ytr, utr, Xvan, yva, uva,
            R_tr=Rtr, C_tr=Ctr, R_va=Rva, C_va=Cva,
            epochs=90, lr=lr, loss_kind=loss, schedule="plateau",
            y_scale=y_scale, hard_weight=hw, patience=10,
            label=label, log_every=10,
        )
        row = {
            "label": label, "kind": kind, "n_feat": Xtrn.shape[-1],
            "best_val": res["best_val"], "best_train": res["best_train"],
            "best_hard": res["best_hard"], "best_epoch": res["best_epoch"],
        }
        results.append(row)
        log(
            f"E3 {label}: val={res['best_val']:.4f} train={res['best_train']:.4f} "
            f"hard={res['best_hard']:.4f}"
        )
        if best_pack is None or res["best_val"] < best_pack["res"]["best_val"]:
            best_pack = {
                "row": row, "res": res, "cols": cols, "kind": kind,
                "Xtrn": Xtrn, "Xvan": Xvan, "ytr": ytr, "yva": yva,
                "utr": utr, "uva": uva, "Rtr": Rtr, "Rva": Rva, "Ctr": Ctr, "Cva": Cva,
                "tr": tr, "va": va, "hw": hw, "hidden": hidden, "layers": layers,
                "y_scale": y_scale, "loss": loss, "lr": lr,
            }

    log(f"E3 best: {best_pack['row']['label']} val={best_pack['row']['best_val']:.4f}")
    return {"results": results, "best": best_pack}


# =========================================================================
# E5 — Ensemble + K-fold finalist
# =========================================================================
def e5_ensemble_and_kfold(feat_all: pd.DataFrame, e2: dict, e3: dict):
    log()
    log("=" * 72)
    log("E5: ENSEMBLING + GROUPED K-FOLD FINALIST")
    log("H: tree/LSTM errors partly complementary; blends can win; finalize with GroupKFold.")
    log("=" * 72)

    # Pick best neural so far between E2 and E3
    e2_val = e2["best"]["best_val"]
    e3_val = e3["best"]["row"]["best_val"]
    if e3_val <= e2_val:
        nn_pack = e3["best"]
        nn_name = e3["best"]["row"]["label"]
        va = nn_pack["va"]
        yva, uva = nn_pack["yva"], nn_pack["uva"]
        nn_pred = nn_pack["res"]["pred_va"]
        log(f"Using E3 model as NN side: {nn_name} val={e3_val:.4f}")
    else:
        nn_pack = None
        nn_name = e2["best"]["label"]
        arr = e2["arrays"]
        va = arr["va"]
        yva, uva = arr["yva"], arr["uva"]
        nn_pred = e2["best_res"]["pred_va"]
        log(f"Using E2 model as NN side: {nn_name} val={e2_val:.4f}")

    # Train full-data-split tree on same tr/va ids
    tr_ids = e2["split"]["tr_ids"]
    va_ids = e2["split"]["va_ids"]
    tr = feat_all[feat_all["breath_id"].isin(tr_ids)]
    va_df = feat_all[feat_all["breath_id"].isin(va_ids)].sort_values(["breath_id", "time_step"])
    tr_insp = tr[tr["u_out"] == 0]
    dtrain = lgb.Dataset(tr_insp[FEATS_TREE].to_numpy(np.float32),
                         label=tr_insp["pressure"].to_numpy(np.float32))
    dvalid = lgb.Dataset(
        va_df.loc[va_df.u_out == 0, FEATS_TREE].to_numpy(np.float32),
        label=va_df.loc[va_df.u_out == 0, "pressure"].to_numpy(np.float32),
    )
    booster = lgb.train(
        {"objective": "regression_l1", "learning_rate": 0.08, "num_leaves": 127,
         "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1,
         "verbosity": -1, "seed": SEED, "n_jobs": 4, "min_data_in_leaf": 40},
        dtrain, num_boost_round=800,
        valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(200)],
    )
    tree_pred_flat = booster.predict(va_df[FEATS_TREE].to_numpy(np.float32),
                                     num_iteration=booster.best_iteration)
    y_flat = va_df["pressure"].to_numpy()
    u_flat = va_df["u_out"].to_numpy()
    tree_mae = insp_mae(y_flat, tree_pred_flat, u_flat)
    log(f"E5 tree (insp-rows, insp-ES) val MAE={tree_mae:.4f} iter={booster.best_iteration}")

    # Align NN preds: nn_pred is [n_breaths, 80] in sorted breath order of va used in training
    # Rebuild NN pred on va_df order via saved model if possible; else assume same sort.
    # Safest: re-predict with best NN model on va sorted.
    if nn_pack is not None:
        model = nn_pack["res"]["model"]
        y_mean, y_std = nn_pack["res"]["y_mean"], nn_pack["res"]["y_std"]
        Xvan = nn_pack["Xvan"]
        pred_s = predict_np(model, Xvan)
        nn_pred_b = pred_s * y_std + y_mean  # [B,80]
        # map breath order
        va_nn = nn_pack["va"].sort_values(["breath_id", "time_step"])
        nn_flat = nn_pred_b.reshape(-1)
        # va_df same ids sorted identically
        assert np.array_equal(
            va_nn["breath_id"].to_numpy(),
            va_df["breath_id"].to_numpy(),
        )
    else:
        model = e2["best_res"]["model"]
        y_mean, y_std = e2["best_res"]["y_mean"], e2["best_res"]["y_std"]
        Xvan = e2["arrays"]["Xvan"]
        pred_s = predict_np(model, Xvan)
        nn_flat = (pred_s * y_std + y_mean).reshape(-1)

    nn_mae = insp_mae(y_flat, nn_flat, u_flat)
    log(f"E5 NN aligned val MAE={nn_mae:.4f}")

    m = u_flat == 0
    corr = float(np.corrcoef(tree_pred_flat[m] - y_flat[m], nn_flat[m] - y_flat[m])[0, 1])
    log(f"E5 inspiratory error correlation tree vs NN: {corr:.4f}")

    blend_rows = []
    for w in np.linspace(0, 1, 11):
        pred = w * nn_flat + (1 - w) * tree_pred_flat
        mae = insp_mae(y_flat, pred, u_flat)
        blend_rows.append({"method": f"weighted_nn{w:.1f}", "w_nn": float(w), "mae": mae})
    stack = np.stack([tree_pred_flat, nn_flat], axis=0)
    med = np.median(stack, axis=0)
    mean_pred = stack.mean(0)
    blend_rows.append({"method": "mean", "w_nn": 0.5, "mae": insp_mae(y_flat, mean_pred, u_flat)})
    blend_rows.append({"method": "median", "w_nn": None, "mae": insp_mae(y_flat, med, u_flat)})
    for r in sorted(blend_rows, key=lambda d: d["mae"]):
        log(f"  blend {r['method']}: MAE={r['mae']:.4f}")
    best_blend = min(blend_rows, key=lambda d: d["mae"])
    log(f"E5 best blend on holdout: {best_blend}")

    # -------- Grouped K-fold for finalist --------
    # Finalist = best single model among NN variants if blend doesn't clearly win,
    # or blend if it wins by >0.005
    single_best = min(nn_mae, tree_mae)
    use_blend = best_blend["mae"] < single_best - 0.002
    log(f"E5 finalist type: {'blend' if use_blend else 'single_nn_or_tree'}")

    # Always K-fold the best neural architecture (primary finalist) + report blend OOF if useful
    # Reconstruct best NN recipe
    if nn_pack is not None:
        cols = nn_pack["cols"]
        kind = nn_pack["kind"]
        hidden, layers = nn_pack["hidden"], nn_pack["layers"]
        y_scale, loss, lr, hw = nn_pack["y_scale"], nn_pack["loss"], nn_pack["lr"], nn_pack["hw"]
    else:
        cols = SEQ_COLS_BASE
        kind = "bilstm"
        hidden, layers = e2["best"].get("hidden", 256), e2["best"].get("layers", 3)
        y_scale, loss, lr = e2["best"]["y_scale"], e2["best"]["loss"], e2["best"]["lr"]
        hw = e2["best"].get("hard_w", 1.0)

    def make_model(n_feat):
        if kind == "conv":
            return ConvBiLSTM(n_feat, hidden=hidden, layers=layers, dropout=0.1)
        if kind == "attn":
            return AttnBiLSTM(n_feat, hidden=hidden, layers=layers, dropout=0.1)
        return BiLSTMReg(n_feat, hidden=hidden, layers=layers, dropout=0.1)

    log(f"E5 GroupKFold K=5 for NN finalist kind={kind} cols={len(cols)} "
        f"loss={loss} y_scale={y_scale} lr={lr}")
    X_all, y_all, u_all, R_all, C_all, bids = reshape_sequences(feat_all, cols)
    gkf = GroupKFold(n_splits=5)
    oof = np.zeros_like(y_all)
    fold_scores = []
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_all, y_all, groups=bids), 1):
        Xtr, Xva = X_all[tr_idx], X_all[va_idx]
        ytr, yva = y_all[tr_idx], y_all[va_idx]
        utr, uva = u_all[tr_idx], u_all[va_idx]
        Rtr, Rva = R_all[tr_idx], R_all[va_idx]
        Ctr, Cva = C_all[tr_idx], C_all[va_idx]
        mean, std = norm_fit(Xtr)
        Xtrn, Xvan = apply_norm(Xtr, mean, std), apply_norm(Xva, mean, std)
        model = make_model(Xtrn.shape[-1])
        res = train_model(
            model, Xtrn, ytr, utr, Xvan, yva, uva,
            R_tr=Rtr, C_tr=Ctr, R_va=Rva, C_va=Cva,
            epochs=80, lr=lr, loss_kind=loss, schedule="plateau",
            y_scale=y_scale, hard_weight=hw, patience=8,
            label=f"kfold{fold}", log_every=20,
        )
        oof[va_idx] = res["pred_va"]
        hard = res["best_hard"]
        fold_scores.append({"fold": fold, "val": res["best_val"], "hard": hard,
                            "train": res["best_train"], "ep": res["best_epoch"]})
        log(f"E5 fold{fold}: val={res['best_val']:.4f} hard={hard:.4f} train={res['best_train']:.4f}")

    oof_mae = insp_mae(y_all.reshape(-1), oof.reshape(-1), u_all.reshape(-1))
    oof_hard = insp_mae_by_rc(
        y_all.reshape(-1), oof.reshape(-1), u_all.reshape(-1),
        R_all.reshape(-1), C_all.reshape(-1),
    )
    log(f"E5 NN GroupKFold OOF insp MAE={oof_mae:.4f} hard50_10={oof_hard:.4f}")
    log(f"E5 fold mean±std: {np.mean([f['val'] for f in fold_scores]):.4f} "
        f"± {np.std([f['val'] for f in fold_scores]):.4f}")

    # Tree OOF for blend evaluation
    log("E5 tree GroupKFold OOF...")
    tree_oof = np.zeros(len(feat_all), dtype=np.float64)
    feat_sorted = feat_all.sort_values(["breath_id", "time_step"]).reset_index(drop=True)
    groups = feat_sorted["breath_id"].to_numpy()
    # unique breath groups for GroupKFold on rows — use breath-level indices
    breath_ids_u = feat_sorted["breath_id"].to_numpy().reshape(-1, 80)[:, 0]
    tree_oof_b = np.zeros_like(y_all)
    tree_fold = []
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_all, y_all, groups=bids), 1):
        tr_bids = set(bids[tr_idx].tolist())
        va_bids = set(bids[va_idx].tolist())
        tr = feat_sorted[feat_sorted.breath_id.isin(tr_bids)]
        va = feat_sorted[feat_sorted.breath_id.isin(va_bids)]
        tr_insp = tr[tr.u_out == 0]
        dtr = lgb.Dataset(tr_insp[FEATS_TREE], label=tr_insp["pressure"])
        dva = lgb.Dataset(va.loc[va.u_out == 0, FEATS_TREE], label=va.loc[va.u_out == 0, "pressure"])
        bst = lgb.train(
            {"objective": "regression_l1", "learning_rate": 0.08, "num_leaves": 127,
             "verbosity": -1, "seed": SEED, "n_jobs": 4},
            dtr, num_boost_round=600,
            valid_sets=[dva],
            callbacks=[lgb.early_stopping(40, verbose=False)],
        )
        pred = bst.predict(va[FEATS_TREE], num_iteration=bst.best_iteration).reshape(-1, 80)
        # order va breaths as in bids[va_idx]
        # va sorted by breath_id; align to va_idx order
        va_bids_order = va["breath_id"].to_numpy().reshape(-1, 80)[:, 0]
        # map
        pred_map = {b: pred[i] for i, b in enumerate(va_bids_order)}
        aligned = np.stack([pred_map[b] for b in bids[va_idx]])
        tree_oof_b[va_idx] = aligned
        mae = insp_mae(y_all[va_idx].reshape(-1), aligned.reshape(-1), u_all[va_idx].reshape(-1))
        tree_fold.append(mae)
        log(f"E5 tree fold{fold}: val={mae:.4f}")

    tree_oof_mae = insp_mae(y_all.reshape(-1), tree_oof_b.reshape(-1), u_all.reshape(-1))
    log(f"E5 tree GroupKFold OOF MAE={tree_oof_mae:.4f}")

    # OOF blends
    oof_blend = []
    for w in np.linspace(0, 1, 11):
        pred = w * oof + (1 - w) * tree_oof_b
        mae = insp_mae(y_all.reshape(-1), pred.reshape(-1), u_all.reshape(-1))
        oof_blend.append({"w_nn": float(w), "mae": mae})
    mean_oof = insp_mae(y_all.reshape(-1), (0.5 * oof + 0.5 * tree_oof_b).reshape(-1), u_all.reshape(-1))
    med_oof = insp_mae(
        y_all.reshape(-1),
        np.median(np.stack([oof, tree_oof_b]), axis=0).reshape(-1),
        u_all.reshape(-1),
    )
    best_oof_w = min(oof_blend, key=lambda d: d["mae"])
    log(f"E5 OOF mean blend={mean_oof:.4f} median={med_oof:.4f} best_w={best_oof_w}")

    return {
        "holdout": {
            "tree_mae": tree_mae, "nn_mae": nn_mae, "corr": corr,
            "blends": blend_rows, "best_blend": best_blend,
        },
        "kfold_nn": {"oof_mae": oof_mae, "oof_hard": oof_hard, "folds": fold_scores, "name": nn_name},
        "kfold_tree": {"oof_mae": tree_oof_mae, "folds": tree_fold},
        "kfold_blend": {"mean": mean_oof, "median": med_oof, "best_w": best_oof_w},
    }


def main():
    import os
    t0 = time.time()
    log("BATCH 2")
    log(f"device={DEVICE} cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"gpu={torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
        # reduce fragmentation risk on 8GB laptop GPUs
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    log("Loading + feature engineering...")
    train = pd.read_csv(DATA / "train.csv")
    feat = add_features(train)
    log(f"feat shape={feat.shape} sec={time.time()-t0:.1f}")

    # Preserve prior best-known independently
    best_known = {"name": "sanity_bilstm_full_60ep", "val_insp_mae": 0.3747}
    log(f"Best-known (frozen reference): {best_known}")

    resume = os.environ.get("BATCH2_RESUME", "0") == "1"
    if resume and (OUT / "batch2_e1e4.json").exists():
        prior = json.loads((OUT / "batch2_e1e4.json").read_text(encoding="utf-8"))
        e1 = {
            "winner": prior["e1"]["winner"],
            "memorized": prior["e1"]["memorized"],
            "probes": [],
            "from_resume": True,
        }
        e4 = prior["e4"]
        log("Resuming: loaded E1/E4 from batch2_e1e4.json")
        # Still run fixed-LR memorization follow-up (lightweight) before E2
        log("E1 follow-up: fixed-LR memorization probe (schedule=none)...")
        rng = np.random.default_rng(0)
        ids = rng.choice(feat["breath_id"].unique(), size=500, replace=False)
        sub = feat[feat["breath_id"].isin(ids)]
        X, y, u, R, C, _ = reshape_sequences(sub, SEQ_COLS_BASE)
        Xn = apply_norm(X, *norm_fit(X))
        model = BiLSTMReg(Xn.shape[-1], hidden=384, layers=3, dropout=0.0)
        res = train_model(
            model, Xn, y, u, Xn, y, u,
            R_tr=R, C_tr=C, R_va=R, C_va=C,
            epochs=250, lr=1e-2, batch_size=64, loss_kind="mse",
            schedule="none", y_scale=True, patience=80,
            label="fixed_mse_lr1e2", log_every=50,
        )
        log(f"E1 follow-up fixed LR: best_train={res['best_train']:.4f} ep={res['best_epoch']}")
        e1["followup_fixed_lr"] = {"train_mae": res["best_train"], "epoch": res["best_epoch"]}
        if res["best_train"] < e1["winner"]["train_mae"]:
            e1["winner"] = {
                "label": "fixed_mse_lr1e2", "y_scale": True, "loss": "mse", "lr": 0.01,
                "train_mae": res["best_train"], "best_epoch": res["best_epoch"],
            }
        del model, res
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        e4 = e4_hard_cases(train, feat)
        e1 = e1_memorization(feat)

    e2 = e2_push_sequence(feat, e1["winner"])
    e3 = e3_architecture(feat, e2)
    e5 = e5_ensemble_and_kfold(feat, e2, e3)

    # Update best-known if beaten on comparable single split AND confirm with kfold
    screen_best = min(e2["best"]["best_val"], e3["best"]["row"]["best_val"])
    kfold_nn = e5["kfold_nn"]["oof_mae"]
    kfold_blend = min(e5["kfold_blend"]["mean"], e5["kfold_blend"]["median"],
                      e5["kfold_blend"]["best_w"]["mae"])
    finalist_mae = min(kfold_nn, kfold_blend, e5["kfold_tree"]["oof_mae"])
    log()
    log("=" * 72)
    log("BATCH 2 SUMMARY")
    log("=" * 72)
    log(f"Frozen prior best-known: {best_known['val_insp_mae']:.4f}")
    log(f"E2 best screen val: {e2['best']['best_val']:.4f} ({e2['best']['label']})")
    log(f"E3 best screen val: {e3['best']['row']['best_val']:.4f} ({e3['best']['row']['label']})")
    log(f"E5 NN GroupKFold OOF: {kfold_nn:.4f}")
    log(f"E5 tree GroupKFold OOF: {e5['kfold_tree']['oof_mae']:.4f}")
    log(f"E5 best OOF blend: {e5['kfold_blend']['best_w']}")
    log(f"Finalist OOF MAE (min NN/tree/blend): {finalist_mae:.4f}")
    if finalist_mae < best_known["val_insp_mae"]:
        log("NEW best-known (by GroupKFold OOF / comparable protocol).")
    else:
        log("Did not claim new best-known over frozen reference without clear OOF win "
            "(single-split screens may still be informative).")

    summary = {
        "best_known_start": best_known,
        "e1": {k: v for k, v in e1.items()},
        "e2": {"results": e2["results"], "best": e2["best"]},
        "e3": {"results": e3["results"], "best": e3["best"]["row"]},
        "e4": e4,
        "e5": e5,
        "finalist_oof_mae": finalist_mae,
        "elapsed_sec": time.time() - t0,
    }
    # Drop non-serializable
    (OUT / "batch2_report.txt").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "batch2_results.json").write_text(
        json.dumps(summary, indent=2, default=float), encoding="utf-8"
    )
    log(f"Wrote {OUT / 'batch2_report.txt'}")
    log(f"Elapsed sec={time.time()-t0:.1f}")


if __name__ == "__main__":
    main()
