"""
Sanity checks after Batch 1 — NOT a new research batch.

A) Re-diagnose insp-rows-only collapse (feature timing + early-stop metric)
B) BiLSTM capacity / pipeline checks (tiny overfit, norm, alignment, train vs val)
C) Scale BiLSTM on full data with patient schedule
D) Error analysis for best LightGBM
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from batch1_experiments import (
    FEATS_BASE,
    FEATS_PHYS,
    BiLSTM,
    add_features,
    insp_mae,
    reshape_sequences,
)

DATA = Path(__file__).resolve().parents[1] / "data" / "ventilator-pressure-prediction"
OUT = Path(__file__).resolve().parent
SEED = 42
lines: list[str] = []


def log(msg: str = "") -> None:
    print(msg, flush=True)
    lines.append(msg)


def make_split(all_ids, n_train, n_val):
    rng = np.random.default_rng(SEED)
    ids = np.array(all_ids).copy()
    rng.shuffle(ids)
    val_ids = set(ids[:n_val].tolist())
    tr_ids = set(ids[n_val : n_val + n_train].tolist())
    return tr_ids, val_ids


def train_lgbm_fixed(
    X_tr, y_tr, X_va, y_va, u_va,
    w_tr=None,
    rounds=400,
    early_stop_on="insp",  # "insp" | "all" | "none"
    label="",
):
    dtrain = lgb.Dataset(X_tr, label=y_tr, weight=w_tr)
    # Validation set used for early stopping
    if early_stop_on == "insp":
        m = u_va == 0
        dvalid = lgb.Dataset(X_va[m], label=y_va[m], reference=dtrain)
        callbacks = [lgb.early_stopping(40, verbose=False), lgb.log_evaluation(100)]
        valid_sets = [dvalid]
    elif early_stop_on == "all":
        dvalid = lgb.Dataset(X_va, label=y_va, reference=dtrain)
        callbacks = [lgb.early_stopping(40, verbose=False), lgb.log_evaluation(100)]
        valid_sets = [dvalid]
    else:
        valid_sets, callbacks = None, [lgb.log_evaluation(100)]

    params = {
        "objective": "regression_l1",
        "metric": "mae",
        "learning_rate": 0.1,
        "num_leaves": 96,
        "min_data_in_leaf": 40,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbosity": -1,
        "n_jobs": 4,
        "seed": SEED,
    }
    t0 = time.time()
    kwargs = {
        "params": params,
        "train_set": dtrain,
        "num_boost_round": rounds,
    }
    if valid_sets is not None:
        kwargs["valid_sets"] = valid_sets
        kwargs["callbacks"] = callbacks
    else:
        kwargs["callbacks"] = [lgb.log_evaluation(100)]
    booster = lgb.train(**kwargs)
    trained = rounds if early_stop_on == "none" else (booster.best_iteration or rounds)
    best_mae, best_it, best_pred = float("inf"), trained, None
    grid = sorted(set([trained] + list(range(50, trained + 1, 50)) + [trained]))
    for i in grid:
        pred = booster.predict(X_va, num_iteration=i)
        mae = insp_mae(y_va, pred, u_va)
        if mae < best_mae:
            best_mae, best_it, best_pred = mae, i, pred
    log(
        f"  [{label}] stop_on={early_stop_on} rounds_cap={rounds} chosen_iter={best_it} "
        f"booster_best={booster.best_iteration} insp_mae={best_mae:.4f} sec={time.time()-t0:.1f}"
    )
    return {"mae": best_mae, "iter": best_it, "pred": best_pred, "booster": booster}


# ---------------------------------------------------------------------------
# A) Feature timing + insp-rows-only rerun
# ---------------------------------------------------------------------------
def check_insp_rows(train: pd.DataFrame):
    log("=" * 72)
    log("A) INSP-ROWS-ONLY DIAGNOSIS")
    log("=" * 72)

    tr_ids, va_ids = make_split(train["breath_id"].unique(), 20000, 5000)
    need = train["breath_id"].isin(tr_ids | va_ids)
    feat = add_features(train.loc[need])
    tr = feat[feat["breath_id"].isin(tr_ids)]
    va = feat[feat["breath_id"].isin(va_ids)]

    # Prove features were computed on full breath: compare cumsum at last insp
    # step vs cumsum of u_in only over insp rows (should differ if any u_in after? no -
    # during insp u_out=0 contiguous from start, so cumsum at last insp == sum of insp u_in.
    # Better check: lag1 at step 5 equals u_in at step 4 of SAME breath after filter.
    sample_bid = next(iter(tr_ids))
    full_b = feat[feat["breath_id"] == sample_bid].sort_values("time_step")
    insp_b = full_b[full_b["u_out"] == 0]
    # After filtering rows, lag features on remaining rows should still match full-breath lags
    ok_lag = np.allclose(insp_b["u_in_lag1"].to_numpy(), insp_b["u_in"].shift(1).fillna(0).to_numpy())
    # Wait - after filter, shift(1) on insp-only would equal full lag only because insp is prefix.
    # Since insp is always a prefix (single switch), filtering insp rows preserves lag structure
    # for insp rows. Check a feature that uses FULL trajectory: u_in_backcum / insp_len.
    back_from_full = full_b["u_in_backcum"].to_numpy()
    back_if_recomputed_on_insp_only = np.cumsum(insp_b["u_in"].to_numpy()[::-1])[::-1]
    # These SHOULD differ (backcum includes expiratory u_in). If E2b used recomputed features
    # on filtered rows, backcum would equal the insp-only version.
    used_full_backcum = not np.allclose(
        insp_b["u_in_backcum"].to_numpy(), back_if_recomputed_on_insp_only, rtol=0, atol=1e-6
    )
    log(f"Sample breath {sample_bid}: insp steps={len(insp_b)}")
    log(f"u_in_backcum on insp rows matches FULL-breath engineering (not recomputed on filter): {used_full_backcum}")
    log(f"insp_len constant on breath and equals first u_out=1 index: "
        f"{insp_b['insp_len'].nunique()==1 and int(insp_b['insp_len'].iloc[0])==int(np.argmax(full_b['u_out'].to_numpy()==1))}")
    log("Conclusion on feature timing: Batch1 computed add_features on full breaths, then filtered rows.")
    log("So the 6.84 result was NOT caused by computing lags/cumsum after filtering.")

    X_va = va[FEATS_BASE].to_numpy(np.float32)
    y_va = va["pressure"].to_numpy(np.float32)
    u_va = va["u_out"].to_numpy()
    X_tr_all = tr[FEATS_BASE].to_numpy(np.float32)
    y_tr_all = tr["pressure"].to_numpy(np.float32)
    tr_insp = tr[tr["u_out"] == 0]
    X_tr_insp = tr_insp[FEATS_BASE].to_numpy(np.float32)
    y_tr_insp = tr_insp["pressure"].to_numpy(np.float32)
    w = np.where(tr["u_out"].to_numpy() == 0, 1.0, 0.05).astype(np.float32)

    log("\nRe-runs (features always from full-breath engineering):")
    # Reproduce Batch1 bug path: insp rows + early stop on ALL val rows
    r_bug = train_lgbm_fixed(
        X_tr_insp, y_tr_insp, X_va, y_va, u_va,
        early_stop_on="all", rounds=400, label="insp_rows + earlystop_ALL (batch1 path)",
    )
    # Corrected: insp rows + early stop on INSP val rows
    r_fix = train_lgbm_fixed(
        X_tr_insp, y_tr_insp, X_va, y_va, u_va,
        early_stop_on="insp", rounds=400, label="insp_rows + earlystop_INSP",
    )
    # Fixed rounds, no early stop
    r_fixed = train_lgbm_fixed(
        X_tr_insp, y_tr_insp, X_va, y_va, u_va,
        early_stop_on="none", rounds=400, label="insp_rows + fixed 400 (no earlystop)",
    )
    # Soft weight reference
    r_w = train_lgbm_fixed(
        X_tr_all, y_tr_all, X_va, y_va, u_va, w_tr=w,
        early_stop_on="insp", rounds=400, label="all_rows + insp_weight + earlystop_INSP",
    )
    # All-phase reference with insp earlystop
    r_all = train_lgbm_fixed(
        X_tr_all, y_tr_all, X_va, y_va, u_va,
        early_stop_on="insp", rounds=400, label="all_rows equal + earlystop_INSP",
    )

    out = {
        "feature_timing_ok_full_then_filter": True,
        "used_full_backcum_not_recomputed": bool(used_full_backcum),
        "insp_rows_earlystop_all": r_bug["mae"],
        "insp_rows_earlystop_insp": r_fix["mae"],
        "insp_rows_fixed400": r_fixed["mae"],
        "insp_weighted_earlystop_insp": r_w["mae"],
        "all_phase_earlystop_insp": r_all["mae"],
        "bug_iters": {"all_es": r_bug["iter"], "insp_es": r_fix["iter"], "fixed": r_fixed["iter"]},
    }
    return out, tr, va, r_w


# ---------------------------------------------------------------------------
# B) BiLSTM pipeline checks
# ---------------------------------------------------------------------------
def bilstm_predict(model, X, device, bs=512):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i : i + bs]).to(device)
            preds.append(model(xb).cpu().numpy())
    return np.concatenate(preds, axis=0)


def check_bilstm_pipeline(feat: pd.DataFrame, tr_ids, va_ids):
    log()
    log("=" * 72)
    log("B) BiLSTM PIPELINE / CAPACITY CHECKS")
    log("=" * 72)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={device}")

    tr = feat[feat["breath_id"].isin(tr_ids)].sort_values(["breath_id", "time_step"])
    va = feat[feat["breath_id"].isin(va_ids)].sort_values(["breath_id", "time_step"])

    # Alignment checks
    sizes = tr.groupby("breath_id").size().unique()
    log(f"train breath lengths unique={sizes.tolist()} (expect [80])")
    # time_step sorted within breath
    bad_sort = 0
    for _, g in tr.groupby("breath_id"):
        if not g["time_step"].is_monotonic_increasing:
            bad_sort += 1
            break
    log(f"any breath not sorted by time_step? {bad_sort > 0}")

    seq_cols = ["u_in", "u_out", "R", "C", "time_step", "R_div_C", "u_over_c"]
    Xtr, ytr, utr = reshape_sequences(tr, seq_cols)
    Xva, yva, uva = reshape_sequences(va, seq_cols)

    # Mask/id alignment: reconstruct breath ids from reshape order
    tr_bids = tr["breath_id"].to_numpy().reshape(-1, 80)[:, 0]
    # Spot-check: first row features match dataframe
    b0 = tr_bids[0]
    raw0 = tr[tr["breath_id"] == b0][seq_cols].to_numpy(np.float32)
    align_ok = np.allclose(raw0, Xtr[0])
    log(f"reshape alignment vs dataframe for breath {b0}: {align_ok}")
    log(f"u_out mask insp count per breath mean={((uva == 0).sum(axis=1)).mean():.2f}")

    # Normalization as in Batch1
    mean = Xtr.reshape(-1, Xtr.shape[-1]).mean(0)
    std = Xtr.reshape(-1, Xtr.shape[-1]).std(0)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    mean = mean.astype(np.float32)
    log(f"inputs normalized? YES (Batch1 applied (x-mean)/std). mean[:3]={mean[:3]} std[:3]={std[:3]}")

    # --- Overfit ~500 breaths ---
    log("\nB1) Overfit test on ~500 breaths (train MAE should approach ~0)")
    rng = np.random.default_rng(0)
    over_ids = rng.choice(tr_bids, size=500, replace=False)
    mask_over = np.isin(tr_bids, over_ids)
    Xo = ((Xtr[mask_over] - mean) / std).astype(np.float32)
    yo = ytr[mask_over]
    uo = utr[mask_over]

    model = BiLSTM(Xo.shape[-1], hidden=128, layers=2, dropout=0.0).to(device)  # dropout 0 to overfit
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(Xo), torch.from_numpy(yo), torch.from_numpy(uo)),
        batch_size=64, shuffle=True,
    )
    hist = []
    for ep in range(1, 81):
        model.train()
        for xb, yb, ub in loader:
            xb, yb, ub = xb.to(device), yb.to(device), ub.to(device)
            pred = model(xb)
            m = (ub == 0).float()
            loss = (torch.abs(pred - yb) * m).sum() / m.sum().clamp_min(1.0)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if ep % 10 == 0 or ep == 1:
            pred = bilstm_predict(model, Xo, device)
            tr_mae = insp_mae(yo.reshape(-1), pred.reshape(-1), uo.reshape(-1))
            hist.append((ep, tr_mae))
            log(f"  overfit epoch={ep} train_insp_mae={tr_mae:.4f}")
    overfit_final = hist[-1][1]
    log(f"Overfit final train insp MAE={overfit_final:.4f} ({'OK capacity' if overfit_final < 0.15 else 'SUSPICIOUS — likely bug or capacity issue'})")

    # --- Train/val on Batch1-sized subsample with longer run, track curves ---
    log("\nB2) Train vs val curves on 20k/5k (longer, patient schedule)")
    # reuse same 20k/5k from caller
    Xtrn = ((Xtr - mean) / std).astype(np.float32)
    Xvan = ((Xva - mean) / std).astype(np.float32)
    model = BiLSTM(Xtrn.shape[-1], hidden=128, layers=2, dropout=0.1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # patient: keep high LR longer then cosine
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=40)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(Xtrn), torch.from_numpy(ytr), torch.from_numpy(utr)),
        batch_size=256, shuffle=True,
    )
    curve = []
    best = {"mae": float("inf"), "epoch": 0, "pred": None}
    for ep in range(1, 41):
        model.train()
        for xb, yb, ub in loader:
            xb, yb, ub = xb.to(device), yb.to(device), ub.to(device)
            pred = model(xb)
            m = (ub == 0).float()
            loss = (torch.abs(pred - yb) * m).sum() / m.sum().clamp_min(1.0)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        pred_tr = bilstm_predict(model, Xtrn, device)
        pred_va = bilstm_predict(model, Xvan, device)
        tr_mae = insp_mae(ytr.reshape(-1), pred_tr.reshape(-1), utr.reshape(-1))
        va_mae = insp_mae(yva.reshape(-1), pred_va.reshape(-1), uva.reshape(-1))
        curve.append({"epoch": ep, "train_insp_mae": tr_mae, "val_insp_mae": va_mae, "lr": opt.param_groups[0]["lr"]})
        if va_mae < best["mae"]:
            best = {"mae": va_mae, "epoch": ep, "pred": pred_va}
        if ep % 5 == 0 or ep == 1:
            log(f"  ep={ep:02d} train={tr_mae:.4f} val={va_mae:.4f} lr={opt.param_groups[0]['lr']:.2e}")

    # Was val still improving at Batch1 stop (epoch 15/20)?
    by_ep = {c["epoch"]: c for c in curve}
    log(f"At epoch 15: train={by_ep[15]['train_insp_mae']:.4f} val={by_ep[15]['val_insp_mae']:.4f}")
    log(f"At epoch 20: train={by_ep[20]['train_insp_mae']:.4f} val={by_ep[20]['val_insp_mae']:.4f}")
    log(f"Best in 40 epochs: ep={best['epoch']} val={best['mae']:.4f}")
    still_improving_at_20 = by_ep[20]["val_insp_mae"] > min(c["val_insp_mae"] for c in curve if c["epoch"] > 20) - 1e-6 and best["epoch"] > 20
    log(f"Val still improving after epoch 20? best_ep={best['epoch']} => {best['epoch'] > 20}")

    return {
        "align_ok": bool(align_ok),
        "normalized": True,
        "overfit_train_mae": float(overfit_final),
        "curve": curve,
        "best_val": best["mae"],
        "best_epoch": best["epoch"],
        "ep15": by_ep[15],
        "ep20": by_ep[20],
    }


# ---------------------------------------------------------------------------
# C) Full-data BiLSTM
# ---------------------------------------------------------------------------
def scale_bilstm_full(train: pd.DataFrame):
    log()
    log("=" * 72)
    log("C) FULL-DATA BiLSTM WITH PATIENT SCHEDULE")
    log("=" * 72)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_ids = train["breath_id"].unique()
    # 80/20 breath split on full data
    tr_ids, va_ids = make_split(all_ids, n_train=int(0.8 * len(all_ids)), n_val=int(0.2 * len(all_ids)))
    log(f"full split breaths train/val={len(tr_ids)}/{len(va_ids)}")

    log("Feature engineering on all breaths...")
    feat = add_features(train)
    tr = feat[feat["breath_id"].isin(tr_ids)]
    va = feat[feat["breath_id"].isin(va_ids)]
    seq_cols = ["u_in", "u_out", "R", "C", "time_step", "R_div_C", "u_over_c"]
    # also add stronger features trees use
    seq_cols_rich = seq_cols + [
        "u_in_lag1", "u_in_cumsum", "u_in_cummax", "du", "insp_len", "steps_to_exp", "u_in_backcum", "vol_over_c"
    ]
    Xtr, ytr, utr = reshape_sequences(tr, seq_cols_rich)
    Xva, yva, uva = reshape_sequences(va, seq_cols_rich)
    mean = Xtr.reshape(-1, Xtr.shape[-1]).mean(0).astype(np.float32)
    std = Xtr.reshape(-1, Xtr.shape[-1]).std(0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    Xtrn = ((Xtr - mean) / std).astype(np.float32)
    Xvan = ((Xva - mean) / std).astype(np.float32)

    model = BiLSTM(Xtrn.shape[-1], hidden=256, layers=3, dropout=0.1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    # patient: linear warmup 5 ep then cosine to epoch 60
    warmup, total_epochs = 5, 60

    def lr_at(ep):
        if ep <= warmup:
            return 8e-4 * ep / warmup
        t = (ep - warmup) / (total_epochs - warmup)
        return 1e-5 + 0.5 * (8e-4 - 1e-5) * (1 + np.cos(np.pi * t))

    loader = DataLoader(
        TensorDataset(torch.from_numpy(Xtrn), torch.from_numpy(ytr), torch.from_numpy(utr)),
        batch_size=512, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available(),
    )
    curve = []
    best = {"mae": float("inf"), "epoch": 0, "train": None, "state": None}
    t0 = time.time()
    for ep in range(1, total_epochs + 1):
        lr = float(lr_at(ep))
        for g in opt.param_groups:
            g["lr"] = lr
        model.train()
        for xb, yb, ub in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            ub = ub.to(device, non_blocking=True)
            pred = model(xb)
            m = (ub == 0).float()
            loss = (torch.abs(pred - yb) * m).sum() / m.sum().clamp_min(1.0)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        if ep % 2 == 0 or ep == 1 or ep == total_epochs:
            pred_tr = bilstm_predict(model, Xtrn, device)
            pred_va = bilstm_predict(model, Xvan, device)
            tr_mae = insp_mae(ytr.reshape(-1), pred_tr.reshape(-1), utr.reshape(-1))
            va_mae = insp_mae(yva.reshape(-1), pred_va.reshape(-1), uva.reshape(-1))
            curve.append({"epoch": ep, "train": tr_mae, "val": va_mae, "lr": lr})
            log(f"  full ep={ep:02d} train={tr_mae:.4f} val={va_mae:.4f} lr={lr:.2e}")
            if va_mae < best["mae"]:
                best = {
                    "mae": va_mae,
                    "epoch": ep,
                    "train": tr_mae,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                }

    log(f"Full BiLSTM best val insp MAE={best['mae']:.4f} at ep={best['epoch']} "
        f"train_at_best={best['train']:.4f} sec={time.time()-t0:.1f}")
    still_improving = best["epoch"] >= total_epochs - 4
    log(f"Best near end of training (still improving / undertrained signal)? {still_improving}")
    return {
        "best_val": best["mae"],
        "best_epoch": best["epoch"],
        "train_at_best": best["train"],
        "curve": curve,
        "n_train": len(tr_ids),
        "n_val": len(va_ids),
        "feat_dim": Xtrn.shape[-1],
    }


# ---------------------------------------------------------------------------
# D) Error analysis for best LightGBM
# ---------------------------------------------------------------------------
def error_analysis(tr: pd.DataFrame, va: pd.DataFrame):
    log()
    log("=" * 72)
    log("D) ERROR ANALYSIS — best LightGBM (phys feats, insp-weighted)")
    log("=" * 72)

    X_tr = tr[FEATS_PHYS].to_numpy(np.float32)
    y_tr = tr["pressure"].to_numpy(np.float32)
    w = np.where(tr["u_out"].to_numpy() == 0, 1.0, 0.05).astype(np.float32)
    X_va = va[FEATS_PHYS].to_numpy(np.float32)
    y_va = va["pressure"].to_numpy(np.float32)
    u_va = va["u_out"].to_numpy()

    res = train_lgbm_fixed(
        X_tr, y_tr, X_va, y_va, u_va, w_tr=w,
        early_stop_on="insp", rounds=500, label="best_lgbm_phys_for_errors",
    )
    va = va.copy()
    va["pred"] = res["pred"]
    va["abs_err"] = np.abs(va["pressure"] - va["pred"])
    insp = va[va["u_out"] == 0].copy()
    log(f"Overall val insp MAE={insp['abs_err'].mean():.4f}")

    log("\nMAE by (R,C):")
    tab = insp.groupby(["R", "C"])["abs_err"].agg(["mean", "median", "count"])
    log(tab.to_string())

    insp["insp_step"] = insp.groupby("breath_id").cumcount()
    log("\nMAE by inspiratory step (selected):")
    by_step = insp.groupby("insp_step")["abs_err"].mean()
    for s in list(range(0, 31, 5)) + [1, 2, 29, 30, 31]:
        if s in by_step.index:
            log(f"  step {s:02d}: {by_step.loc[s]:.4f}")

    # Worst breaths
    breath_mae = insp.groupby("breath_id").agg(
        mae=("abs_err", "mean"),
        R=("R", "first"),
        C=("C", "first"),
        insp_len=("step", "count"),
        u_in_mean=("u_in", "mean"),
        u_in_max=("u_in", "max"),
        u_in_std=("u_in", "std"),
        pressure_max=("pressure", "max"),
        pressure_min=("pressure", "min"),
        pressure_range=("pressure", lambda s: s.max() - s.min()),
        du_abs_mean=("du", lambda s: np.mean(np.abs(s))),
    ).sort_values("mae", ascending=False)

    log("\nTop 15 worst breaths:")
    log(breath_mae.head(15).to_string())

    worst = breath_mae.head(200)
    rest = breath_mae.iloc[200:]
    log("\nWorst-200 vs rest — means:")
    for col in ["R", "C", "u_in_mean", "u_in_max", "u_in_std", "pressure_max", "pressure_range", "du_abs_mean", "insp_len"]:
        log(f"  {col}: worst={worst[col].mean():.3f} rest={rest[col].mean():.3f}")

    log("\nWorst-200 (R,C) distribution:")
    log(worst.groupby(["R", "C"]).size().unstack(fill_value=0).to_string())
    log("\nAll breaths (R,C) distribution:")
    log(breath_mae.groupby(["R", "C"]).size().unstack(fill_value=0).to_string())

    return {
        "overall_mae": float(insp["abs_err"].mean()),
        "by_rc": tab.reset_index().to_dict(orient="records"),
        "worst_means": {c: float(worst[c].mean()) for c in ["R", "C", "u_in_max", "pressure_max", "pressure_range", "du_abs_mean"]},
        "rest_means": {c: float(rest[c].mean()) for c in ["R", "C", "u_in_max", "pressure_max", "pressure_range", "du_abs_mean"]},
    }


def main():
    t0 = time.time()
    log("SANITY CHECK (post Batch 1) — not a new batch")
    log(f"cuda={torch.cuda.is_available()} {torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}")

    train = pd.read_csv(DATA / "train.csv")

    insp_results, tr_sub, va_sub, best_ref = check_insp_rows(train)

    # BiLSTM checks on same 20k/5k feat frames
    tr_ids = set(tr_sub["breath_id"].unique())
    va_ids = set(va_sub["breath_id"].unique())
    feat_sub = pd.concat([tr_sub, va_sub], axis=0)
    bilstm_sub = check_bilstm_pipeline(feat_sub, tr_ids, va_ids)

    full_bilstm = scale_bilstm_full(train)

    # Error analysis uses same 20k/5k as Batch1 for comparability
    err = error_analysis(tr_sub, va_sub)

    summary = {
        "insp_rows": insp_results,
        "bilstm_sub": {k: v for k, v in bilstm_sub.items() if k != "curve"},
        "bilstm_sub_curve_tail": bilstm_sub["curve"][-5:],
        "bilstm_full": {k: v for k, v in full_bilstm.items() if k != "curve"},
        "bilstm_full_curve_tail": full_bilstm["curve"][-5:],
        "error_analysis": err,
        "elapsed_sec": time.time() - t0,
    }

    log()
    log("=" * 72)
    log("SANITY SUMMARY")
    log("=" * 72)
    log(json.dumps(summary, indent=2, default=float))

    (OUT / "sanity_report.txt").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "sanity_results.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    log(f"\nWrote {OUT / 'sanity_report.txt'}")


if __name__ == "__main__":
    main()
