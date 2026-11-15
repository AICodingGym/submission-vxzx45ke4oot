"""
Batch 1 — hypothesis-driven experiments for ventilator pressure prediction.

H1: Nonlinear trees + history features beat open-loop ARX (~2.91).
H2: Inspiratory-only training still beats all-phase training for trees.
H3: BiLSTM (GPU) beats handcrafted-feature trees by learning dynamics.
H4: Snapping continuous preds to the pressure lattice improves insp MAE.
H5: Physics-inspired features improve trees beyond lag/cumsum features.

Metric: inspiratory MAE (u_out==0).
Split: breath-level holdout. For speed/ranking we use a breath subsample
for trees; BiLSTM uses the same subsample so comparisons are aligned.
A final note records subsample sizes; ARX baseline re-run on same split.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

DATA = Path(__file__).resolve().parents[1] / "data" / "ventilator-pressure-prediction"
OUT = Path(__file__).resolve().parent
SEED = 42
PRESSURE_STEP = 0.07030214544382634
# Subsample for Batch-1 ranking (still thousands of breaths)
N_TRAIN_BREATHS = 20000
N_VAL_BREATHS = 5000


def log(msg: str, lines: list[str]) -> None:
    print(msg, flush=True)
    lines.append(msg)


def insp_mae(y_true: np.ndarray, y_pred: np.ndarray, u_out: np.ndarray) -> float:
    m = u_out == 0
    return float(np.mean(np.abs(y_true[m] - y_pred[m])))


def snap_to_lattice(pred: np.ndarray, unique_pressures: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(unique_pressures, pred)
    idx = np.clip(idx, 1, len(unique_pressures) - 1)
    left = unique_pressures[idx - 1]
    right = unique_pressures[idx]
    return np.where(np.abs(pred - left) <= np.abs(pred - right), left, right)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["breath_id", "time_step"]).copy()
    g = df.groupby("breath_id", sort=False)
    df["step"] = g.cumcount().astype(np.int16)
    df["u_in_lag1"] = g["u_in"].shift(1).fillna(0.0)
    df["u_in_lag2"] = g["u_in"].shift(2).fillna(0.0)
    df["u_in_lag3"] = g["u_in"].shift(3).fillna(0.0)
    df["du"] = df["u_in"] - df["u_in_lag1"]
    df["du2"] = df["u_in_lag1"] - df["u_in_lag2"]
    df["u_in_cumsum"] = g["u_in"].cumsum()
    df["u_in_cummax"] = g["u_in"].cummax()
    df["u_in_cummean"] = df["u_in_cumsum"] / (df["step"] + 1)
    df["u_in_roll3"] = (df["u_in"] + df["u_in_lag1"] + df["u_in_lag2"]) / 3.0
    df["u_out_lag1"] = g["u_out"].shift(1).fillna(0.0)
    u_out = df["u_out"].to_numpy().reshape(-1, 80)
    switch = np.argmax(u_out == 1, axis=1).astype(np.int16)
    df["insp_len"] = np.repeat(switch, 80)
    df["steps_to_exp"] = df["insp_len"] - df["step"]
    u_in = df["u_in"].to_numpy().reshape(-1, 80)
    df["u_in_backcum"] = np.cumsum(u_in[:, ::-1], axis=1)[:, ::-1].reshape(-1)
    df["u_over_c"] = df["u_in"] / df["C"]
    df["u_r_over_c"] = df["u_in"] * df["R"] / df["C"]
    df["vol_over_c"] = df["u_in_cumsum"] / df["C"]
    df["vol_r_over_c"] = df["u_in_cumsum"] * df["R"] / df["C"]
    df["R_C"] = df["R"] * df["C"]
    df["R_div_C"] = df["R"] / df["C"]
    return df


FEATS_BASE = [
    "R", "C", "u_in", "u_out", "time_step", "step",
    "u_in_lag1", "u_in_lag2", "u_in_lag3", "du", "du2",
    "u_in_cumsum", "u_in_cummax", "u_in_cummean", "u_in_roll3",
    "u_out_lag1", "insp_len", "steps_to_exp", "u_in_backcum",
    "R_C", "R_div_C",
]
FEATS_PHYS = FEATS_BASE + ["u_over_c", "u_r_over_c", "vol_over_c", "vol_r_over_c"]


def make_split(all_breath_ids: np.ndarray):
    rng = np.random.default_rng(SEED)
    ids = all_breath_ids.copy()
    rng.shuffle(ids)
    val_ids = ids[:N_VAL_BREATHS]
    train_pool = ids[N_VAL_BREATHS:]
    tr_ids = train_pool[:N_TRAIN_BREATHS]
    return set(tr_ids.tolist()), set(val_ids.tolist())


def run_lgbm(X_tr, y_tr, X_va, y_va, u_out_va, w_tr=None, label=""):
    import lightgbm as lgb

    dtrain = lgb.Dataset(X_tr, label=y_tr, weight=w_tr)
    dvalid = lgb.Dataset(X_va, label=y_va, reference=dtrain)
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
    # Early-stop on overall MAE for speed; report inspiratory MAE.
    t_fit = time.time()
    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=400,
        valid_sets=[dvalid],
        valid_names=["valid"],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(100)],
    )
    # Choose best iteration by inspiratory MAE among a short checkpoint grid
    best_mae, best_iter, best_pred = float("inf"), booster.best_iteration, None
    candidates = sorted(set(
        [max(1, booster.best_iteration)]
        + [i for i in range(50, (booster.best_iteration or 400) + 1, 50)]
        + [booster.best_iteration or 400]
    ))
    for it in candidates:
        pred = booster.predict(X_va, num_iteration=it)
        mae = insp_mae(y_va, pred, u_out_va)
        if mae < best_mae:
            best_mae, best_iter, best_pred = mae, it, pred
    print(
        f"  [{label}] fit_sec={time.time()-t_fit:.1f} mae={best_mae:.4f} iter={best_iter}",
        flush=True,
    )
    return {
        "mae": float(best_mae),
        "best_iter": int(best_iter),
        "device": "cpu",
        "pred": best_pred,
        "label": label,
    }


def arx_baseline(tr: pd.DataFrame, va: pd.DataFrame) -> float:
    from numpy.linalg import lstsq

    init_p = (
        tr.groupby(["R", "C", "breath_id"]).first().reset_index()
        .groupby(["R", "C"])["pressure"].mean()
    )
    coefs = {}
    for (r, c), gtr in tr.groupby(["R", "C"]):
        rows, ys = [], []
        for _, gb in gtr.groupby("breath_id"):
            gb = gb.sort_values("time_step")
            p = gb["pressure"].to_numpy()
            u = gb["u_in"].to_numpy()
            uo = gb["u_out"].to_numpy()
            for i in range(1, len(gb)):
                if uo[i] != 0:
                    continue
                rows.append([p[i - 1], u[i], u[i - 1], u[i] - u[i - 1], uo[i], 1.0])
                ys.append(p[i])
        coefs[(r, c)] = lstsq(np.asarray(rows, float), np.asarray(ys, float), rcond=None)[0]

    va = va.sort_values(["breath_id", "time_step"]).copy()
    preds = np.zeros(len(va))
    va["_i"] = np.arange(len(va))
    for (r, c), gva in va.groupby(["R", "C"]):
        coef = coefs[(r, c)]
        p0 = float(init_p.loc[(r, c)])
        for _, gb in gva.groupby("breath_id"):
            idx = gb["_i"].to_numpy()
            u = gb["u_in"].to_numpy()
            uo = gb["u_out"].to_numpy()
            p_hat = np.zeros(len(gb))
            p_hat[0] = p0
            for i in range(1, len(gb)):
                x = np.array([p_hat[i - 1], u[i], u[i - 1], u[i] - u[i - 1], uo[i], 1.0])
                p_hat[i] = x @ coef
            preds[idx] = p_hat
    return insp_mae(va["pressure"].to_numpy(), preds, va["u_out"].to_numpy())


class BiLSTM(nn.Module):
    def __init__(self, n_feat: int, hidden: int = 128, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            n_feat, hidden, num_layers=layers, batch_first=True,
            bidirectional=True, dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        h, _ = self.lstm(x)
        return self.head(h).squeeze(-1)


def reshape_sequences(df: pd.DataFrame, feature_cols: list[str]):
    df = df.sort_values(["breath_id", "time_step"])
    X = df[feature_cols].to_numpy(dtype=np.float32).reshape(-1, 80, len(feature_cols))
    y = df["pressure"].to_numpy(dtype=np.float32).reshape(-1, 80)
    u_out = df["u_out"].to_numpy(dtype=np.float32).reshape(-1, 80)
    return X, y, u_out


def normalize_fit(X_tr: np.ndarray):
    mean = X_tr.reshape(-1, X_tr.shape[-1]).mean(axis=0)
    std = X_tr.reshape(-1, X_tr.shape[-1]).std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def run_bilstm(X_tr, y_tr, u_tr, X_va, y_va, u_va, mask_train_insp: bool, epochs: int = 20, label=""):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean, std = normalize_fit(X_tr)
    X_tr_n = (X_tr - mean) / std
    X_va_n = (X_va - mean) / std

    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_tr_n),
            torch.from_numpy(y_tr),
            torch.from_numpy(u_tr),
        ),
        batch_size=256,
        shuffle=True,
    )
    model = BiLSTM(X_tr.shape[-1], hidden=128, layers=2, dropout=0.1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_mae, best_state, best_pred = float("inf"), None, None
    t_fit = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb, ub in loader:
            xb, yb, ub = xb.to(device), yb.to(device), ub.to(device)
            pred = model(xb)
            err = torch.abs(pred - yb)
            if mask_train_insp:
                m = (ub == 0).float()
                loss = (err * m).sum() / m.sum().clamp_min(1.0)
            else:
                loss = err.mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            chunks = []
            for i in range(0, len(X_va_n), 512):
                xb = torch.from_numpy(X_va_n[i : i + 512]).to(device)
                chunks.append(model(xb).cpu().numpy())
            pred_va = np.concatenate(chunks, axis=0)
            mae = insp_mae(y_va.reshape(-1), pred_va.reshape(-1), u_va.reshape(-1))
        print(f"  [{label}] epoch={ep} val_insp_mae={mae:.4f}", flush=True)
        if mae < best_mae:
            best_mae = mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_pred = pred_va
    print(f"  [{label}] fit_sec={time.time()-t_fit:.1f} best_mae={best_mae:.4f}", flush=True)
    return {
        "mae": float(best_mae),
        "device": str(device),
        "pred": best_pred.reshape(-1),
        "label": label,
    }


def main():
    lines: list[str] = []
    t0 = time.time()
    log("=" * 72, lines)
    log("BATCH 1 EXPERIMENTS", lines)
    log(
        f"torch cuda={torch.cuda.is_available()} "
        f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}",
        lines,
    )
    log(f"subsample breaths train/val: {N_TRAIN_BREATHS}/{N_VAL_BREATHS}", lines)
    log("=" * 72, lines)

    train = pd.read_csv(DATA / "train.csv")
    tr_ids, va_ids = make_split(train["breath_id"].unique())
    log(f"actual breaths train/val: {len(tr_ids)}/{len(va_ids)}", lines)

    log("Engineering features...", lines)
    t_feat = time.time()
    # Feature only needed breaths
    need = train["breath_id"].isin(tr_ids | va_ids)
    feat_df = add_features(train.loc[need])
    log(f"feature eng sec={time.time()-t_feat:.1f}", lines)

    tr = feat_df[feat_df["breath_id"].isin(tr_ids)]
    va = feat_df[feat_df["breath_id"].isin(va_ids)]
    unique_p = np.sort(tr["pressure"].unique())

    results = {}

    # ------------------------------------------------------------------
    # E0: ARX on same split (calibrated baseline)
    # ------------------------------------------------------------------
    log("\n--- E0: ARX baseline on same subsample split ---", lines)
    arx_mae = arx_baseline(tr, va)
    log(f"E0 ARX MAE={arx_mae:.4f}", lines)
    results["E0_arx"] = {"mae": arx_mae}

    X_tr = tr[FEATS_BASE].to_numpy(dtype=np.float32)
    y_tr = tr["pressure"].to_numpy(dtype=np.float32)
    X_va = va[FEATS_BASE].to_numpy(dtype=np.float32)
    y_va = va["pressure"].to_numpy(dtype=np.float32)
    u_va = va["u_out"].to_numpy()

    # ------------------------------------------------------------------
    # E1: LightGBM + history, insp-weighted
    # ------------------------------------------------------------------
    log("\n--- E1: LightGBM history features (insp-weighted) ---", lines)
    w_insp = np.where(tr["u_out"].to_numpy() == 0, 1.0, 0.05).astype(np.float32)
    e1 = run_lgbm(X_tr, y_tr, X_va, y_va, u_va, w_tr=w_insp, label="lgbm_hist_insp_w")
    log(f"E1 MAE={e1['mae']:.4f} best_iter={e1['best_iter']}", lines)
    results["E1"] = {k: v for k, v in e1.items() if k != "pred"}

    # ------------------------------------------------------------------
    # E2: loss-mask ablation
    # ------------------------------------------------------------------
    log("\n--- E2: loss-mask ablation for LightGBM ---", lines)
    e2a = run_lgbm(X_tr, y_tr, X_va, y_va, u_va, w_tr=None, label="lgbm_all_phase")
    log(f"E2a all-phase MAE={e2a['mae']:.4f}", lines)

    tr_insp = tr[tr["u_out"] == 0]
    e2b = run_lgbm(
        tr_insp[FEATS_BASE].to_numpy(dtype=np.float32),
        tr_insp["pressure"].to_numpy(dtype=np.float32),
        X_va, y_va, u_va, w_tr=None, label="lgbm_insp_rows_only",
    )
    log(f"E2b insp-rows-only MAE={e2b['mae']:.4f}", lines)
    results["E2a"] = {k: v for k, v in e2a.items() if k != "pred"}
    results["E2b"] = {k: v for k, v in e2b.items() if k != "pred"}

    tree_candidates = [("E1", e1), ("E2a", e2a), ("E2b", e2b)]
    best_tree_name, best_tree = min(tree_candidates, key=lambda kv: kv[1]["mae"])
    log(f"Best tree so far: {best_tree_name} MAE={best_tree['mae']:.4f}", lines)

    # ------------------------------------------------------------------
    # E3: BiLSTM on GPU
    # ------------------------------------------------------------------
    log("\n--- E3: BiLSTM sequence model (GPU) ---", lines)
    seq_cols = ["u_in", "u_out", "R", "C", "time_step", "R_div_C", "u_over_c"]
    Xtr_s, ytr_s, utr_s = reshape_sequences(tr, seq_cols)
    Xva_s, yva_s, uva_s = reshape_sequences(va, seq_cols)
    e3a = run_bilstm(
        Xtr_s, ytr_s, utr_s, Xva_s, yva_s, uva_s,
        mask_train_insp=True, epochs=20, label="bilstm_insp_loss",
    )
    log(f"E3a BiLSTM insp-masked loss MAE={e3a['mae']:.4f}", lines)
    e3b = run_bilstm(
        Xtr_s, ytr_s, utr_s, Xva_s, yva_s, uva_s,
        mask_train_insp=False, epochs=20, label="bilstm_all_loss",
    )
    log(f"E3b BiLSTM all-timestep loss MAE={e3b['mae']:.4f}", lines)
    results["E3a"] = {k: v for k, v in e3a.items() if k != "pred"}
    results["E3b"] = {k: v for k, v in e3b.items() if k != "pred"}

    # ------------------------------------------------------------------
    # E4: lattice snap
    # ------------------------------------------------------------------
    log("\n--- E4: snap predictions to pressure lattice ---", lines)
    best_lstm = e3a if e3a["mae"] <= e3b["mae"] else e3b
    for name, obj, y_true, u in [
        ("tree", best_tree, y_va, u_va),
        ("bilstm", best_lstm, yva_s.reshape(-1), uva_s.reshape(-1)),
    ]:
        raw = obj["pred"]
        snapped = snap_to_lattice(raw, unique_p)
        p0 = unique_p.min()
        snapped2 = p0 + np.round((raw - p0) / PRESSURE_STEP) * PRESSURE_STEP
        mae_raw = insp_mae(y_true, raw, u)
        mae_snap = insp_mae(y_true, snapped, u)
        mae_step = insp_mae(y_true, snapped2, u)
        log(
            f"E4 {name}: raw={mae_raw:.4f} nearest_unique={mae_snap:.4f} step_round={mae_step:.4f}",
            lines,
        )
        results[f"E4_{name}"] = {
            "raw": mae_raw, "nearest_unique": mae_snap, "step_round": mae_step
        }

    # ------------------------------------------------------------------
    # E5: physics features
    # ------------------------------------------------------------------
    log("\n--- E5: physics features for LightGBM ---", lines)
    use_insp_rows = results["E2b"]["mae"] <= min(results["E1"]["mae"], results["E2a"]["mae"])
    if use_insp_rows:
        tr_fit, w, scheme = tr[tr["u_out"] == 0], None, "insp_rows_only"
    elif results["E1"]["mae"] <= results["E2a"]["mae"]:
        tr_fit = tr
        w = np.where(tr["u_out"].to_numpy() == 0, 1.0, 0.05).astype(np.float32)
        scheme = "insp_weighted"
    else:
        tr_fit, w, scheme = tr, None, "all_phase"

    e5a = run_lgbm(
        tr_fit[FEATS_BASE].to_numpy(dtype=np.float32),
        tr_fit["pressure"].to_numpy(dtype=np.float32),
        va[FEATS_BASE].to_numpy(dtype=np.float32),
        y_va, u_va, w_tr=w, label=f"lgbm_base_{scheme}",
    )
    e5b = run_lgbm(
        tr_fit[FEATS_PHYS].to_numpy(dtype=np.float32),
        tr_fit["pressure"].to_numpy(dtype=np.float32),
        va[FEATS_PHYS].to_numpy(dtype=np.float32),
        y_va, u_va, w_tr=w, label=f"lgbm_phys_{scheme}",
    )
    log(f"E5a base feats MAE={e5a['mae']:.4f} scheme={scheme}", lines)
    log(f"E5b phys feats MAE={e5b['mae']:.4f} scheme={scheme}", lines)
    results["E5a"] = {k: v for k, v in e5a.items() if k != "pred"}
    results["E5b"] = {k: v for k, v in e5b.items() if k != "pred"}

    # ------------------------------------------------------------------
    # Complementarity probe
    # ------------------------------------------------------------------
    log("\n--- Complementarity probe (mean ensemble) ---", lines)
    va_sorted = va.sort_values(["breath_id", "time_step"])
    tree_pred_series = pd.Series(best_tree["pred"], index=va.index)
    tree_aligned = tree_pred_series.loc[va_sorted.index].to_numpy()
    lstm_aligned = best_lstm["pred"]
    y_aligned = va_sorted["pressure"].to_numpy()
    u_aligned = va_sorted["u_out"].to_numpy()
    ens_mean = 0.5 * tree_aligned + 0.5 * lstm_aligned
    ens_mae = insp_mae(y_aligned, ens_mean, u_aligned)
    m = u_aligned == 0
    corr = float(np.corrcoef(tree_aligned[m] - y_aligned[m], lstm_aligned[m] - y_aligned[m])[0, 1])
    tree_mae = insp_mae(y_aligned, tree_aligned, u_aligned)
    lstm_mae = insp_mae(y_aligned, lstm_aligned, u_aligned)
    log(
        f"Ensemble mean MAE={ens_mae:.4f} (tree={tree_mae:.4f}, lstm={lstm_mae:.4f}) err_corr={corr:.4f}",
        lines,
    )
    results["ensemble_mean"] = {
        "mae": ens_mae, "err_corr": corr, "tree": tree_mae, "lstm": lstm_mae
    }

    log("\n" + "=" * 72, lines)
    log("BATCH 1 SUMMARY (inspiratory MAE, same subsample split)", lines)
    log("=" * 72, lines)
    flat = {
        "E0_arx": results["E0_arx"]["mae"],
        "E1_lgbm_insp_w": results["E1"]["mae"],
        "E2a_lgbm_all_phase": results["E2a"]["mae"],
        "E2b_lgbm_insp_rows": results["E2b"]["mae"],
        "E3a_bilstm_insp_loss": results["E3a"]["mae"],
        "E3b_bilstm_all_loss": results["E3b"]["mae"],
        "E5a_base": results["E5a"]["mae"],
        "E5b_phys": results["E5b"]["mae"],
        "ens_tree_lstm": results["ensemble_mean"]["mae"],
    }
    for k, v in sorted(flat.items(), key=lambda kv: kv[1]):
        log(f"  {v:.4f}  {k}", lines)
    best_name = min(flat, key=flat.get)
    log(f"\nBest this batch: {best_name} = {flat[best_name]:.4f}", lines)
    log(f"Elapsed sec: {time.time() - t0:.1f}", lines)

    report = "\n".join(lines)
    (OUT / "batch1_report.txt").write_text(report, encoding="utf-8")
    (OUT / "batch1_results.json").write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    log(f"Wrote {OUT / 'batch1_report.txt'}", lines)


if __name__ == "__main__":
    main()
