"""Shared utilities for Batch 2."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

DATA = Path(__file__).resolve().parents[1] / "data" / "ventilator-pressure-prediction"
OUT = Path(__file__).resolve().parent
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEQ_COLS_BASE = [
    "u_in", "u_out", "R", "C", "time_step", "R_div_C", "u_over_c",
    "u_in_lag1", "u_in_cumsum", "u_in_cummax", "du", "insp_len",
    "steps_to_exp", "u_in_backcum", "vol_over_c",
]
SEQ_COLS_RICH = SEQ_COLS_BASE + [
    "u_in_lag2", "u_in_lag3", "du2", "u_in_roll3", "u_in_cummean",
    "u_r_over_c", "vol_r_over_c", "R_C", "u_out_lag1",
    "area_uin", "uin_sq_cum", "dvol_over_c",
]


def insp_mae(y_true, y_pred, u_out) -> float:
    m = u_out == 0
    return float(np.mean(np.abs(y_true[m] - y_pred[m])))


def insp_mae_by_rc(y_true, y_pred, u_out, R, C, r=50, c=10) -> float:
    m = (u_out == 0) & (R == r) & (C == c)
    if m.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true[m] - y_pred[m])))


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
    dt = g["time_step"].diff().fillna(0.033)
    df["area_uin"] = (df["u_in"] * dt).groupby(df["breath_id"]).cumsum()
    df["uin_sq_cum"] = (df["u_in"] ** 2).groupby(df["breath_id"]).cumsum()
    df["u_over_c"] = df["u_in"] / df["C"]
    df["u_r_over_c"] = df["u_in"] * df["R"] / df["C"]
    df["vol_over_c"] = df["u_in_cumsum"] / df["C"]
    df["vol_r_over_c"] = df["u_in_cumsum"] * df["R"] / df["C"]
    df["dvol_over_c"] = df["du"] / df["C"]
    df["R_C"] = df["R"] * df["C"]
    df["R_div_C"] = df["R"] / df["C"]
    return df


def reshape_sequences(df: pd.DataFrame, cols: list[str]):
    df = df.sort_values(["breath_id", "time_step"])
    X = df[cols].to_numpy(np.float32).reshape(-1, 80, len(cols))
    y = df["pressure"].to_numpy(np.float32).reshape(-1, 80)
    u = df["u_out"].to_numpy(np.float32).reshape(-1, 80)
    R = df["R"].to_numpy(np.float32).reshape(-1, 80)
    C = df["C"].to_numpy(np.float32).reshape(-1, 80)
    bids = df["breath_id"].to_numpy().reshape(-1, 80)[:, 0]
    return X, y, u, R, C, bids


def norm_fit(X):
    mean = X.reshape(-1, X.shape[-1]).mean(0).astype(np.float32)
    std = X.reshape(-1, X.shape[-1]).std(0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


def apply_norm(X, mean, std):
    return ((X - mean) / std).astype(np.float32)


class BiLSTMReg(nn.Module):
    def __init__(self, n_feat, hidden=256, layers=3, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            n_feat, hidden, num_layers=layers, batch_first=True,
            bidirectional=True, dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        h, _ = self.lstm(x)
        return self.head(h).squeeze(-1)


class ConvBiLSTM(nn.Module):
    def __init__(self, n_feat, hidden=256, layers=3, dropout=0.1, conv_channels=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_feat, conv_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=5, padding=2),
            nn.GELU(),
        )
        self.lstm = nn.LSTM(
            conv_channels, hidden, num_layers=layers, batch_first=True,
            bidirectional=True, dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        # x: B,T,F -> conv needs B,F,T
        c = self.conv(x.transpose(1, 2)).transpose(1, 2)
        h, _ = self.lstm(c)
        return self.head(h).squeeze(-1)


class AttnBiLSTM(nn.Module):
    def __init__(self, n_feat, hidden=256, layers=3, dropout=0.1, n_heads=4):
        super().__init__()
        self.lstm = nn.LSTM(
            n_feat, hidden, num_layers=layers, batch_first=True,
            bidirectional=True, dropout=dropout if layers > 1 else 0.0,
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden * 2, num_heads=n_heads, dropout=dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden * 2)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        h, _ = self.lstm(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        h = self.norm(h + a)
        return self.head(h).squeeze(-1)


def masked_loss(pred, y, u, kind="mae", hard_weight=1.0, R=None, C=None):
    """kind in mae|mse|huber. Optional upweight R=50,C=10 inspiratory steps."""
    m = u == 0
    if kind == "mae":
        err = torch.abs(pred - y)
    elif kind == "mse":
        err = (pred - y) ** 2
    elif kind == "huber":
        err = nn.functional.huber_loss(pred, y, reduction="none", delta=1.0)
    else:
        raise ValueError(kind)
    w = m.float()
    if hard_weight != 1.0 and R is not None and C is not None:
        hard = m & (R == 50) & (C == 10)
        w = w + hard.float() * (hard_weight - 1.0)
    denom = w.sum().clamp_min(1.0)
    return (err * w).sum() / denom


@torch.no_grad()
def predict_np(model, X, bs=512):
    model.eval()
    outs = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i : i + bs]).to(DEVICE, non_blocking=True)
        outs.append(model(xb).float().cpu().numpy())
    return np.concatenate(outs, axis=0)


def train_model(
    model,
    X_tr, y_tr, u_tr,
    X_va, y_va, u_va,
    *,
    R_tr=None, C_tr=None, R_va=None, C_va=None,
    epochs=60,
    lr=1e-3,
    batch_size=256,
    loss_kind="mae",
    schedule="plateau",  # plateau|cosine|none
    y_scale=False,
    hard_weight=1.0,
    patience=12,
    min_lr=1e-5,
    label="",
    log_every=2,
    use_amp: bool = True,
    screen_stop=None,
    screen_leader_mae=None,
):
    """Returns dict with best val insp MAE (unscaled), preds, history."""
    if screen_stop is not None:
        patience = screen_stop.plateau_patience
    model = model.to(DEVICE)
    use_amp = use_amp and DEVICE.type == "cuda"
    # Target scaling
    if y_scale:
        y_mean = float(y_tr[u_tr == 0].mean()) if (u_tr == 0).any() else float(y_tr.mean())
        y_std = float(y_tr[u_tr == 0].std()) if (u_tr == 0).any() else float(y_tr.std())
        y_std = max(y_std, 1e-6)
        y_tr_t = (y_tr - y_mean) / y_std
        y_va_t = (y_va - y_mean) / y_std
    else:
        y_mean, y_std = 0.0, 1.0
        y_tr_t, y_va_t = y_tr, y_va

    tensors = [
        torch.from_numpy(X_tr),
        torch.from_numpy(y_tr_t.astype(np.float32)),
        torch.from_numpy(u_tr),
    ]
    if R_tr is not None:
        tensors += [torch.from_numpy(R_tr), torch.from_numpy(C_tr)]
    loader = DataLoader(
        TensorDataset(*tensors),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        num_workers=0,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    if schedule == "plateau":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=patience, min_lr=min_lr
        )
    elif schedule == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=min_lr)
    else:
        sched = None

    history = []
    best = {"mae": float("inf"), "epoch": 0, "pred": None, "train_mae": None, "state": None}
    stale = 0
    stop_patience = (
        screen_stop.improve_patience if screen_stop is not None
        else max(18, patience * 2 + 3)
    )
    stop_reason = "max_epochs"
    for ep in range(1, epochs + 1):
        model.train()
        for batch in loader:
            xb = batch[0].to(DEVICE, non_blocking=True)
            yb = batch[1].to(DEVICE, non_blocking=True)
            ub = batch[2].to(DEVICE, non_blocking=True)
            Rb = batch[3].to(DEVICE, non_blocking=True) if len(batch) > 3 else None
            Cb = batch[4].to(DEVICE, non_blocking=True) if len(batch) > 4 else None
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = model(xb)
                loss = masked_loss(pred, yb, ub, kind=loss_kind, hard_weight=hard_weight, R=Rb, C=Cb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

        # eval unscaled (train MAE only periodically — full-train predict is expensive/VRAM-heavy)
        do_train_eval = (ep == 1 or ep == epochs or ep % max(log_every, 5) == 0 or stale > stop_patience - 2)
        pred_va_s = predict_np(model, X_va)
        pred_va = pred_va_s * y_std + y_mean
        va_mae = insp_mae(y_va.reshape(-1), pred_va.reshape(-1), u_va.reshape(-1))
        if do_train_eval:
            pred_tr_s = predict_np(model, X_tr)
            pred_tr = pred_tr_s * y_std + y_mean
            tr_mae = insp_mae(y_tr.reshape(-1), pred_tr.reshape(-1), u_tr.reshape(-1))
        else:
            tr_mae = float("nan")
        hard_mae = None
        if R_va is not None:
            hard_mae = insp_mae_by_rc(
                y_va.reshape(-1), pred_va.reshape(-1), u_va.reshape(-1),
                R_va.reshape(-1), C_va.reshape(-1),
            )
        cur_lr = opt.param_groups[0]["lr"]
        history.append({"epoch": ep, "train": tr_mae, "val": va_mae, "hard": hard_mae, "lr": cur_lr})
        if schedule == "plateau":
            sched.step(va_mae)
        elif schedule == "cosine":
            sched.step()

        if va_mae < best["mae"] - 1e-5:
            # refresh train MAE at best checkpoint
            if np.isnan(tr_mae):
                pred_tr_s = predict_np(model, X_tr)
                tr_mae = insp_mae(y_tr.reshape(-1), (pred_tr_s * y_std + y_mean).reshape(-1), u_tr.reshape(-1))
            best = {
                "mae": va_mae,
                "epoch": ep,
                "pred": pred_va.copy(),
                "train_mae": tr_mae,
                "hard": hard_mae,
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "y_mean": y_mean,
                "y_std": y_std,
            }
            stale = 0
        else:
            stale += 1
        if log_every and (ep % log_every == 0 or ep == 1 or ep == epochs):
            extra = f" hard50_10={hard_mae:.4f}" if hard_mae is not None else ""
            tr_s = f"{tr_mae:.4f}" if not np.isnan(tr_mae) else "NA"
            print(
                f"  [{label}] ep={ep:03d} train={tr_s} val={va_mae:.4f} "
                f"lr={cur_lr:.2e}{extra}",
                flush=True,
            )
        if screen_stop is not None and ep >= screen_stop.warmup_epochs:
            from batch3_screen_policy import check_screen_stop
            val_hist = [h["val"] for h in history]
            do_stop, why = check_screen_stop(
                ep, stale, va_mae, val_hist,
                cfg=screen_stop, leader_mae=screen_leader_mae,
            )
            if do_stop:
                stop_reason = why
                print(f"  [{label}] screen stop at ep={ep}: {why}", flush=True)
                break
        if stale >= stop_patience:
            stop_reason = f"no_improve_{stale}ep"
            print(f"  [{label}] early stop at ep={ep} (no val improve for {stale})", flush=True)
            break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "stopping_reason": stop_reason,
        "history": history,
        "epochs_ran": len(history),
        "model": model,
        "best_val": float(best["mae"]),
        "best_train": float(best["train_mae"]) if best["train_mae"] is not None else None,
        "best_epoch": int(best["epoch"]),
        "best_hard": float(best["hard"]) if best.get("hard") is not None else None,
        "pred_va": best["pred"],
        "history": history,
        "y_mean": best.get("y_mean", 0.0),
        "y_std": best.get("y_std", 1.0),
        "label": label,
    }


def breath_split(breath_ids, val_frac=0.2, seed=SEED):
    rng = np.random.default_rng(seed)
    ids = np.array(breath_ids).copy()
    rng.shuffle(ids)
    n_val = int(len(ids) * val_frac)
    return set(ids[n_val:].tolist()), set(ids[:n_val].tolist())


FEATS_TREE = [
    "R", "C", "u_in", "u_out", "time_step", "step",
    "u_in_lag1", "u_in_lag2", "u_in_lag3", "du", "du2",
    "u_in_cumsum", "u_in_cummax", "u_in_cummean", "u_in_roll3",
    "u_out_lag1", "insp_len", "steps_to_exp", "u_in_backcum",
    "R_C", "R_div_C", "u_over_c", "u_r_over_c", "vol_over_c", "vol_r_over_c",
    "area_uin", "uin_sq_cum", "dvol_over_c",
]
