"""Pulp Fiction feature set, reimplemented on gym train/test (no third-party weights)."""
from __future__ import annotations

import numpy as np
import pandas as pd

R_VALS = (5, 20, 50)
C_VALS = (10, 20, 50)


def _shift(x: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros_like(x)
    if k > 0:
        out[:, k:] = x[:, :-k]
    elif k < 0:
        out[:, :k] = x[:, -k:]
    return out


def _roll_sum(x: np.ndarray, w: int = 15) -> np.ndarray:
    c = np.cumsum(x, axis=1)
    s = np.empty_like(x)
    s[:, :w] = c[:, :w]
    s[:, w:] = c[:, w:] - c[:, :-w]
    return s


def _roll_minmax(x: np.ndarray, w: int = 15) -> tuple[np.ndarray, np.ndarray]:
    n, t = x.shape
    rmin = np.empty_like(x)
    rmax = np.empty_like(x)
    for i in range(t):
        sl = x[:, max(0, i - w + 1) : i + 1]
        rmin[:, i] = sl.min(axis=1)
        rmax[:, i] = sl.max(axis=1)
    return rmin, rmax


def _ewm(x: np.ndarray, halflife: float = 9.0) -> np.ndarray:
    alpha = 1.0 - np.exp(-np.log(2.0) / halflife)
    out = np.empty_like(x)
    out[:, 0] = x[:, 0]
    for i in range(1, x.shape[1]):
        out[:, i] = alpha * x[:, i] + (1.0 - alpha) * out[:, i - 1]
    return out


def pf_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str], dict]:
    """Return X (n_breaths, 80, F), column names, and per-breath meta."""
    df = df.sort_values(["breath_id", "time_step"])
    n = len(df) // 80
    if n * 80 != len(df):
        raise ValueError(f"rows {len(df)} not divisible by 80")
    u_in = df["u_in"].to_numpy(np.float32).reshape(n, 80)
    u_out = df["u_out"].to_numpy(np.float32).reshape(n, 80)
    ts = df["time_step"].to_numpy(np.float32).reshape(n, 80)
    R = df["R"].to_numpy(np.float32).reshape(n, 80)
    C = df["C"].to_numpy(np.float32).reshape(n, 80)
    bids = df["breath_id"].to_numpy().reshape(n, 80)[:, 0]
    ids = df["id"].to_numpy().reshape(n, 80) if "id" in df.columns else np.arange(n * 80).reshape(n, 80)
    y = (
        df["pressure"].to_numpy(np.float32).reshape(n, 80)
        if "pressure" in df.columns
        else np.zeros((n, 80), np.float32)
    )

    dt = np.diff(ts, axis=1, prepend=ts[:, :1])
    dt[:, 0] = 0.0
    channels: dict[str, np.ndarray] = {
        "u_in": u_in,
        "u_out": u_out,
        "time_step": ts,
        "cross": u_in * u_out,
        "cross2": ts * u_out,
        "area": np.cumsum(ts * u_in, axis=1),
        "time_step_cumsum": np.cumsum(ts, axis=1),
        "u_in_cumsum": np.cumsum(u_in, axis=1),
        "time_step_diff": dt,
    }
    for k in range(1, 5):
        channels[f"u_in_lag{k}"] = _shift(u_in, k)
        channels[f"u_out_lag{k}"] = _shift(u_out, k)
        channels[f"u_in_lag_back{k}"] = _shift(u_in, -k)
        channels[f"u_out_lag_back{k}"] = _shift(u_out, -k)
        channels[f"u_in_diff{k}"] = u_in - channels[f"u_in_lag{k}"]
        channels[f"u_out_diff{k}"] = u_out - channels[f"u_out_lag{k}"]
    umax = u_in.max(axis=1, keepdims=True)
    umean = u_in.mean(axis=1, keepdims=True)
    channels["breath_u_in_max"] = np.repeat(umax, 80, axis=1)
    channels["breath_u_in_mean"] = np.repeat(umean, 80, axis=1)
    channels["breath_u_in_diffmax"] = umax - u_in
    channels["breath_u_in_diffmean"] = umean - u_in
    channels["u_in_cummean"] = channels["u_in_cumsum"] / np.arange(1, 81, dtype=np.float32)[None, :]
    channels["u_in_lagback_diff1"] = u_in - channels["u_in_lag_back1"]
    channels["u_out_lagback_diff1"] = u_out - channels["u_out_lag_back1"]
    channels["u_in_lagback_diff2"] = u_in - channels["u_in_lag_back2"]
    channels["u_out_lagback_diff2"] = u_out - channels["u_out_lag_back2"]
    channels["ewm_u_in_mean"] = _ewm(u_in, 9.0)
    rsum = _roll_sum(u_in, 15)
    rmin, rmax = _roll_minmax(u_in, 15)
    count = np.minimum(np.arange(1, 81, dtype=np.float32)[None, :], 15.0)
    channels["win15_sum"] = rsum
    channels["win15_min"] = rmin
    channels["win15_max"] = rmax
    channels["win15_mean"] = rsum / count
    r0, c0 = R[:, 0], C[:, 0]
    for rv in R_VALS:
        channels[f"R_{rv}"] = np.broadcast_to((r0 == rv).astype(np.float32)[:, None], (n, 80)).copy()
    for cv in C_VALS:
        channels[f"C_{cv}"] = np.broadcast_to((c0 == cv).astype(np.float32)[:, None], (n, 80)).copy()
    for rv in R_VALS:
        for cv in C_VALS:
            channels[f"RC_{rv}_{cv}"] = np.broadcast_to(
                ((r0 == rv) & (c0 == cv)).astype(np.float32)[:, None], (n, 80)
            ).copy()

    cols = list(channels.keys())
    X = np.stack([channels[c] for c in cols], axis=-1).astype(np.float32)
    meta = {"breath_id": bids, "id": ids, "y": y, "u_out": u_out, "R": R, "C": C, "u_in": u_in, "time_step": ts}
    return X, cols, meta


def robust_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = X.reshape(-1, X.shape[-1])
    q1 = np.quantile(flat, 0.25, axis=0).astype(np.float32)
    q3 = np.quantile(flat, 0.75, axis=0).astype(np.float32)
    med = np.median(flat, axis=0).astype(np.float32)
    iqr = q3 - q1
    iqr = np.where(iqr < 1e-6, 1.0, iqr).astype(np.float32)
    return med, iqr


def robust_apply(X: np.ndarray, med: np.ndarray, iqr: np.ndarray) -> np.ndarray:
    return ((X - med) / iqr).astype(np.float32)
