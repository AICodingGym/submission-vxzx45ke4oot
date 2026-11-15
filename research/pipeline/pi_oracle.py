"""Label-supervised PI oracle: find the discretization/T that actually fits train u_in.

Uses true pressure only for diagnosis. Inverse search is separate.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inverse_control"))

from inv_common import (
    CLIP_EPS,
    I_COEF_GRID,
    P_COEF_GRID,
    P_STAR_GRID,
    U_CLIP_HI,
    U_CLIP_LO,
)
from pipe_common import DATA, PIPE, keep_awake

PC = P_COEF_GRID[P_COEF_GRID > 1e-8].astype(np.float64)
IC = I_COEF_GRID[np.abs(I_COEF_GRID) > 1e-8].astype(np.float64)
PS = P_STAR_GRID.astype(np.float64)


@njit(cache=True)
def forward_u(p, dt, u, clip, pc, ic, ps, T, q_first):
    """Simulate u from true p. q_first=1: AmbrosM (update q then u). q_first=0: u then q."""
    n = p.shape[0]
    q0 = (u[0] - pc * (ps - p[0])) / ic
    q = q0
    n_ok = 0
    n_use = 0
    abs_err = 0.0
    # step 0 always uses q0
    if not clip[0]:
        n_use += 1
        uh = pc * (ps - p[0]) + ic * q
        if abs(uh - u[0]) < 0.05:
            n_ok += 1
        abs_err += abs(uh - u[0])
    for i in range(1, n):
        s = dt[i] / (dt[i] + T)
        if q_first == 1:
            q = (1.0 - s) * q + s * (ps - p[i])
            uh = pc * (ps - p[i]) + ic * q
        else:
            uh = pc * (ps - p[i]) + ic * q
            q = (1.0 - s) * q + s * (ps - p[i])
        if clip[i]:
            continue
        n_use += 1
        d = abs(uh - u[i])
        abs_err += d
        if d < 0.05:
            n_ok += 1
    frac = n_ok / n_use if n_use > 0 else 0.0
    mae = abs_err / n_use if n_use > 0 else 1e9
    return frac, mae, n_ok, n_use, q0


@njit(cache=True)
def best_forward(p, dt, u, clip, pc_g, ic_g, ps_g, T, q_first):
    best_frac = -1.0
    best_mae = 1e9
    best_pc = 0.0
    best_ic = 0.0
    best_ps = 0.0
    best_q0 = 0.0
    best_ok = 0
    best_use = 0
    for ipc in range(pc_g.shape[0]):
        pc = pc_g[ipc]
        for iic in range(ic_g.shape[0]):
            ic = ic_g[iic]
            for ips in range(ps_g.shape[0]):
                ps = ps_g[ips]
                frac, mae, n_ok, n_use, q0 = forward_u(p, dt, u, clip, pc, ic, ps, T, q_first)
                if n_use < 8:
                    continue
                better = frac > best_frac + 1e-12 or (abs(frac - best_frac) <= 1e-12 and mae < best_mae)
                if better:
                    best_frac = frac
                    best_mae = mae
                    best_pc = pc
                    best_ic = ic
                    best_ps = ps
                    best_q0 = q0
                    best_ok = n_ok
                    best_use = n_use
    return best_frac, best_mae, best_pc, best_ic, best_ps, best_q0, best_ok, best_use


def pack(df, n=None):
    df = df.sort_values(["breath_id", "time_step"])
    nb = len(df) // 80
    if n is not None:
        nb = min(nb, n)
        df = df.iloc[: nb * 80]
    t = df["time_step"].to_numpy(np.float64).reshape(nb, 80)
    dt = np.diff(t, axis=1, prepend=t[:, :1])
    dt[:, 0] = 0.0
    u = df["u_in"].to_numpy(np.float64).reshape(nb, 80)
    p = df["pressure"].to_numpy(np.float64).reshape(nb, 80)
    uout = df["u_out"].to_numpy(np.float64).reshape(nb, 80)
    clip = (u <= U_CLIP_LO + CLIP_EPS) | (u >= U_CLIP_HI - CLIP_EPS) | (uout != 0)
    return p, dt, u, clip


def main():
    n = 800
    train = pd.read_csv(DATA / "train.csv")
    p, dt, u, clip = pack(train, n)
    configs = []
    for T in (0.3, 0.4, 0.5, 0.6, 0.8, 1.0):
        for q_first in (1, 0):
            configs.append((T, q_first))

    # warmup
    best_forward(p[0], dt[0], u[0], clip[0], PC, IC, PS, 0.5, 1)

    out = []
    with keep_awake():
        for T, q_first in configs:
            t0 = time.time()
            fracs = np.empty(n, np.float64)
            maes = np.empty(n, np.float64)
            for i in range(n):
                frac, mae, *_ = best_forward(p[i], dt[i], u[i], clip[i], PC, IC, PS, T, q_first)
                fracs[i] = frac
                maes[i] = mae
            rec = {
                "T": T,
                "q_first": int(q_first),
                "mean_frac": float(fracs.mean()),
                "pct_frac_ge_0.85": float(100.0 * (fracs >= 0.85).mean()),
                "pct_frac_ge_0.50": float(100.0 * (fracs >= 0.50).mean()),
                "pct_mae_lt_0.05": float(100.0 * (maes < 0.05).mean()),
                "pct_mae_lt_0.5": float(100.0 * (maes < 0.5).mean()),
                "median_mae": float(np.median(maes)),
                "median_frac": float(np.median(fracs)),
                "sec": time.time() - t0,
            }
            out.append(rec)
            print(json.dumps(rec), flush=True)

    best = max(out, key=lambda r: (r["pct_frac_ge_0.85"], -r["median_mae"]))
    payload = {"n": n, "configs": out, "best": best}
    (PIPE / "pi_oracle.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("BEST", json.dumps(best), flush=True)


if __name__ == "__main__":
    main()
