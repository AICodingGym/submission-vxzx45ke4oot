"""Oracle: fit q0 by least squares / median (u[0] may be perturbed)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inverse_control"))

from inv_common import CLIP_EPS, I_COEF_GRID, P_COEF_GRID, P_STAR_GRID, U_CLIP_HI, U_CLIP_LO
from pipe_common import DATA, PIPE, keep_awake

PC = P_COEF_GRID[P_COEF_GRID > 1e-8].astype(np.float64)
IC = I_COEF_GRID[np.abs(I_COEF_GRID) > 1e-8].astype(np.float64)
PS = P_STAR_GRID.astype(np.float64)
T = 0.5


@njit(cache=True)
def score_ls_q0(p, dt, u, clip, pc, ic, ps, T):
    n = p.shape[0]
    A = np.empty(n, np.float64)
    B = np.empty(n, np.float64)
    A[0] = 1.0
    B[0] = 0.0
    for i in range(1, n):
        s = dt[i] / (dt[i] + T)
        A[i] = (1.0 - s) * A[i - 1]
        B[i] = (1.0 - s) * B[i - 1] + s * (ps - p[i])
    # median q0 from unclipped steps
    qs = np.empty(n, np.float64)
    nq = 0
    for i in range(n):
        if clip[i]:
            continue
        d = ic * A[i]
        if abs(d) < 1e-12:
            continue
        c = pc * (ps - p[i]) + ic * B[i]
        qs[nq] = (u[i] - c) / d
        nq += 1
    if nq < 8:
        return 0.0, 1e9, 0.0, 0, 0
    # median
    tmp = qs[:nq].copy()
    tmp.sort()
    q0 = tmp[nq // 2]
    n_ok = 0
    n_use = 0
    abs_err = 0.0
    for i in range(n):
        if clip[i]:
            continue
        uh = pc * (ps - p[i]) + ic * (A[i] * q0 + B[i])
        n_use += 1
        dlt = abs(uh - u[i])
        abs_err += dlt
        if dlt < 0.05:
            n_ok += 1
    frac = n_ok / n_use if n_use else 0.0
    mae = abs_err / n_use if n_use else 1e9
    return frac, mae, q0, n_ok, n_use


@njit(cache=True)
def best_ls(p, dt, u, clip, pc_g, ic_g, ps_g, T):
    best_frac = -1.0
    best_mae = 1e9
    best_pc = 0.0
    best_ic = 0.0
    best_ps = 0.0
    best_q0 = 0.0
    for ipc in range(pc_g.shape[0]):
        for iic in range(ic_g.shape[0]):
            for ips in range(ps_g.shape[0]):
                frac, mae, q0, n_ok, n_use = score_ls_q0(
                    p, dt, u, clip, pc_g[ipc], ic_g[iic], ps_g[ips], T
                )
                if n_use < 8:
                    continue
                if frac > best_frac + 1e-12 or (abs(frac - best_frac) <= 1e-12 and mae < best_mae):
                    best_frac = frac
                    best_mae = mae
                    best_pc = pc_g[ipc]
                    best_ic = ic_g[iic]
                    best_ps = ps_g[ips]
                    best_q0 = q0
    return best_frac, best_mae, best_pc, best_ic, best_ps, best_q0


@njit(cache=True)
def score_p_only(p, u, clip, pc_g, ps_g):
    """Pure P: u = pc*(ps-p)."""
    best_frac = -1.0
    best_mae = 1e9
    n = p.shape[0]
    for ipc in range(pc_g.shape[0]):
        pc = pc_g[ipc]
        for ips in range(ps_g.shape[0]):
            ps = ps_g[ips]
            n_ok = 0
            n_use = 0
            abs_err = 0.0
            for i in range(n):
                if clip[i]:
                    continue
                uh = pc * (ps - p[i])
                n_use += 1
                dlt = abs(uh - u[i])
                abs_err += dlt
                if dlt < 0.05:
                    n_ok += 1
            if n_use < 8:
                continue
            frac = n_ok / n_use
            mae = abs_err / n_use
            if frac > best_frac + 1e-12 or (abs(frac - best_frac) <= 1e-12 and mae < best_mae):
                best_frac = frac
                best_mae = mae
    return best_frac, best_mae


def main():
    n = 800
    df = pd.read_csv(DATA / "train.csv")
    df = df.sort_values(["breath_id", "time_step"]).iloc[: n * 80]
    t = df["time_step"].to_numpy(np.float64).reshape(n, 80)
    dt = np.diff(t, axis=1, prepend=t[:, :1])
    dt[:, 0] = 0.0
    u = df["u_in"].to_numpy(np.float64).reshape(n, 80)
    p = df["pressure"].to_numpy(np.float64).reshape(n, 80)
    uout = df["u_out"].to_numpy(np.float64).reshape(n, 80)
    clip = (u <= U_CLIP_LO + CLIP_EPS) | (u >= U_CLIP_HI - CLIP_EPS) | (uout != 0)

    print(
        json.dumps({
            "dt_median": float(np.median(dt[:, 1:])),
            "dt_mean": float(dt[:, 1:].mean()),
            "t0_median": float(np.median(t[:, 0])),
            "t_max_median": float(np.median(t[:, -1])),
            "u0_clip_pct": float(100.0 * clip[:, 0].mean()),
            "insp_frac": float((uout == 0).mean()),
        }),
        flush=True,
    )

    best_ls(p[0], dt[0], u[0], clip[0], PC, IC, PS, T)
    score_p_only(p[0], u[0], clip[0], PC, PS)

    fracs = np.empty(n)
    maes = np.empty(n)
    pfracs = np.empty(n)
    pmaes = np.empty(n)
    with keep_awake():
        for i in range(n):
            f, m, *_ = best_ls(p[i], dt[i], u[i], clip[i], PC, IC, PS, T)
            fracs[i] = f
            maes[i] = m
            pf, pm = score_p_only(p[i], u[i], clip[i], PC, PS)
            pfracs[i] = pf
            pmaes[i] = pm
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{n} PI>=0.85 {100*(fracs[:i+1]>=0.85).mean():.1f}% P>=0.85 {100*(pfracs[:i+1]>=0.85).mean():.1f}%", flush=True)

    rec = {
        "n": n,
        "PI_ls_q0": {
            "mean_frac": float(fracs.mean()),
            "pct_frac_ge_0.85": float(100.0 * (fracs >= 0.85).mean()),
            "pct_frac_ge_0.50": float(100.0 * (fracs >= 0.50).mean()),
            "pct_mae_lt_0.05": float(100.0 * (maes < 0.05).mean()),
            "median_mae": float(np.median(maes)),
            "median_frac": float(np.median(fracs)),
        },
        "P_only": {
            "mean_frac": float(pfracs.mean()),
            "pct_frac_ge_0.85": float(100.0 * (pfracs >= 0.85).mean()),
            "pct_frac_ge_0.50": float(100.0 * (pfracs >= 0.50).mean()),
            "median_mae": float(np.median(pmaes)),
        },
        "union_ge_0.85": float(100.0 * ((fracs >= 0.85) | (pfracs >= 0.85)).mean()),
    }
    print(json.dumps(rec, indent=2), flush=True)
    (PIPE / "pi_oracle_q0.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
