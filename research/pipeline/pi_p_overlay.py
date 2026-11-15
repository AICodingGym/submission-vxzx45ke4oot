"""Pure-P overlay on steps the q0 PI did not trust. Fast; uses existing ensemble OOF."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inverse_control"))

from inv_common import (
    CLIP_EPS,
    P_COEF_GRID,
    P_STAR_GRID,
    U_CLIP_HI,
    U_CLIP_LO,
    on_unique_pressure,
    snap_unique,
)
from pipe_common import DATA, PIPE, RESEARCH, insp_mae, keep_awake

PC = P_COEF_GRID[P_COEF_GRID > 1e-8]
PS = P_STAR_GRID
W_E1, W_E3 = 0.4771337422498183, 0.5228662577501817


def identify_p_breath(uin, uout, uniq):
    insp = uout == 0
    clipped = (uin <= U_CLIP_LO + CLIP_EPS) | (uin >= U_CLIP_HI - CLIP_EPS)
    m = insp & ~clipped
    if int(m.sum()) < 8:
        return None
    u = uin[m]
    ph = PS[None, :, None] - u[None, None, :] / PC[:, None, None]
    hits = on_unique_pressure(ph, uniq).sum(axis=-1)
    k, s = np.unravel_index(int(np.argmax(hits)), hits.shape)
    frac = float(hits[k, s]) / max(int(m.sum()), 1)
    if hits[k, s] < 8 or frac < 0.70:
        return None
    pred = np.full(len(uin), np.nan)
    pred[~clipped] = PS[s] - uin[~clipped] / PC[k]
    trust = insp & ~clipped & on_unique_pressure(pred, uniq)
    if int(trust.sum()) < 8:
        return None
    pred_out = np.full(len(uin), np.nan)
    pred_out[trust] = snap_unique(pred[trust], uniq)
    return pred_out, trust, frac


def main():
    train = pd.read_csv(DATA / "train.csv")
    uniq = np.sort(train["pressure"].unique().astype(np.float64))
    e3 = np.load(RESEARCH / "batch3_oof_e3_aug_uin.npz")
    e1 = np.load(RESEARCH / "batch3_oof_e1_baseline.npz")
    ens = snap_unique((W_E1 * e1["oof"] + W_E3 * e3["oof"]).reshape(-1), uniq).reshape(e3["oof"].shape)
    q0 = np.load(PIPE / "hybrid_q0_oof.npz")
    hy = q0["hybrid"].astype(np.float64)
    already = q0["trust"]
    y, u = e3["y"], e3["u_out"]
    print("baseline hybrid", insp_mae(y, hy, u), flush=True)

    df = train.sort_values(["breath_id", "time_step"])
    n = len(df) // 80
    uin = df["u_in"].to_numpy(np.float64).reshape(n, 80)
    uout = df["u_out"].to_numpy(np.float64).reshape(n, 80)
    bids = df["breath_id"].to_numpy().reshape(n, 80)[:, 0]
    pos = {int(b): i for i, b in enumerate(bids)}
    take = np.array([pos[int(b)] for b in e3["breath_id"]])

    n_add = 0
    p_pred = np.full_like(hy, np.nan)
    p_trust = np.zeros_like(already)
    with keep_awake():
        for j, i in enumerate(take):
            rec = identify_p_breath(uin[i], uout[i], uniq)
            if rec is None:
                continue
            pred, trust, frac = rec
            extra = trust & ~already[j]
            if extra.sum() < 3:
                continue
            p_pred[j] = pred
            p_trust[j] = extra
            n_add += 1
    print(f"P extra breaths={n_add} extra_steps={int(p_trust.sum())}", flush=True)

    rows = []
    for tau in (0.5, 1.0, 1.5, 2.0, 3.0, 5.0):
        g = p_trust & np.isfinite(p_pred) & (np.abs(p_pred - ens) < tau)
        out = hy.copy()
        out[g] = p_pred[g]
        rec = {"tau": tau, "hybrid": insp_mae(y, out, u), "extra_cov": float(100 * g.reshape(-1)[u.reshape(-1) == 0].mean())}
        rows.append(rec)
        print(rec, flush=True)
    best = min(rows, key=lambda r: r["hybrid"])
    (PIPE / "pi_p_overlay.json").write_text(json.dumps({"baseline": insp_mae(y, hy, u), "best": best, "sweep": rows, "n_add": n_add}, indent=2), encoding="utf-8")
    print("BEST", best, flush=True)


if __name__ == "__main__":
    main()
