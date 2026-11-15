"""Inverse PI that treats q0 as a free initial state.

u[0] is clipped on ~35% of breaths, so q0 cannot be taken from p[0] via u[0].
Instead: for each (p_coef, i_coef, p_star) and each of 950 lattice values of p[1],
solve q0, require p[2] and p[3] on the sensor set, then reconstruct.

Train labels are used only to score held-out breaths.
"""
from __future__ import annotations

import argparse
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
    T_PI_HYPOTHESIS,
    U_CLIP_HI,
    U_CLIP_LO,
    UNIQUE_P_TOL,
    on_unique_pressure,
    snap_unique,
)
from pipe_common import DATA, PIPE, RESEARCH, insp_mae, keep_awake, update_track

PC = P_COEF_GRID[P_COEF_GRID > 1e-8].astype(np.float64)
IC = I_COEF_GRID[np.abs(I_COEF_GRID) > 1e-8].astype(np.float64)
PS = P_STAR_GRID.astype(np.float64)
T = T_PI_HYPOTHESIS


@njit(cache=True)
def _near(uniq, val, tol):
    lo = 0
    hi = uniq.shape[0]
    while lo < hi:
        mid = (lo + hi) // 2
        if uniq[mid] < val:
            lo = mid + 1
        else:
            hi = mid
    best = 1e9
    for j in (lo - 1, lo, lo + 1):
        if 0 <= j < uniq.shape[0]:
            d = abs(uniq[j] - val)
            if d < best:
                best = d
    return best <= tol


@njit(cache=True)
def _reconstruct(uin, dt, pc, ic, ps, q0, T):
    n = uin.shape[0]
    ph = np.empty(n, np.float64)
    q = q0
    k0 = pc
    if abs(k0) < 1e-12:
        ph[:] = np.nan
        return ph
    ph[0] = ps - (uin[0] - ic * q) / k0
    for i in range(1, n):
        s = dt[i] / (dt[i] + T)
        k = pc + ic * s
        if abs(k) < 1e-12:
            for j in range(i, n):
                ph[j] = np.nan
            break
        ph[i] = ps - (uin[i] - ic * (1.0 - s) * q) / k
        q = (1.0 - s) * q + s * (ps - ph[i])
    return ph


@njit(cache=True)
def _count_hits(ph, uin, uout, uniq, tol, clip_lo, clip_hi, clip_eps, t_lo, t_hi):
    hits = 0
    n_use = 0
    for t in range(t_lo, t_hi):
        if uout[t] != 0.0:
            continue
        if uin[t] <= clip_lo + clip_eps or uin[t] >= clip_hi - clip_eps:
            continue
        if not np.isfinite(ph[t]):
            continue
        n_use += 1
        if _near(uniq, ph[t], tol):
            hits += 1
    return hits, n_use


@njit(cache=True)
def identify_q0(
    uin, uout, dt, uniq, pc_g, ic_g, ps_g, T, tol,
    clip_lo, clip_hi, clip_eps, min_early, min_frac,
):
    """Return best pc, ic, ps, q0, hits, n_use, frac. hits=-1 if none."""
    n = uin.shape[0]
    if n < 8:
        return 0.0, 0.0, 0.0, 0.0, -1, 0, 0.0
    # need u[1] invertible
    if uout[1] != 0.0 or uin[1] <= clip_lo + clip_eps or uin[1] >= clip_hi - clip_eps:
        return 0.0, 0.0, 0.0, 0.0, -1, 0, 0.0
    s1 = dt[1] / (dt[1] + T)
    s2 = dt[2] / (dt[2] + T)
    s3 = dt[3] / (dt[3] + T)
    u1, u2, u3 = uin[1], uin[2], uin[3]
    clip2 = (uout[2] != 0.0) or (uin[2] <= clip_lo + clip_eps) or (uin[2] >= clip_hi - clip_eps)
    clip3 = (uout[3] != 0.0) or (uin[3] <= clip_lo + clip_eps) or (uin[3] >= clip_hi - clip_eps)
    best_hits = -1
    best_frac = -1.0
    best_pc = 0.0
    best_ic = 0.0
    best_ps = 0.0
    best_q0 = 0.0
    best_use = 0
    nlim = 12 if n >= 12 else n
    for ipc in range(pc_g.shape[0]):
        pc = pc_g[ipc]
        for iic in range(ic_g.shape[0]):
            ic = ic_g[iic]
            if abs(ic) < 1e-12:
                continue
            for ips in range(ps_g.shape[0]):
                ps = ps_g[ips]
                k1 = pc + ic * s1
                if abs(k1) < 1e-12:
                    continue
                den = ic * (1.0 - s1)
                if abs(den) < 1e-12:
                    continue
                k2 = pc + ic * s2
                k3 = pc + ic * s3
                for ip in range(uniq.shape[0]):
                    p1 = uniq[ip]
                    q0 = (u1 - k1 * (ps - p1)) / den
                    q1 = (1.0 - s1) * q0 + s1 * (ps - p1)
                    if not clip2:
                        if abs(k2) < 1e-12:
                            continue
                        p2 = ps - (u2 - ic * (1.0 - s2) * q1) / k2
                        if not _near(uniq, p2, tol):
                            continue
                        q2 = (1.0 - s2) * q1 + s2 * (ps - p2)
                    else:
                        q2 = q1
                    if not clip3:
                        if abs(k3) < 1e-12:
                            continue
                        p3 = ps - (u3 - ic * (1.0 - s3) * q2) / k3
                        if not _near(uniq, p3, tol):
                            continue
                    ph12 = _reconstruct(uin[:nlim], dt[:nlim], pc, ic, ps, q0, T)
                    h12, n12 = _count_hits(
                        ph12, uin, uout, uniq, tol, clip_lo, clip_hi, clip_eps, 0, nlim
                    )
                    if n12 < 6 or h12 < min_early:
                        continue
                    ph = _reconstruct(uin, dt, pc, ic, ps, q0, T)
                    hits, n_use = _count_hits(
                        ph, uin, uout, uniq, tol, clip_lo, clip_hi, clip_eps, 0, n
                    )
                    if n_use < 8:
                        continue
                    frac = hits / n_use
                    if frac < min_frac:
                        continue
                    if hits > best_hits or (hits == best_hits and frac > best_frac):
                        best_hits = hits
                        best_frac = frac
                        best_pc = pc
                        best_ic = ic
                        best_ps = ps
                        best_q0 = q0
                        best_use = n_use
                        if frac >= 0.99 and hits >= 20:
                            return best_pc, best_ic, best_ps, best_q0, best_hits, best_use, best_frac
    return best_pc, best_ic, best_ps, best_q0, best_hits, best_use, best_frac


def trust_mask(uin, uout, ph, uniq):
    clipped = (uin <= U_CLIP_LO + CLIP_EPS) | (uin >= U_CLIP_HI - CLIP_EPS)
    return (uout == 0) & ~clipped & np.isfinite(ph) & on_unique_pressure(ph, uniq)


def identify_breath(uin, uout, t, uniq, min_early=8, min_frac=0.55):
    dt = np.diff(t.astype(np.float64), prepend=t[0])
    dt[0] = 0.0
    pc, ic, ps, q0, hits, n_use, frac = identify_q0(
        uin.astype(np.float64), uout.astype(np.float64), dt, uniq.astype(np.float64),
        PC, IC, PS, T, UNIQUE_P_TOL, U_CLIP_LO, U_CLIP_HI, CLIP_EPS, min_early, min_frac,
    )
    if hits < 8:
        return None
    ph = _reconstruct(uin.astype(np.float64), dt, pc, ic, ps, q0, T)
    m = trust_mask(uin, uout, ph, uniq)
    if int(m.sum()) < 3:
        return None
    pred = np.full_like(ph, np.nan)
    pred[m] = snap_unique(ph[m], uniq)
    return {
        "p_coef": float(pc), "i_coef": float(ic), "p_star": float(ps),
        "q0": float(q0), "hits": int(hits), "frac": float(frac),
        "pred": pred, "trust": m, "ph_raw": ph,
    }


def _one(args):
    uin, uout, t, uniq, min_early, min_frac = args
    rec = identify_breath(uin, uout, t, uniq, min_early, min_frac)
    if rec is None:
        return None
    return rec["pred"], rec["trust"], rec["frac"]


def pack_train(df, n=None):
    df = df.sort_values(["breath_id", "time_step"])
    nb = len(df) // 80
    if n is not None:
        nb = min(nb, n)
        df = df.iloc[: nb * 80]
    return {
        "uin": df["u_in"].to_numpy(np.float64).reshape(nb, 80),
        "uout": df["u_out"].to_numpy(np.float64).reshape(nb, 80),
        "t": df["time_step"].to_numpy(np.float64).reshape(nb, 80),
        "y": df["pressure"].to_numpy(np.float64).reshape(nb, 80),
        "R": df["R"].to_numpy(np.float64).reshape(nb, 80),
        "C": df["C"].to_numpy(np.float64).reshape(nb, 80),
        "bids": df["breath_id"].to_numpy().reshape(nb, 80)[:, 0],
    }


def run(uin, uout, t, uniq, n_jobs=1, min_early=8, min_frac=0.55, log_every=50):
    n = len(uin)
    pred = np.full((n, 80), np.nan, np.float64)
    trust = np.zeros((n, 80), dtype=bool)
    fracs = []
    t0 = time.time()
    n_ok = 0
    if n_jobs <= 1:
        for i in range(n):
            rec = identify_breath(uin[i], uout[i], t[i], uniq, min_early, min_frac)
            if rec is not None:
                n_ok += 1
                pred[i] = rec["pred"]
                trust[i] = rec["trust"]
                fracs.append(rec["frac"])
            if (i + 1) % log_every == 0 or i == 0:
                ms = 1000 * (time.time() - t0) / (i + 1)
                print(f"  [{i+1}/{n}] id={n_ok} ({100*n_ok/(i+1):.1f}%) {ms:.1f} ms/breath", flush=True)
        return pred, trust, n_ok, fracs
    from joblib import Parallel, delayed
    batch = 500
    for start in range(0, n, batch):
        sl = slice(start, min(start + batch, n))
        recs = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_one)((uin[i], uout[i], t[i], uniq, min_early, min_frac))
            for i in range(sl.start, sl.stop)
        )
        for j, rec in enumerate(recs):
            if rec is None:
                continue
            n_ok += 1
            pred[sl.start + j], trust[sl.start + j], fr = rec
            fracs.append(fr)
        done = sl.stop
        ms = 1000 * (time.time() - t0) / done
        print(f"  [{done}/{n}] id={n_ok} ({100*n_ok/done:.1f}%) {ms:.1f} ms/breath", flush=True)
    return pred, trust, n_ok, fracs


def overlay_eval(pred, trust, y, uout, nn=None):
    insp = uout.reshape(-1) == 0
    tf = trust.reshape(-1)
    yf = y.reshape(-1)
    pf = pred.reshape(-1)
    cov = float(100.0 * tf[insp].mean())
    mae_t = float(np.mean(np.abs(yf[insp & tf] - pf[insp & tf]))) if (insp & tf).any() else float("nan")
    rec = {"insp_coverage_pct": cov, "mae_trusted": mae_t, "n_trusted": int((insp & tf).sum())}
    if nn is not None:
        hy = nn.copy()
        hy[trust] = pred[trust]
        rec["nn_mae"] = insp_mae(y, nn, uout)
        rec["hybrid_mae"] = insp_mae(y, hy, uout)
        rec["delta"] = rec["nn_mae"] - rec["hybrid_mae"]
        # NN-consistency gate
        d = np.abs(pred - nn)
        for tau in (0.15, 0.25, 0.35, 0.50, 0.70, 1.0):
            g = trust & np.isfinite(pred) & (d < tau)
            hy2 = nn.copy()
            hy2[g] = pred[g]
            rec[f"hybrid_tau{tau}"] = insp_mae(y, hy2, uout)
            rec[f"cov_tau{tau}"] = float(100.0 * g.reshape(-1)[insp].mean())
            rec[f"mae_tau{tau}"] = (
                float(np.mean(np.abs(yf[insp & g.reshape(-1)] - pf[insp & g.reshape(-1)])))
                if (insp & g.reshape(-1)).any() else float("nan")
            )
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-breaths", type=int, default=200)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--min-frac", type=float, default=0.55)
    ap.add_argument("--min-early", type=int, default=8)
    ap.add_argument("--tag", default="q0")
    ap.add_argument("--eval-oof", action="store_true")
    args = ap.parse_args()

    train = pd.read_csv(DATA / "train.csv")
    uniq = np.sort(train["pressure"].unique().astype(np.float64))
    pk = pack_train(train, args.max_breaths)
    print(f"n={len(pk['bids'])} uniq={len(uniq)} min_frac={args.min_frac} min_early={args.min_early}", flush=True)
    # warmup
    identify_breath(pk["uin"][0], pk["uout"][0], pk["t"][0], uniq, args.min_early, args.min_frac)

    with keep_awake():
        pred, trust, n_ok, fracs = run(
            pk["uin"], pk["uout"], pk["t"], uniq,
            n_jobs=args.n_jobs, min_early=args.min_early, min_frac=args.min_frac,
        )

    nn = None
    if args.eval_oof:
        e3 = np.load(RESEARCH / "batch3_oof_e3_aug_uin.npz")
        e1 = np.load(RESEARCH / "batch3_oof_e1_baseline.npz")
        pos = {int(b): i for i, b in enumerate(e3["breath_id"])}
        take = np.array([pos[int(b)] for b in pk["bids"]])
        nn = 0.4771337422498183 * e1["oof"][take] + 0.5228662577501817 * e3["oof"][take]
        nn = snap_unique(nn.reshape(-1), uniq).reshape(nn.shape)

    metrics = overlay_eval(pred, trust, pk["y"], pk["uout"], nn)
    rec = {
        "n": len(pk["bids"]), "n_id": n_ok, "id_rate": n_ok / len(pk["bids"]),
        "min_frac": args.min_frac, "min_early": args.min_early,
        "median_frac": float(np.median(fracs)) if fracs else None,
        **metrics,
    }
    print(json.dumps(rec, indent=2, default=float), flush=True)
    np.savez_compressed(
        PIPE / f"pi_{args.tag}.npz",
        breath_id=pk["bids"], pred=pred.astype(np.float32), trust=trust,
        y=pk["y"], u_out=pk["uout"],
    )
    (PIPE / f"pi_{args.tag}.json").write_text(json.dumps(rec, indent=2, default=float), encoding="utf-8")
    update_track("inverse_pi_neural_fallback", {"status": "q0_search", **rec})


if __name__ == "__main__":
    main()
