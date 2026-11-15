"""Faithful AmbrosM inverse P / PI (writeup, 2021).

PI: search 950 p[0] in decreasing frequency, vectorize p_coef × i_coef × p_star,
accept the first (p0, params) where reconstructed p[1] and p[2] lie on the 950-value
sensor set. Then reconstruct the breath; skip clipped u_in and exploratory
perturbations (steps not on the sensor set).

P: p = p_star - u_in/p_coef; accept if enough non-clipped insp steps match.

Does not use the 75%-hit-fraction gate that killed coverage.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inverse_control"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

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

PC = P_COEF_GRID[P_COEF_GRID > 1e-8]
IC = I_COEF_GRID[np.abs(I_COEF_GRID) > 1e-8]
PS = P_STAR_GRID
T = T_PI_HYPOTHESIS


def _on_uniq(p: np.ndarray, uniq: np.ndarray, tol: float = UNIQUE_P_TOL) -> np.ndarray:
    p = np.asarray(p, np.float64)
    idx = np.searchsorted(uniq, p)
    idx = np.clip(idx, 1, len(uniq) - 1)
    d = np.minimum(np.abs(p - uniq[idx - 1]), np.abs(p - uniq[idx]))
    return d <= tol


def _dt(t: np.ndarray) -> np.ndarray:
    dt = np.diff(t.astype(np.float64), prepend=t[0])
    dt[0] = 0.0
    return dt


def reconstruct_pi(uin, dt, pc, ic, ps, q0, T=T) -> np.ndarray:
    n = len(uin)
    ph = np.empty(n, np.float64)
    q = float(q0)
    ph[0] = ps - (uin[0] - ic * q) / pc
    for i in range(1, n):
        s = dt[i] / (dt[i] + T)
        k = pc + ic * s
        if abs(k) < 1e-12:
            ph[i:] = np.nan
            break
        ph[i] = ps - (uin[i] - ic * (1.0 - s) * q) / k
        q = (1.0 - s) * q + s * (ps - ph[i])
    return ph


def _pi_candidates(uin, dt, uniq, p0_chunk):
    """Return list of (p0, pc, ic, ps, q0) where p[1] and p[2] hit the sensor set."""
    u0, u1, u2 = float(uin[0]), float(uin[1]), float(uin[2])
    s1 = dt[1] / (dt[1] + T)
    s2 = dt[2] / (dt[2] + T)
    p0 = p0_chunk.astype(np.float32)[:, None, None, None]
    pc = PC.astype(np.float32)[None, :, None, None]
    ic = IC.astype(np.float32)[None, None, :, None]
    ps = PS.astype(np.float32)[None, None, None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        q0 = (u0 + pc * (p0 - ps)) / ic
        k1 = pc + ic * s1
        p1 = ps - (u1 - ic * (1.0 - s1) * q0) / k1
        q1 = (1.0 - s1) * q0 + s1 * (ps - p1)
        k2 = pc + ic * s2
        p2 = ps - (u2 - ic * (1.0 - s2) * q1) / k2
    ok = (
        _on_uniq(p1, uniq) & _on_uniq(p2, uniq)
        & np.isfinite(p1) & np.isfinite(p2)
        & (np.abs(k1) > 1e-8) & (np.abs(k2) > 1e-8)
    )
    if not ok.any():
        return []
    idx = np.argwhere(ok)
    out = []
    for ip, ik, ii, is_ in idx:
        p0v = float(p0_chunk[ip])
        pcv, icv, psv = float(PC[ik]), float(IC[ii]), float(PS[is_])
        q0v = float((u0 + pcv * (p0v - psv)) / icv)
        out.append((p0v, pcv, icv, psv, q0v))
        if len(out) >= 40:  # cap coincidences
            break
    return out


def identify_pi(uin, uout, t, uniq, p0_cands) -> dict | None:
    """Search all 950 p0 (freq order, early stop). Accept only high lattice hit-fraction."""
    dt = _dt(t)
    uin64 = uin.astype(np.float64)
    clipped = (uin <= U_CLIP_LO + CLIP_EPS) | (uin >= U_CLIP_HI - CLIP_EPS)
    usable = (uout == 0) & ~clipped
    n_use = int(usable.sum())
    if n_use < 8:
        return None
    best = None
    best_frac = 0.0
    chunk = 32
    for start in range(0, len(p0_cands), chunk):
        cands = _pi_candidates(uin, dt, uniq, p0_cands[start : start + chunk])
        for p0v, pcv, icv, psv, q0v in cands:
            ph = reconstruct_pi(uin64, dt, pcv, icv, psv, q0v)
            hits = int((_on_uniq(ph[usable], uniq)).sum()) if n_use else 0
            frac = hits / n_use
            early = usable.copy()
            early[12:] = False
            n_e = int(early.sum())
            h_e = int(_on_uniq(ph[early], uniq).sum()) if n_e else 0
            if n_e >= 6 and h_e < 8:
                continue
            if frac > best_frac:
                best_frac = frac
                best = {
                    "kind": "PI", "p_coef": pcv, "i_coef": icv, "p_star": psv,
                    "q0": q0v, "p0": p0v, "pred": ph, "hit_frac": frac,
                }
                if frac >= 0.90:
                    return best
        if best_frac >= 0.85:
            break
    if best is None or best_frac < 0.90:
        return None
    return best


def identify_p(uin, uout, uniq) -> dict | None:
    insp = uout == 0
    clipped = (uin <= U_CLIP_LO + CLIP_EPS) | (uin >= U_CLIP_HI - CLIP_EPS)
    m = insp & ~clipped
    if int(m.sum()) < 5:
        return None
    u = uin[m].astype(np.float64)
    # (K, S, T)
    ph = PS[None, :, None] - u[None, None, :] / PC[:, None, None]
    hits = _on_uniq(ph, uniq).sum(axis=-1)
    k, s = np.unravel_index(int(np.argmax(hits)), hits.shape)
    frac = float(hits[k, s]) / max(int(m.sum()), 1)
    if int(hits[k, s]) < 8 or frac < 0.90:
        return None
    pc_v, ps_v = float(PC[k]), float(PS[s])
    pred = np.full(len(uin), np.nan, np.float64)
    pred[~clipped] = ps_v - uin[~clipped] / pc_v
    return {"kind": "P", "p_coef": pc_v, "p_star": ps_v, "pred": pred, "hits": int(hits[k, s])}


def trust_from_pred(uin, uout, ph, uniq) -> np.ndarray:
    clipped = (uin <= U_CLIP_LO + CLIP_EPS) | (uin >= U_CLIP_HI - CLIP_EPS)
    return (uout == 0) & ~clipped & np.isfinite(ph) & on_unique_pressure(ph, uniq)


def identify_breath(uin, uout, t, uniq, p0_cands) -> dict | None:
    rec = identify_pi(uin, uout, t, uniq, p0_cands)
    if rec is None:
        return None
    m = trust_from_pred(uin, uout, rec["pred"], uniq)
    if int(m.sum()) < 3:
        return None
    pred = np.full_like(rec["pred"], np.nan)
    pred[m] = snap_unique(rec["pred"][m], uniq)
    rec["trust"] = m
    rec["pred"] = pred
    rec["n_trust"] = int(m.sum())
    return rec


def p0_by_frequency(train_df: pd.DataFrame, uniq: np.ndarray) -> np.ndarray:
    y0 = train_df.groupby("breath_id", sort=False)["pressure"].first().to_numpy(np.float64)
    y0 = snap_unique(y0, uniq)
    vals, cnt = np.unique(y0, return_counts=True)
    # all 950 unique pressures, frequent first, rest (unseen p0) after
    order = np.argsort(-cnt)
    freq = vals[order]
    rest = uniq[~np.isin(uniq, freq)]
    return np.concatenate([freq, rest])


def _one(args):
    uin, uout, t, uniq, p0_cands = args
    rec = identify_breath(uin, uout, t, uniq, p0_cands)
    if rec is None:
        return None
    return rec["pred"], rec["trust"], rec["kind"]


def run_on_breaths(uin, uout, t, uniq, p0_cands, max_n=None, log_every=200, n_jobs=1):
    n = len(uin) if max_n is None else min(len(uin), max_n)
    pred = np.full((n, 80), np.nan, np.float64)
    trust = np.zeros((n, 80), dtype=bool)
    kinds = []
    t0 = time.time()
    n_ok = 0
    if n_jobs <= 1:
        iterator = range(n)
        results = []
        for i in iterator:
            rec = identify_breath(uin[i], uout[i], t[i], uniq, p0_cands)
            results.append(None if rec is None else (rec["pred"], rec["trust"], rec["kind"]))
            if rec is not None:
                n_ok += 1
                pred[i] = rec["pred"]
                trust[i] = rec["trust"]
                kinds.append(rec["kind"])
            if (i + 1) % log_every == 0 or i == 0:
                ms = 1000 * (time.time() - t0) / (i + 1)
                print(f"  [{i+1}/{n}] id={n_ok} ({100*n_ok/(i+1):.1f}%) {ms:.1f} ms/breath", flush=True)
        return pred, trust, kinds, n_ok
    from joblib import Parallel, delayed
    batch = 400
    for start in range(0, n, batch):
        sl = slice(start, min(start + batch, n))
        recs = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_one)((uin[i], uout[i], t[i], uniq, p0_cands)) for i in range(sl.start, sl.stop)
        )
        for j, rec in enumerate(recs):
            if rec is None:
                continue
            n_ok += 1
            pred[sl.start + j], trust[sl.start + j], knd = rec
            kinds.append(knd)
        done = sl.stop
        ms = 1000 * (time.time() - t0) / done
        print(f"  [{done}/{n}] id={n_ok} ({100*n_ok/done:.1f}%) {ms:.1f} ms/breath", flush=True)
    return pred, trust, kinds, n_ok


def pack_train(df: pd.DataFrame):
    df = df.sort_values(["breath_id", "time_step"])
    n = len(df) // 80
    return {
        "uin": df["u_in"].to_numpy(np.float64).reshape(n, 80),
        "uout": df["u_out"].to_numpy(np.float64).reshape(n, 80),
        "t": df["time_step"].to_numpy(np.float64).reshape(n, 80),
        "y": df["pressure"].to_numpy(np.float64).reshape(n, 80) if "pressure" in df.columns else None,
        "R": df["R"].to_numpy(np.float64).reshape(n, 80),
        "C": df["C"].to_numpy(np.float64).reshape(n, 80),
        "bids": df["breath_id"].to_numpy().reshape(n, 80)[:, 0],
        "id": df["id"].to_numpy().reshape(n, 80) if "id" in df.columns else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-breaths", type=int, default=None)
    ap.add_argument("--tag", default="ambrosm")
    ap.add_argument("--eval-oof", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=1)
    args = ap.parse_args()

    train = pd.read_csv(DATA / "train.csv")
    uniq = np.sort(train["pressure"].unique().astype(np.float64))
    p0 = p0_by_frequency(train, uniq)
    print(f"uniq={len(uniq)} p0_cands={len(p0)} (all unique p, freq-sorted)", flush=True)

    pk = pack_train(train)
    n_run = args.max_breaths or len(pk["bids"])
    with keep_awake():
        pred, trust, kinds, n_ok = run_on_breaths(
            pk["uin"], pk["uout"], pk["t"], uniq, p0, max_n=n_run, n_jobs=args.n_jobs,
        )
    y, u = pk["y"][:n_run], pk["uout"][:n_run]
    insp = u.reshape(-1) == 0
    tf = trust.reshape(-1)
    cov = 100.0 * tf[insp].mean()
    mae_t = float(np.mean(np.abs(y.reshape(-1)[insp & tf] - pred.reshape(-1)[insp & tf]))) if (insp & tf).any() else float("nan")
    rec = {
        "n": n_run, "n_id_breaths": n_ok, "breath_id_rate": n_ok / n_run,
        "insp_coverage_pct": cov, "mae_trusted": mae_t,
        "n_PI": kinds.count("PI"), "n_P": kinds.count("P"),
        "p0_k": int(len(p0)),
    }
    print(json.dumps(rec, indent=2, default=float), flush=True)
    np.savez_compressed(
        PIPE / f"pi_{args.tag}.npz",
        breath_id=pk["bids"][:n_run], pred=pred.astype(np.float32), trust=trust,
        y=y, u_out=u, R=pk["R"][:n_run], C=pk["C"][:n_run],
    )
    (PIPE / f"pi_{args.tag}.json").write_text(json.dumps(rec, indent=2, default=float), encoding="utf-8")

    if args.eval_oof:
        e3 = np.load(RESEARCH / "batch3_oof_e3_aug_uin.npz")
        e1 = np.load(RESEARCH / "batch3_oof_e1_baseline.npz")
        # align by breath_id
        idx = np.argsort(pk["bids"][:n_run])
        bmap = {int(b): i for i, b in enumerate(pk["bids"][:n_run])}
        # assume e3 order may differ
        ens = 0.4771337422498183 * e1["oof"] + 0.5228662577501817 * e3["oof"]
        ens_s = snap_unique(ens.reshape(-1), uniq).reshape(ens.shape)
        order_e = np.array([bmap[int(b)] for b in e3["breath_id"] if int(b) in bmap])
        # simpler: if same n, reindex
        pos = {int(b): i for i, b in enumerate(pk["bids"][:n_run])}
        take = np.array([pos[int(b)] for b in e3["breath_id"]])
        pi_al = pred[take]
        tr_al = trust[take]
        hy = ens_s.copy()
        hy[tr_al] = pi_al[tr_al]
        oof = {
            "ensemble_snap": insp_mae(e3["y"], ens_s, e3["u_out"]),
            "hybrid": insp_mae(e3["y"], hy, e3["u_out"]),
            "pi_cov": float(100.0 * tr_al.reshape(-1)[e3["u_out"].reshape(-1) == 0].mean()),
        }
        print("OOF overlay", oof, flush=True)
        rec["oof"] = oof
        (PIPE / f"pi_{args.tag}.json").write_text(json.dumps(rec, indent=2, default=float), encoding="utf-8")
        update_track("inverse_pi_neural_fallback", {"status": "ambrosm", **rec})
        if oof["hybrid"] < oof["ensemble_snap"] - 0.001:
            update_track("full_hybrid", {"status": "ambrosm_hybrid", **oof})
            np.savez_compressed(PIPE / "hybrid_ambrosm_oof.npz", hybrid=hy, trust=tr_al, y=e3["y"], u_out=e3["u_out"], breath_id=e3["breath_id"])


if __name__ == "__main__":
    main()
