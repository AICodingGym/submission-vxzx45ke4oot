"""
Inverse P (tight unique-pressure match) + small-scale inverse PI PoC.

CPU only. Does not touch Batch 3. Uses authorized train.csv only.

Complexity (PI inference, vectorized over p_star):
  n_p0 * n_pcoef * n_icoef per breath  ≈ 80 * 20 * 20 = 32k
  × n_val breaths (200) ≈ 6.4M candidates. Expected wall time: minutes, not hours.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from batch2_common import DATA, breath_split, insp_mae  # noqa: E402
from inv_common import (  # noqa: E402
    I_COEF_GRID,
    P_COEF_GRID,
    P_STAR_GRID,
    T_PI_HYPOTHESIS,
    UNIQUE_P_TOL,
    clip_mask,
    on_unique_pressure,
    snap_unique,
    unique_pressure_residual,
    linear_p_fit,
    p_params_from_linear,
)

OUT = Path(__file__).resolve().parent
SEED = 42
VAL_FRAC = 0.2
MAX_BREATHS_P = 1200
MAX_BREATHS_PI = 400
N_P0_CAND = 80
MIN_HITS_P = 8  # multi-step, not a single lattice coincidence


def breath_pack(df: pd.DataFrame, bids: np.ndarray) -> dict:
    sub = df[df["breath_id"].isin(bids)].sort_values(["breath_id", "time_step"])
    n = len(bids)
    return {
        "y": sub["pressure"].to_numpy(np.float64).reshape(n, 80),
        "uin": sub["u_in"].to_numpy(np.float64).reshape(n, 80),
        "uout": sub["u_out"].to_numpy(np.float64).reshape(n, 80),
        "t": sub["time_step"].to_numpy(np.float64).reshape(n, 80),
        "R": sub["R"].to_numpy(np.float64).reshape(n, 80)[:, 0],
        "C": sub["C"].to_numpy(np.float64).reshape(n, 80)[:, 0],
        "ids": bids,
    }


def identify_p_tight(uin, y, uout, uniq) -> dict:
    insp = uout == 0
    u, p = uin[insp], y[insp]
    m = ~clip_mask(u)
    u, p = u[m], p[m]
    n_use = len(u)
    if n_use < MIN_HITS_P:
        return {"ok": False, "hits": 0, "n_use": n_use, "mae": np.nan, "r2": 0.0}
    a, b, r2 = linear_p_fit(u, p)
    best_hits, best_mae, best_pc, best_ps = -1, np.inf, None, None
    for pc in P_COEF_GRID:
        if pc < 1e-9:
            continue
        p_hat = P_STAR_GRID[:, None] - u[None, :] / pc
        hits = on_unique_pressure(p_hat, uniq).sum(axis=1)
        mae = np.mean(np.abs(p_hat - p[None, :]), axis=1)
        j = int(np.argmax(hits))
        if int(hits[j]) > best_hits or (int(hits[j]) == best_hits and mae[j] < best_mae):
            best_hits, best_mae = int(hits[j]), float(mae[j])
            best_pc, best_ps = float(pc), float(P_STAR_GRID[j])
    return {
        "ok": best_hits >= MIN_HITS_P and best_mae < 0.5,
        "hits": best_hits,
        "n_use": n_use,
        "hit_frac": best_hits / max(n_use, 1),
        "mae": best_mae,
        "p_coef": best_pc,
        "p_star": best_ps,
        "r2": r2,
        "lin_p_coef": p_params_from_linear(a, b)[0],
    }


def reconstruct_p_steps(uin, uout, p_coef, p_star, uniq):
    pred = np.full_like(uin, np.nan)
    trust = np.zeros_like(uin, dtype=bool)
    insp = uout == 0
    for i in np.where(insp)[0]:
        if clip_mask(np.array([uin[i]]))[0]:
            continue
        ph = p_star - uin[i] / p_coef
        if on_unique_pressure(np.array([ph]), uniq)[0]:
            pred[i] = snap_unique(np.array([ph]), uniq)[0]
            trust[i] = True
    return pred, trust


def reconstruct_pi(uin, dt, p_coef, i_coef, p_star, q0, T=T_PI_HYPOTHESIS):
    n = len(uin)
    p_hat = np.empty(n, dtype=np.float64)
    q = float(q0)
    p_hat[0] = p_star - (uin[0] - i_coef * q) / p_coef
    for i in range(1, n):
        s = dt[i] / (dt[i] + T)
        k = p_coef + i_coef * s
        if abs(k) < 1e-12:
            p_hat[i:] = np.nan
            break
        p_hat[i] = p_star - (uin[i] - i_coef * (1.0 - s) * q) / k
        e = p_star - p_hat[i]
        q = (1.0 - s) * q + s * e
    return p_hat


def oracle_pi_one(uin, y, uout, dt, uniq, T=T_PI_HYPOTHESIS) -> dict:
    """Grid (p_coef, i_coef, p_star); q0 from step 0 using labels. Eval only."""
    insp = uout == 0
    n_insp = int(insp.sum())
    best = {"mae": np.inf, "hits": 0}
    for pc in P_COEF_GRID:
        if pc < 1e-9:
            continue
        for ic in I_COEF_GRID:
            if abs(ic) < 1e-9:
                continue
            for ps in P_STAR_GRID:
                q0 = (uin[0] + pc * (y[0] - ps)) / ic
                ph = reconstruct_pi(uin, dt, pc, ic, ps, q0, T)
                m = insp & ~clip_mask(uin) & np.isfinite(ph)
                if m.sum() < MIN_HITS_P:
                    continue
                hits = int(on_unique_pressure(ph[m], uniq).sum())
                mae = float(np.mean(np.abs(ph[m] - y[m])))
                if mae < best["mae"]:
                    best = {
                        "mae": mae, "hits": hits, "n_use": int(m.sum()),
                        "p_coef": float(pc), "i_coef": float(ic), "p_star": float(ps),
                        "q0": float(q0), "hit_frac": hits / max(int(m.sum()), 1),
                    }
    best["n_insp"] = n_insp
    best["ok"] = best["mae"] < 0.15 and best.get("hits", 0) >= MIN_HITS_P
    return best


def infer_pi_uin_only(uin, uout, dt, uniq, p0_cands, T=T_PI_HYPOTHESIS):
    """Test-time: no labels. Require p[1] and p[2] on unique set, then score extra hits."""
    best = None
    best_score = -1
    u0, u1, u2 = uin[0], uin[1], uin[2]
    s1 = dt[1] / (dt[1] + T)
    s2 = dt[2] / (dt[2] + T)
    for p0 in p0_cands:
        for pc in P_COEF_GRID:
            if pc < 1e-9:
                continue
            for ic in I_COEF_GRID:
                if abs(ic) < 1e-9:
                    continue
                # vectorize p_star
                ps = P_STAR_GRID
                q0 = (u0 + pc * (p0 - ps)) / ic
                k1 = pc + ic * s1
                p1 = ps - (u1 - ic * (1.0 - s1) * q0) / k1
                e1 = ps - p1
                q1 = (1.0 - s1) * q0 + s1 * e1
                k2 = pc + ic * s2
                p2 = ps - (u2 - ic * (1.0 - s2) * q1) / k2
                ok1 = on_unique_pressure(p1, uniq)
                ok2 = on_unique_pressure(p2, uniq)
                both = ok1 & ok2
                if not both.any():
                    continue
                for j in np.where(both)[0]:
                    # score on first 12 steps, full reconstruct only for current best
                    n_sc = min(12, len(uin))
                    ph12 = reconstruct_pi(uin[:n_sc], dt[:n_sc], pc, ic, float(ps[j]), float(q0[j]), T)
                    insp12 = uout[:n_sc] == 0
                    m12 = insp12 & ~clip_mask(uin[:n_sc]) & np.isfinite(ph12)
                    hits = int(on_unique_pressure(ph12[m12], uniq).sum()) if m12.any() else 0
                    if hits > best_score:
                        best_score = hits
                        ph = reconstruct_pi(uin, dt, pc, ic, float(ps[j]), float(q0[j]), T)
                        insp = uout == 0
                        m = insp & ~clip_mask(uin) & np.isfinite(ph)
                        best = {
                            "p_coef": float(pc), "i_coef": float(ic),
                            "p_star": float(ps[j]), "q0": float(q0[j]),
                            "p0": float(p0), "hits": hits, "n_use": int(m.sum()),
                            "pred": ph,
                        }
    return best


def coverage_metrics(y, uout, pred, trust) -> dict:
    yf, uf, pf, tf = y.reshape(-1), uout.reshape(-1), pred.reshape(-1), trust.reshape(-1)
    insp = uf == 0
    ok = insp & tf & np.isfinite(pf)
    n_insp = int(insp.sum())
    n_ok = int(ok.sum())
    mae = float(np.mean(np.abs(yf[ok] - pf[ok]))) if n_ok else float("nan")
    mean_insp = float(np.mean(yf[insp]))
    hybrid = np.where(ok, pf, mean_insp)
    return {
        "insp_coverage_pct": 100.0 * n_ok / max(n_insp, 1),
        "n_trusted": n_ok,
        "n_insp": n_insp,
        "mae_trusted": mae,
        "hybrid_mean_fallback_mae": insp_mae(yf, hybrid, uf),
    }


def main():
    t0 = time.time()
    lines = ["INVERSE P (tight unique-P) + PI PoC", f"unique_tol={UNIQUE_P_TOL} T={T_PI_HYPOTHESIS}"]
    print(lines[0], flush=True)

    df = pd.read_csv(DATA / "train.csv")
    uniq = np.sort(df["pressure"].unique().astype(np.float64))
    lines.append(f"unique pressures in train: {len(uniq)}")

    rng = np.random.default_rng(SEED)
    all_ids = df["breath_id"].unique()

    # ----- P redo, tight unique match -----
    bids_p = rng.choice(all_ids, size=min(MAX_BREATHS_P, len(all_ids)), replace=False)
    tr_set, va_set = breath_split(bids_p, VAL_FRAC, SEED)
    tr_ids, va_ids = np.array(sorted(tr_set)), np.array(sorted(va_set))
    tr, va = breath_pack(df, tr_ids), breath_pack(df, va_ids)

    tr_p = [identify_p_tight(tr["uin"][i], tr["y"][i], tr["uout"][i], uniq) for i in range(len(tr_ids))]
    n_ok = sum(1 for s in tr_p if s["ok"])
    lines.append(
        f"P-tight TRAIN: {n_ok}/{len(tr_ids)} breaths ok "
        f"(≥{MIN_HITS_P} unique hits and MAE<0.5); "
        f"mean r2={np.nanmean([s['r2'] for s in tr_p]):.3f}"
    )

    pred_u = np.full_like(va["y"], np.nan)
    trust_u = np.zeros_like(va["y"], dtype=bool)
    pred_o = np.full_like(va["y"], np.nan)
    trust_o = np.zeros_like(va["y"], dtype=bool)
    for i in range(len(va_ids)):
        st = identify_p_tight(va["uin"][i], va["y"][i], va["uout"][i], uniq)
        if st.get("p_coef"):
            po, to_ = reconstruct_p_steps(va["uin"][i], va["uout"][i], st["p_coef"], st["p_star"], uniq)
            pred_o[i], trust_o[i] = po, to_
        # u_in-only: pick (pc, ps) maximizing unique hits (no labels)
        uin, uout = va["uin"][i], va["uout"][i]
        insp = uout == 0
        u = uin[insp]
        u = u[~clip_mask(u)]
        if len(u) < MIN_HITS_P:
            continue
        bh, bpc, bps = -1, None, None
        for pc in P_COEF_GRID:
            if pc < 1e-9:
                continue
            ph = P_STAR_GRID[:, None] - u[None, :] / pc
            hits = on_unique_pressure(ph, uniq).sum(axis=1)
            j = int(np.argmax(hits))
            if int(hits[j]) > bh:
                bh, bpc, bps = int(hits[j]), float(pc), float(P_STAR_GRID[j])
        if bpc is None or bh < MIN_HITS_P:
            continue
        pu, tu = reconstruct_p_steps(uin, uout, bpc, bps, uniq)
        pred_u[i], trust_u[i] = pu, tu

    p_val_u = coverage_metrics(va["y"], va["uout"], pred_u, trust_u)
    p_val_o = coverage_metrics(va["y"], va["uout"], pred_o, trust_o)
    lines.append(f"P-tight VAL oracle: {p_val_o}")
    lines.append(f"P-tight VAL u_in-only: {p_val_u}")

    # ----- PI small subset -----
    bids_pi = rng.choice(all_ids, size=min(MAX_BREATHS_PI, len(all_ids)), replace=False)
    trs, vas = breath_split(bids_pi, VAL_FRAC, SEED)
    tri, vai = np.array(sorted(trs)), np.array(sorted(vas))
    ptr, pva = breath_pack(df, tri), breath_pack(df, vai)
    p0_freq = pd.Series(np.round(ptr["y"][:, 0], 6)).value_counts()
    p0_cands = np.array(p0_freq.index[:N_P0_CAND], dtype=np.float64)
    # snap p0 candidates onto unique set
    p0_cands = np.unique(snap_unique(p0_cands, uniq))

    lines.append(
        f"PI complexity: n_p0={len(p0_cands)} × {len(P_COEF_GRID)} × {len(I_COEF_GRID)} "
        f"× {len(P_STAR_GRID)} /breath; val breaths={len(vai)} "
        f"(vectorized over p_star → {len(p0_cands)*len(P_COEF_GRID)*len(I_COEF_GRID)} loops/breath)"
    )

    oracle_ok = 0
    oracle_maes = []
    for i in range(len(tri)):
        dt = np.diff(ptr["t"][i], prepend=ptr["t"][i, 0])
        dt[0] = 0.0
        st = oracle_pi_one(ptr["uin"][i], ptr["y"][i], ptr["uout"][i], dt, uniq)
        if st.get("ok"):
            oracle_ok += 1
            oracle_maes.append(st["mae"])
        if (i + 1) % 50 == 0:
            print(f"  PI oracle train {i+1}/{len(tri)} ok={oracle_ok}", flush=True)
    lines.append(
        f"PI oracle TRAIN: {oracle_ok}/{len(tri)} breaths MAE<0.15; "
        f"mean MAE among ok={np.mean(oracle_maes) if oracle_maes else np.nan:.4f}"
    )

    pred_pi_o = np.full_like(pva["y"], np.nan)
    trust_pi_o = np.zeros_like(pva["y"], dtype=bool)
    pred_pi_u = np.full_like(pva["y"], np.nan)
    trust_pi_u = np.zeros_like(pva["y"], dtype=bool)
    infer_hits = 0
    t_pi = time.time()
    for i in range(len(vai)):
        dt = np.diff(pva["t"][i], prepend=pva["t"][i, 0])
        dt[0] = 0.0
        ost = oracle_pi_one(pva["uin"][i], pva["y"][i], pva["uout"][i], dt, uniq)
        if ost.get("p_coef"):
            ph = reconstruct_pi(
                pva["uin"][i], dt, ost["p_coef"], ost["i_coef"], ost["p_star"], ost["q0"]
            )
            insp = pva["uout"][i] == 0
            m = insp & ~clip_mask(pva["uin"][i]) & np.isfinite(ph) & on_unique_pressure(ph, uniq)
            pred_pi_o[i] = np.where(m, snap_unique(ph, uniq), np.nan)
            trust_pi_o[i] = m
        inf = infer_pi_uin_only(pva["uin"][i], pva["uout"][i], dt, uniq, p0_cands)
        if inf and inf["hits"] >= MIN_HITS_P:
            infer_hits += 1
            ph = inf["pred"]
            insp = pva["uout"][i] == 0
            m = insp & ~clip_mask(pva["uin"][i]) & np.isfinite(ph) & on_unique_pressure(ph, uniq)
            pred_pi_u[i] = np.where(m, snap_unique(ph, uniq), np.nan)
            trust_pi_u[i] = m
        if (i + 1) % 20 == 0:
            print(f"  PI val {i+1}/{len(vai)} infer_ok={infer_hits}", flush=True)
    pi_val_o = coverage_metrics(pva["y"], pva["uout"], pred_pi_o, trust_pi_o)
    pi_val_u = coverage_metrics(pva["y"], pva["uout"], pred_pi_u, trust_pi_u)
    lines.append(f"PI VAL oracle: {pi_val_o}")
    lines.append(f"PI VAL u_in-only ({infer_hits}/{len(vai)} breaths): {pi_val_u}")
    lines.append(f"PI val loop sec={time.time() - t_pi:.1f}")

    elapsed = time.time() - t0
    results = {
        "unique_pressures": int(len(uniq)),
        "unique_tol": UNIQUE_P_TOL,
        "p_tight": {
            "n_breaths": int(len(bids_p)),
            "train_ok": n_ok,
            "train_n": len(tr_ids),
            "val_oracle": p_val_o,
            "val_uin_only": p_val_u,
        },
        "pi": {
            "T": T_PI_HYPOTHESIS,
            "n_breaths": int(len(bids_pi)),
            "train_oracle_ok": oracle_ok,
            "train_n": len(tri),
            "train_mae_ok": float(np.mean(oracle_maes)) if oracle_maes else None,
            "val_oracle": pi_val_o,
            "val_uin_only": pi_val_u,
            "val_infer_breaths": infer_hits,
        },
        "neural_ref": {"batch2_oof": 0.2506, "batch2_holdout": 0.2389, "batch3_screen_best": 0.2616},
        "runtime_sec": elapsed,
        "note": "Subset PoC; not comparable to GroupKFold OOF. Mean fallback is not the NN.",
    }
    (OUT / "pi_poc_results.json").write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    (OUT / "pi_poc_report.txt").write_text("\n".join(lines), encoding="utf-8")
    log = OUT / "experiment_log.md"
    if not log.exists():
        log.write_text("# Inverse control experiment log\n", encoding="utf-8")
    with log.open("a", encoding="utf-8") as f:
        f.write(
            f"\n## PoC-P2-tight + PI ({time.strftime('%Y-%m-%d %H:%M')})\n"
            f"- Unique-P tol={UNIQUE_P_TOL} (fixed arithmetic-lattice false positives)\n"
            f"- P val oracle cov={p_val_o['insp_coverage_pct']:.1f}% MAE={p_val_o['mae_trusted']}\n"
            f"- PI val oracle cov={pi_val_o['insp_coverage_pct']:.1f}% MAE={pi_val_o['mae_trusted']}\n"
            f"- PI val u_in-only cov={pi_val_u['insp_coverage_pct']:.1f}% MAE={pi_val_u['mae_trusted']}\n"
            f"- Runtime {elapsed:.1f}s\n"
        )
    print("\n".join(lines))
    print(f"Runtime {elapsed:.1f}s")
    return results


if __name__ == "__main__":
    main()
