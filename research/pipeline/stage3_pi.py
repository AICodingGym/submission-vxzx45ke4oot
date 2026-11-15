"""Stage 3 — inverse PI overlay on neural predictions.

Train labels used only to build the p[0] candidate list and to *score*.
Identification on val/test is u_in + lattice only.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inverse_control"))

from pi_engine import frequent_p0, identify_breath, overlay_nn
from pipe_common import DATA, PIPE, insp_mae, insp_mae_hard, keep_awake, update_track
from inv_common import UNIQUE_P_TOL


def run_overlay(nn_npz: str, p0_k: int = 80, max_breaths: int | None = None, tag="pi"):
    p = Path(nn_npz)
    if not p.exists():
        p = PIPE / nn_npz
    pack = np.load(p)
    bids = pack["breath_id"]
    nn = pack["pred"].astype(np.float64)
    y = pack["y"].astype(np.float64)
    u = pack["u_out"]
    R, C = pack["R"], pack["C"]
    n = len(bids)
    if max_breaths is not None:
        n = min(n, max_breaths)
        bids, nn, y, u, R, C = bids[:n], nn[:n], y[:n], u[:n], R[:n], C[:n]

    df = pd.read_csv(DATA / "train.csv", usecols=["breath_id", "u_in", "u_out", "time_step", "pressure"])
    uniq = np.sort(df["pressure"].unique().astype(np.float64))
    # p0 list is a train-set statistic (AmbrosM searches observed p[0]s).
    # Do not drop the eval breaths: OOF covers all of train, which emptied the list.
    p0_all = df.groupby("breath_id", sort=False)["pressure"].first().to_numpy()
    p0_cands = frequent_p0(p0_all, uniq, k=p0_k)
    val_set = set(int(b) for b in bids.tolist())

    # u_in / t aligned to val breaths
    sub = df[df["breath_id"].isin(val_set)].sort_values(["breath_id", "time_step"])
    # map breath -> row in nn
    order = {int(b): i for i, b in enumerate(bids)}
    uin = np.zeros((n, 80), np.float64)
    t = np.zeros((n, 80), np.float64)
    uout = np.zeros((n, 80), np.float64)
    for b, g in sub.groupby("breath_id", sort=False):
        i = order.get(int(b))
        if i is None:
            continue
        uin[i] = g["u_in"].to_numpy(np.float64)
        t[i] = g["time_step"].to_numpy(np.float64)
        uout[i] = g["u_out"].to_numpy(np.float64)

    print(f"PI overlay n={n} p0={len(p0_cands)} uniq={len(uniq)}", flush=True)
    # warmup numba
    identify_breath(uin[0], uout[0], t[0], uniq, p0_cands[:8])

    pi_pred = np.full_like(nn, np.nan)
    trust = np.zeros_like(nn, dtype=bool)
    n_id = 0
    t0 = time.time()
    times = []
    with keep_awake():
        for i in range(n):
            t1 = time.time()
            rec = identify_breath(uin[i], uout[i], t[i], uniq, p0_cands)
            times.append(time.time() - t1)
            if rec is not None:
                n_id += 1
                pi_pred[i] = rec["pred"]
                trust[i] = rec["trust"]
            if (i + 1) % 50 == 0 or i == 0:
                med_ms = 1000 * float(np.median(times))
                eta = med_ms * (n - i - 1) / 1000
                print(
                    f"  [{i+1}/{n}] id_ok={n_id} med_ms={med_ms:.1f} eta_s={eta:.0f}",
                    flush=True,
                )
                if i == 19:
                    # cost gate after 20 breaths
                    est_min = med_ms * n / 1000 / 60
                    print(f"  COST GATE: est {est_min:.1f} min for {n} breaths", flush=True)
                    (PIPE / f"{tag}_cost_gate.json").write_text(
                        json.dumps({"med_ms": med_ms, "n": n, "est_min": est_min, "p0_k": p0_k}, indent=2),
                        encoding="utf-8",
                    )
                    if est_min > 90 and max_breaths is None:
                        print("  est > 90 min — stop this scale, reduce p0 or n", flush=True)
                        break

    hybrid = overlay_nn(nn, pi_pred, trust)
    nn_mae = insp_mae(y, nn, u)
    hy_mae = insp_mae(y, hybrid, u)
    insp = u.reshape(-1) == 0
    cov = 100.0 * trust.reshape(-1)[insp].mean()
    trusted = insp & trust.reshape(-1)
    mae_t = (
        float(np.mean(np.abs(y.reshape(-1)[trusted] - pi_pred.reshape(-1)[trusted])))
        if trusted.any() else float("nan")
    )
    rec = {
        "status": "done",
        "n_breaths_attempted": int(i + 1),
        "n_identified": n_id,
        "insp_coverage_pct": float(cov),
        "mae_trusted": mae_t,
        "nn_insp_mae": nn_mae,
        "hybrid_insp_mae": hy_mae,
        "delta_vs_nn": float(nn_mae - hy_mae),
        "hard_nn": insp_mae_hard(y, nn, u, R, C),
        "hard_hybrid": insp_mae_hard(y, hybrid, u, R, C),
        "med_ms_per_breath": float(np.median(times)) * 1000 if times else None,
        "runtime_sec": time.time() - t0,
        "p0_k": p0_k,
        "unique_tol": UNIQUE_P_TOL,
        "nn_source": str(p),
    }
    np.savez_compressed(
        PIPE / f"{tag}_overlay.npz",
        breath_id=bids[: i + 1], nn=nn[: i + 1], hybrid=hybrid[: i + 1],
        pi=pi_pred[: i + 1], trust=trust[: i + 1], y=y[: i + 1], u_out=u[: i + 1],
        R=R[: i + 1], C=C[: i + 1],
    )
    (PIPE / f"{tag}_overlay.json").write_text(json.dumps(rec, indent=2, default=float), encoding="utf-8")
    update_track("inverse_pi_neural_fallback", rec)
    update_track("full_hybrid", {**rec, "note": "PI overlay on this neural; ensemble not applied unless stage 2 ran"})
    print(json.dumps(rec, indent=2, default=float), flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nn-npz", required=True)
    ap.add_argument("--p0-k", type=int, default=80)
    ap.add_argument("--max-breaths", type=int, default=None)
    ap.add_argument("--tag", default="pi")
    args = ap.parse_args()
    run_overlay(args.nn_npz, p0_k=args.p0_k, max_breaths=args.max_breaths, tag=args.tag)


if __name__ == "__main__":
    main()
