"""Full-train q0 inverse-PI, OOF tau selection, test overlay, write predictions.csv.

Does not touch the GPU. Resume-safe via pi_q0_train.npz / pi_q0_test.npz.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inverse_control"))

from inv_common import snap_unique
from pi_q0_search import identify_breath, pack_train, run, trust_mask
from pipe_common import DATA, PIPE, RESEARCH, insp_mae, insp_mae_hard, keep_awake, update_track

W_E1 = 0.4771337422498183
W_E3 = 0.5228662577501817
TAUS = (0.5, 0.7, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0)


def load_ensemble_oof():
    e3 = np.load(RESEARCH / "batch3_oof_e3_aug_uin.npz")
    e1 = np.load(RESEARCH / "batch3_oof_e1_baseline.npz")
    uniq = np.sort(pd.read_csv(DATA / "train.csv", usecols=["pressure"])["pressure"].unique().astype(np.float64))
    ens = W_E1 * e1["oof"] + W_E3 * e3["oof"]
    ens_s = snap_unique(ens.reshape(-1), uniq).reshape(ens.shape)
    return e3, e1, ens, ens_s, uniq


def apply_tau(nn, pred, trust, tau):
    g = trust & np.isfinite(pred) & (np.abs(pred - nn) < tau)
    hy = nn.copy()
    hy[g] = pred[g]
    return hy, g


def sweep_tau(y, u, nn, pred, trust, R=None, C=None):
    rows = []
    for tau in TAUS:
        hy, g = apply_tau(nn, pred, trust, tau)
        insp = u.reshape(-1) == 0
        gf = g.reshape(-1)
        mae_t = (
            float(np.mean(np.abs(y.reshape(-1)[insp & gf] - pred.reshape(-1)[insp & gf])))
            if (insp & gf).any() else float("nan")
        )
        rec = {
            "tau": tau,
            "hybrid": insp_mae(y, hy, u),
            "cov": float(100.0 * gf[insp].mean()),
            "mae_trusted": mae_t,
        }
        if R is not None:
            rec["hard"] = insp_mae_hard(y, hy, u, R, C)
        rows.append(rec)
        print(f"  tau={tau:.1f} hybrid={rec['hybrid']:.5f} cov={rec['cov']:.2f}% mae_t={mae_t:.5f}", flush=True)
    best = min(rows, key=lambda r: r["hybrid"])
    return rows, best


def identify_all(pk, uniq, dest, n_jobs, min_frac, min_early):
    n = len(pk["bids"])
    if dest.exists():
        old = np.load(dest, allow_pickle=True)
        done = ("complete" in old.files and int(np.asarray(old["complete"]).reshape(-1)[0]) == 1
                and len(old["breath_id"]) == n)
        if done:
            print(f"resume {dest} complete n={n}", flush=True)
            return old["pred"].astype(np.float64), old["trust"].astype(bool)
    print(f"ID n={n} -> {dest.name}", flush=True)
    pred, trust, n_ok, fracs = run(
        pk["uin"], pk["uout"], pk["t"], uniq,
        n_jobs=n_jobs, min_early=min_early, min_frac=min_frac, log_every=2000,
    )
    np.savez_compressed(
        dest,
        breath_id=pk["bids"], pred=pred.astype(np.float32), trust=trust,
        complete=np.array([1]), n_ok=np.array([n_ok]),
    )
    print(f"saved {dest} id={n_ok}/{n}", flush=True)
    return pred, trust


def pack_test(df, bids_order):
    sub = df.sort_values(["breath_id", "time_step"])
    n = len(bids_order)
    order = {int(b): i for i, b in enumerate(bids_order)}
    uin = np.zeros((n, 80), np.float64)
    t = np.zeros((n, 80), np.float64)
    uout = np.zeros((n, 80), np.float64)
    ids = np.zeros((n, 80), np.int64)
    for b, g in sub.groupby("breath_id", sort=False):
        i = order.get(int(b))
        if i is None:
            continue
        uin[i] = g["u_in"].to_numpy(np.float64)
        t[i] = g["time_step"].to_numpy(np.float64)
        uout[i] = g["u_out"].to_numpy(np.float64)
        ids[i] = g["id"].to_numpy(np.int64)
    return {"uin": uin, "uout": uout, "t": t, "id": ids, "bids": np.asarray(bids_order)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=6)
    ap.add_argument("--min-frac", type=float, default=0.55)
    ap.add_argument("--min-early", type=int, default=8)
    ap.add_argument("--skip-train-id", action="store_true")
    args = ap.parse_args()

    fallback = PIPE / "predictions_lb020624.csv"
    src_pred = RESEARCH.parent / "predictions.csv"
    if src_pred.exists() and not fallback.exists():
        shutil.copy2(src_pred, fallback)
        print(f"preserved fallback {fallback}", flush=True)

    train = pd.read_csv(DATA / "train.csv")
    uniq = np.sort(train["pressure"].unique().astype(np.float64))
    e3, e1, ens, ens_s, uniq = load_ensemble_oof()
    # warmup JIT
    pk0 = pack_train(train, 2)
    identify_breath(pk0["uin"][0], pk0["uout"][0], pk0["t"][0], uniq, args.min_early, args.min_frac)

    with keep_awake():
        pk = pack_train(train, None)
        train_npz = PIPE / "pi_q0_train.npz"
        pred_tr, trust_tr = identify_all(pk, uniq, train_npz, args.n_jobs, args.min_frac, args.min_early)

        pos = {int(b): i for i, b in enumerate(pk["bids"])}
        take = np.array([pos[int(b)] for b in e3["breath_id"]])
        pred_al = pred_tr[take]
        tr_al = trust_tr[take]
        y, u, R, C = e3["y"], e3["u_out"], e3["R"], e3["C"]

        print("OOF ensemble_snap", insp_mae(y, ens_s, u), flush=True)
        print("tau sweep on ensemble_snap + q0 PI", flush=True)
        rows, best = sweep_tau(y, u, ens_s, pred_al, tr_al, R, C)
        hy, g = apply_tau(ens_s, pred_al, tr_al, best["tau"])
        oof = {
            "ensemble_snap": insp_mae(y, ens_s, u),
            "best_tau": best["tau"],
            "best_hybrid": best["hybrid"],
            "best_cov": best["cov"],
            "best_mae_trusted": best["mae_trusted"],
            "sweep": rows,
            "n_id_train": int(trust_tr.any(axis=1).sum()),
            "min_frac": args.min_frac,
        }
        print("BEST", json.dumps({k: oof[k] for k in ("ensemble_snap", "best_tau", "best_hybrid", "best_cov")}), flush=True)
        (PIPE / "pi_q0_oof.json").write_text(json.dumps(oof, indent=2, default=float), encoding="utf-8")
        np.savez_compressed(PIPE / "hybrid_q0_oof.npz", hybrid=hy, trust=g, y=y, u_out=u, breath_id=e3["breath_id"])
        update_track("inverse_pi_neural_fallback", {"status": "q0_full", **{k: oof[k] for k in oof if k != "sweep"}})
        update_track("full_hybrid", {"status": "q0_tau", "protocol": "GroupKFold OOF + q0 PI", **best, "ensemble_snap": oof["ensemble_snap"]})

        test = pd.read_csv(DATA / "test.csv")
        tpack = np.load(PIPE / "test_pred_e3_folds.npz")
        test_nn = tpack["inv_mae_weighted_snap"].astype(np.float64)
        test_bids = tpack["breath_id"]
        pk_te = pack_test(test, test_bids)
        test_npz = PIPE / "pi_q0_test.npz"
        pred_te, trust_te = identify_all(pk_te, uniq, test_npz, args.n_jobs, args.min_frac, args.min_early)
        hy_te, g_te = apply_tau(test_nn, pred_te, trust_te, best["tau"])
        out = pd.DataFrame({"id": pk_te["id"].reshape(-1), "pressure": hy_te.reshape(-1)})
        out = out.sort_values("id")
        dest = RESEARCH.parent / "predictions.csv"
        out.to_csv(dest, index=False)
        meta = {
            "file": str(dest),
            "n": len(out),
            "model": f"e1+e3 snap + q0 inverse-PI tau={best['tau']}",
            "oof_ensemble_snap": oof["ensemble_snap"],
            "oof_hybrid": best["hybrid"],
            "oof_cov": best["cov"],
            "tau": best["tau"],
            "test_id_breaths": int(trust_te.any(axis=1).sum()),
            "test_overlay_steps": int(g_te.sum()),
            "pressure_mean": float(out["pressure"].mean()),
        }
        (PIPE / "submit_q0_meta.json").write_text(json.dumps(meta, indent=2, default=float), encoding="utf-8")
        print(json.dumps(meta, indent=2, default=float), flush=True)
        print(f"Wrote {dest} rows={len(out)}", flush=True)


if __name__ == "__main__":
    main()
