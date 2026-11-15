"""Apply validated P overlay (tau=5) on test, on top of q0 PI hybrid."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inverse_control"))

from inv_common import snap_unique
from pi_p_overlay import identify_p_breath
from pipe_common import DATA, PIPE, RESEARCH, keep_awake, update_track

TAU = 5.0


def main():
    uniq = np.sort(pd.read_csv(DATA / "train.csv", usecols=["pressure"])["pressure"].unique().astype(np.float64))
    tpack = np.load(PIPE / "test_pred_e3_folds.npz")
    nn = tpack["inv_mae_weighted_snap"].astype(np.float64)
    bids = tpack["breath_id"]
    q0 = np.load(PIPE / "pi_q0_test.npz")
    pi_pred = q0["pred"].astype(np.float64)
    pi_trust = q0["trust"].astype(bool)
    # rebuild q0 hybrid at tau=5
    g_pi = pi_trust & np.isfinite(pi_pred) & (np.abs(pi_pred - nn) < TAU)
    hy = nn.copy()
    hy[g_pi] = pi_pred[g_pi]

    test = pd.read_csv(DATA / "test.csv")
    sub = test.sort_values(["breath_id", "time_step"])
    n = len(bids)
    order = {int(b): i for i, b in enumerate(bids)}
    uin = np.zeros((n, 80), np.float64)
    uout = np.zeros((n, 80), np.float64)
    ids = np.zeros((n, 80), np.int64)
    for b, g in sub.groupby("breath_id", sort=False):
        i = order.get(int(b))
        if i is None:
            continue
        uin[i] = g["u_in"].to_numpy(np.float64)
        uout[i] = g["u_out"].to_numpy(np.float64)
        ids[i] = g["id"].to_numpy(np.int64)

    n_add = 0
    extra = 0
    with keep_awake():
        for i in range(n):
            rec = identify_p_breath(uin[i], uout[i], uniq)
            if rec is None:
                continue
            pred, trust, _ = rec
            m = trust & ~g_pi[i] & np.isfinite(pred) & (np.abs(pred - nn[i]) < TAU)
            if m.sum() < 3:
                continue
            hy[i][m] = pred[m]
            n_add += 1
            extra += int(m.sum())
            if (i + 1) % 1000 == 0:
                print(f"  {i+1}/{n} p_add={n_add}", flush=True)

    out = pd.DataFrame({"id": ids.reshape(-1), "pressure": hy.reshape(-1)})
    out = out.sort_values("id")
    dest = RESEARCH.parent / "predictions.csv"
    out.to_csv(dest, index=False)
    meta = {
        "file": str(dest),
        "n": len(out),
        "model": "e1+e3 snap + q0 PI tau=5 + P tau=5",
        "oof_hybrid": 0.14927280181439606,
        "prev_lb": 0.13884,
        "test_p_breaths": n_add,
        "test_p_steps": extra,
        "pressure_mean": float(out["pressure"].mean()),
    }
    (PIPE / "submit_q0p_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    update_track("full_hybrid", {"status": "q0_plus_P", "oof": 0.14927280181439606, "prev_lb": 0.13884})
    print(json.dumps(meta, indent=2), flush=True)
    print(f"Wrote {dest}", flush=True)


if __name__ == "__main__":
    main()
