"""Apply saved PI trust masks to e1+e3 ensemble OOF; optionally ID test breaths."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inverse_control"))

from inv_common import snap_unique
from pi_engine import frequent_p0, identify_breath, overlay_nn
from pipe_common import DATA, PIPE, RESEARCH, insp_mae, insp_mae_hard, update_track
from stage2_ensemble import inv_mae_weights, weighted_mean


def main():
    e3 = np.load(RESEARCH / "batch3_oof_e3_aug_uin.npz")
    e1 = np.load(RESEARCH / "batch3_oof_e1_baseline.npz")
    y, u, R, C, bids = e3["y"], e3["u_out"], e3["R"], e3["C"], e3["breath_id"]
    preds = np.stack([e1["oof"], e3["oof"]], 0)
    w = inv_mae_weights(preds, y, u)
    ens = weighted_mean(preds, w)
    uniq = np.sort(pd.read_csv(DATA / "train.csv", usecols=["pressure"])["pressure"].unique().astype(np.float64))
    ens_snap = snap_unique(ens.reshape(-1), uniq).reshape(ens.shape)

    pi = np.load(PIPE / "pi_e3_full_overlay.npz")
    assert np.array_equal(pi["breath_id"], bids)
    hy_ens = overlay_nn(ens, pi["pi"], pi["trust"])
    hy_ens_snap = overlay_nn(ens_snap, pi["pi"], pi["trust"])
    rec = {
        "e3_oof": insp_mae(y, e3["oof"], u),
        "ensemble": insp_mae(y, ens, u),
        "ensemble_snap": insp_mae(y, ens_snap, u),
        "e3_plus_pi": insp_mae(y, pi["hybrid"], u),
        "ensemble_plus_pi": insp_mae(y, hy_ens, u),
        "ensemble_snap_plus_pi": insp_mae(y, hy_ens_snap, u),
        "hard_ensemble_snap": insp_mae_hard(y, ens_snap, u, R, C),
        "hard_ensemble_snap_pi": insp_mae_hard(y, hy_ens_snap, u, R, C),
        "pi_coverage_pct": float(100.0 * pi["trust"].reshape(-1)[u.reshape(-1) == 0].mean()),
        "weights": w.tolist(),
    }
    best_name = min(
        ["ensemble", "ensemble_snap", "e3_plus_pi", "ensemble_plus_pi", "ensemble_snap_plus_pi"],
        key=lambda k: rec[k],
    )
    rec["best"] = best_name
    rec["best_mae"] = rec[best_name]
    (PIPE / "hybrid_oof.json").write_text(json.dumps(rec, indent=2, default=float), encoding="utf-8")
    update_track("full_hybrid", {"status": "done", "protocol": "GroupKFold OOF", **rec})
    print(json.dumps(rec, indent=2, default=float), flush=True)

    # Test overlay if we have fold-mean preds
    test_npz = PIPE / "test_pred_e3_folds.npz"
    if not test_npz.exists():
        return
    tpack = np.load(test_npz)
    test_pred = tpack["inv_mae_weighted_snap"] if "inv_mae_weighted_snap" in tpack.files else tpack["inv_mae_weighted"]
    test_bids = tpack["breath_id"]
    df_tr = pd.read_csv(DATA / "train.csv", usecols=["breath_id", "pressure"])
    p0 = frequent_p0(df_tr.groupby("breath_id")["pressure"].first().to_numpy(), uniq, 80)
    df_te = pd.read_csv(DATA / "test.csv", usecols=["breath_id", "u_in", "u_out", "time_step", "id"])
    n = len(test_bids)
    sub = df_te.sort_values(["breath_id", "time_step"])
    order = {int(b): i for i, b in enumerate(test_bids)}
    uin = np.zeros((n, 80), np.float64)
    tt = np.zeros((n, 80), np.float64)
    uout = np.zeros((n, 80), np.float64)
    for b, g in sub.groupby("breath_id", sort=False):
        i = order.get(int(b))
        if i is None:
            continue
        uin[i] = g["u_in"].to_numpy(np.float64)
        tt[i] = g["time_step"].to_numpy(np.float64)
        uout[i] = g["u_out"].to_numpy(np.float64)

    print(f"PI test n={n}", flush=True)
    identify_breath(uin[0], uout[0], tt[0], uniq, p0[:8])
    pi_pred = np.full_like(test_pred, np.nan, dtype=np.float64)
    trust = np.zeros(test_pred.shape, dtype=bool)
    n_id = 0
    for i in range(n):
        rec_i = identify_breath(uin[i], uout[i], tt[i], uniq, p0)
        if rec_i is not None:
            n_id += 1
            pi_pred[i] = rec_i["pred"]
            trust[i] = rec_i["trust"]
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{n} id_ok={n_id}", flush=True)
    hybrid = overlay_nn(test_pred.astype(np.float64), pi_pred, trust)
    test_sorted = sub.reset_index(drop=True)
    out = pd.DataFrame({"id": test_sorted["id"].to_numpy(), "pressure": hybrid.reshape(-1)})
    out = out.sort_values("id")
    dest = RESEARCH.parent / "predictions.csv"
    # only replace file if OOF said PI helps the chosen ensemble
    if rec["ensemble_snap_plus_pi"] <= rec["ensemble_snap"] + 1e-6:
        out.to_csv(dest, index=False)
        print(f"Updated {dest} with PI overlay, test_id={n_id}/{n}", flush=True)
    else:
        print("PI did not help ensemble OOF — keep snap-only predictions.csv", flush=True)
    (PIPE / "test_pi_meta.json").write_text(
        json.dumps({"n_id": n_id, "n": n, "used": rec["ensemble_snap_plus_pi"] <= rec["ensemble_snap"] + 1e-6}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
