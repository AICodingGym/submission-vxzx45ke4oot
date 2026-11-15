"""
Pulp Fiction deltas on OUR BiLSTM — holdout screens only.

Not a 7-fold / 5-stack TPU clone. Two one-factor feature ablations vs the
Batch 3 e1_baseline screen (insp MAE 0.2808, same 80/20 breath split seed 42).

  1. pf_lagback   — within-breath future u_in lags 1–2 (+ diffs)
  2. pf_rc_onehot — R / C / R__C one-hots instead of numeric R, C

Writes pf_ablation_results.json, pf_ablation_report.txt, pf_ablation_*.npz.
Does not touch batch3_state.json or gym submit.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from batch2_common import (  # noqa: E402
    DATA,
    SEQ_COLS_RICH,
    BiLSTMReg,
    add_features,
    apply_norm,
    breath_split,
    insp_mae,
    insp_mae_by_rc,
    norm_fit,
    reshape_sequences,
    train_model,
)
from batch3_common import set_global_seed  # noqa: E402

HERE = Path(__file__).resolve().parent
SEED = 42
VAL_FRAC = 0.2
EPOCHS = 75
LR = 3e-3
E1_HOLD = 0.2807616591453552
E1_HARD = 0.3119116723537445
E3_HOLD = 0.2615533471107483
R_VALS = (5, 20, 50)
C_VALS = (10, 20, 50)

LAGBACK_COLS = [
    "u_in_lag_back1",
    "u_in_lag_back2",
    "u_in_lagback_diff1",
    "u_in_lagback_diff2",
]
RC_OH_COLS = (
    [f"R_{r}" for r in R_VALS]
    + [f"C_{c}" for c in C_VALS]
    + [f"RC_{r}_{c}" for r in R_VALS for c in C_VALS]
)

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002


@contextmanager
def keep_awake():
    k32 = ctypes.windll.kernel32
    k32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED)
    try:
        yield
    finally:
        k32.SetThreadExecutionState(ES_CONTINUOUS)


def add_pf_features(df: pd.DataFrame) -> pd.DataFrame:
    """PF-inspired extras on top of our add_features (own implementation)."""
    df = df.copy()
    g = df.groupby("breath_id", sort=False)
    df["u_in_lag_back1"] = g["u_in"].shift(-1).fillna(0.0)
    df["u_in_lag_back2"] = g["u_in"].shift(-2).fillna(0.0)
    df["u_in_lagback_diff1"] = df["u_in"] - df["u_in_lag_back1"]
    df["u_in_lagback_diff2"] = df["u_in"] - df["u_in_lag_back2"]
    r = df["R"].to_numpy()
    c = df["C"].to_numpy()
    for rv in R_VALS:
        df[f"R_{rv}"] = (r == rv).astype(np.float32)
    for cv in C_VALS:
        df[f"C_{cv}"] = (c == cv).astype(np.float32)
    for rv in R_VALS:
        for cv in C_VALS:
            df[f"RC_{rv}_{cv}"] = ((r == rv) & (c == cv)).astype(np.float32)
    return df


def sanity_lagback(df: pd.DataFrame) -> None:
    u = df["u_in"].to_numpy().reshape(-1, 80)
    b1 = df["u_in_lag_back1"].to_numpy().reshape(-1, 80)
    if not np.allclose(b1[:, :-1], u[:, 1:]):
        raise RuntimeError("u_in_lag_back1 is not next-step u_in within breath")
    if not np.allclose(b1[:, -1], 0.0):
        raise RuntimeError("u_in_lag_back1 last step should be 0")


def load_results() -> dict:
    path = HERE / "pf_ablation_results.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "protocol": "Batch3 e1 screen: holdout 20% breath_split seed 42, 75ep AMP mse y-scale BiLSTM 256x3",
        "control_e1_baseline": {"val_insp_mae": E1_HOLD, "hard50_10_mae": E1_HARD},
        "reference_e3_aug_uin_screen": {"val_insp_mae": E3_HOLD},
        "runs": {},
    }


def save_results(blob: dict) -> None:
    (HERE / "pf_ablation_results.json").write_text(
        json.dumps(blob, indent=2, default=float), encoding="utf-8"
    )


def write_report(blob: dict) -> None:
    lines = [
        "Pulp Fiction feature ablations — holdout screen (not OOF)",
        f"Control e1_baseline holdout insp MAE {E1_HOLD:.4f}  hard50/10 {E1_HARD:.4f}",
        f"e3_aug_uin screen (not this recipe) {E3_HOLD:.4f}",
        "",
    ]
    for key, run in blob.get("runs", {}).items():
        if not run.get("done"):
            lines.append(f"{key}: in progress / incomplete")
            continue
        d_e1 = E1_HOLD - run["val_insp_mae"]
        lines.append(
            f"{key}: val={run['val_insp_mae']:.4f}  hard={run['hard50_10_mae']:.4f}  "
            f"ep={run['best_epoch']}  n_feat={run['n_feat']}  "
            f"Δ vs e1={d_e1:+.4f}  sec={run['runtime_sec']:.0f}"
        )
        lines.append(f"  cols extra: {run.get('extra_cols')}")
    lines.append("")
    lines.append("Holdout only. No GroupKFold, no gym submit, no 7-model clone.")
    (HERE / "pf_ablation_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def screen_one(feat: pd.DataFrame, cols: list[str], key: str) -> dict:
    X, y, u, R, C, bids = reshape_sequences(feat, cols)
    tr_ids, va_ids = breath_split(np.unique(bids), VAL_FRAC, SEED)
    tr_idx = np.flatnonzero(np.isin(bids, list(tr_ids)))
    va_idx = np.flatnonzero(np.isin(bids, list(va_ids)))
    mean, std = norm_fit(X[tr_idx])
    Xtrn = apply_norm(X[tr_idx], mean, std)
    Xvan = apply_norm(X[va_idx], mean, std)
    print(
        f"  [{key}] n_feat={Xtrn.shape[-1]} n_tr={len(tr_idx)} n_va={len(va_idx)}",
        flush=True,
    )
    set_global_seed(SEED)
    model = BiLSTMReg(Xtrn.shape[-1], 256, 3, 0.1)
    t0 = time.time()
    res = train_model(
        model,
        Xtrn, y[tr_idx], u[tr_idx],
        Xvan, y[va_idx], u[va_idx],
        R_va=R[va_idx], C_va=C[va_idx],
        epochs=EPOCHS, lr=LR, loss_kind="mse", schedule="plateau",
        y_scale=True, label=f"PF:{key}", log_every=10, use_amp=True,
    )
    pred = res["pred_va"]
    val_mae = insp_mae(y[va_idx].reshape(-1), pred.reshape(-1), u[va_idx].reshape(-1))
    hard = insp_mae_by_rc(
        y[va_idx].reshape(-1), pred.reshape(-1), u[va_idx].reshape(-1),
        R[va_idx].reshape(-1), C[va_idx].reshape(-1),
    )
    np.savez_compressed(
        HERE / f"pf_ablation_{key}.npz",
        breath_id=bids[va_idx],
        pred=pred.astype(np.float32),
        y=y[va_idx],
        u_out=u[va_idx],
        R=R[va_idx],
        C=C[va_idx],
    )
    out = {
        "done": True,
        "val_insp_mae": float(val_mae),
        "hard50_10_mae": float(hard),
        "best_epoch": int(res["best_epoch"]),
        "best_train": res.get("best_train"),
        "n_feat": int(Xtrn.shape[-1]),
        "n_tr": int(len(tr_idx)),
        "n_va": int(len(va_idx)),
        "runtime_sec": float(time.time() - t0),
        "stopping_reason": res.get("stopping_reason"),
        "delta_vs_e1": float(E1_HOLD - val_mae),
        "extra_cols": [c for c in cols if c not in SEQ_COLS_RICH],
        "dropped_vs_rich": [c for c in SEQ_COLS_RICH if c not in cols],
    }
    del model, res
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


CONFIGS = {
    "pf_lagback": {
        "cols": SEQ_COLS_RICH + LAGBACK_COLS,
        "note": "future u_in within breath (known at test; not label leak)",
    },
    "pf_rc_onehot": {
        "cols": [c for c in SEQ_COLS_RICH if c not in ("R", "C")] + RC_OH_COLS,
        "note": "replace numeric R,C with R/C/RC one-hots; keep physics products",
    },
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--only", nargs="+", choices=list(CONFIGS), default=list(CONFIGS))
    args = p.parse_args()

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False
        torch.cuda.empty_cache()
        print(f"GPU: {torch.cuda.get_device_name(0)} (cudnn disabled)", flush=True)
    else:
        print("WARNING: CUDA not available — aborting PF GPU screen", flush=True)
        raise SystemExit(1)

    blob = load_results()
    print("Loading train + features...", flush=True)
    train = pd.read_csv(DATA / "train.csv")
    feat = add_pf_features(add_features(train))
    sanity_lagback(feat)
    print(f"feat rows={len(feat)} lagback+oh cols ok", flush=True)

    with keep_awake():
        for key in args.only:
            prev = blob.get("runs", {}).get(key, {})
            if prev.get("done"):
                print(f"  [{key}] cached val={prev['val_insp_mae']:.4f}", flush=True)
                continue
            spec = CONFIGS[key]
            print(f"\n=== {key}: {spec['note']} ===", flush=True)
            blob.setdefault("runs", {})[key] = {"done": False, "note": spec["note"]}
            save_results(blob)
            blob["runs"][key] = screen_one(feat, spec["cols"], key)
            blob["runs"][key]["note"] = spec["note"]
            save_results(blob)
            write_report(blob)
            d = blob["runs"][key]["delta_vs_e1"]
            print(
                f"  [{key}] DONE val={blob['runs'][key]['val_insp_mae']:.4f} "
                f"hard={blob['runs'][key]['hard50_10_mae']:.4f} Δe1={d:+.4f}",
                flush=True,
            )

    write_report(blob)
    log_path = HERE / "experiment_log.md"
    finished = [k for k in args.only if blob["runs"].get(k, {}).get("done")]
    stamp = time.strftime("%Y-%m-%d %H:%M")
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"\n## PF ablations ({stamp})\n")
        for k in finished:
            r = blob["runs"][k]
            f.write(
                f"- {k}: val={r['val_insp_mae']:.4f} hard={r['hard50_10_mae']:.4f} "
                f"Δe1={r['delta_vs_e1']:+.4f} ep={r['best_epoch']} "
                f"sec={r['runtime_sec']:.0f}\n"
            )
    print("Wrote pf_ablation_report.txt", flush=True)


if __name__ == "__main__":
    main()
