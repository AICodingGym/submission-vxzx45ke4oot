"""Stage 1 — Pulp Fiction neural, own weights, gym data only.

Probe (subset, few epochs) → estimate cost → holdout 80/20 seed 42.
Promote to a second seed only if holdout beats e1 (0.2808) by ≥0.005.
Promote to GroupKFold only if holdout ≤ 0.27.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from pf_features import pf_feature_matrix, robust_apply, robust_fit
from pf_model import PF_SMALL, PF_TINY, PulpFictionNet
from pipe_common import (
    DATA,
    DEVICE,
    E1_HOLD,
    E3_HOLD,
    PIPE,
    SEED,
    VAL_FRAC,
    insp_mae,
    insp_mae_hard,
    keep_awake,
    setup_cuda,
    update_track,
)

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from batch2_common import breath_split  # noqa: E402
from batch3_common import set_global_seed  # noqa: E402


def predict_np(model, X, bs=256):
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i : i + bs]).to(DEVICE, non_blocking=True)
            outs.append(model(xb).float().cpu().numpy())
    return np.concatenate(outs, axis=0)


def train_pf(
    model,
    Xtr, ytr, utr,
    Xva, yva, uva,
    Rva=None, Cva=None,
    *,
    epochs=80,
    lr=1e-3,
    batch_size=256,
    patience=20,
    plateau_patience=7,
    factor=0.85,
    label="PF",
    log_every=5,
    use_amp=True,
):
    """Notebook: Adam + full-sequence MAE + ReduceLROnPlateau(0.85, 7) + ES.

    We optimize their loss (all 80 steps) but *select* on inspiratory MAE.
    AMP is a 4060 speed concession, not in the TPU notebook.
    """
    model = model.to(DEVICE)
    use_amp = use_amp and DEVICE.type == "cuda"
    loader = DataLoader(
        TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
        batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0,
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=factor, patience=plateau_patience, min_lr=1e-6,
    )
    best = {"mae": float("inf"), "epoch": 0, "pred": None, "state": None, "full_mae": None}
    stale = 0
    hist = []
    t_ep = []
    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        for xb, yb in loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = model(xb)
                loss = (pred - yb).abs().mean()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        pred_va = predict_np(model, Xva)
        va_insp = insp_mae(yva, pred_va, uva)
        va_full = float(np.mean(np.abs(yva - pred_va)))
        hard = insp_mae_hard(yva, pred_va, uva, Rva, Cva) if Rva is not None else None
        sched.step(va_insp)
        lr_now = opt.param_groups[0]["lr"]
        hist.append({"epoch": ep, "insp": va_insp, "full": va_full, "hard": hard, "lr": lr_now})
        t_ep.append(time.time() - t0)
        if va_insp < best["mae"] - 1e-5:
            best = {
                "mae": va_insp, "epoch": ep, "pred": pred_va.copy(),
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "full_mae": va_full, "hard": hard,
            }
            stale = 0
        else:
            stale += 1
        if ep == 1 or ep % log_every == 0 or ep == epochs:
            extra = f" hard={hard:.4f}" if hard is not None else ""
            print(
                f"  [{label}] ep={ep:03d} insp={va_insp:.4f} full={va_full:.4f} "
                f"lr={lr_now:.2e} sec/ep={t_ep[-1]:.1f}{extra}",
                flush=True,
            )
        if stale >= patience:
            print(f"  [{label}] early stop ep={ep} stale={stale}", flush=True)
            break
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "model": model,
        "best_val": float(best["mae"]),
        "best_full": float(best["full_mae"]) if best["full_mae"] is not None else None,
        "best_epoch": int(best["epoch"]),
        "best_hard": float(best["hard"]) if best.get("hard") is not None else None,
        "pred_va": best["pred"],
        "history": hist,
        "sec_per_epoch": float(np.median(t_ep)) if t_ep else None,
        "epochs_ran": len(hist),
    }


def load_pack(csv_name: str):
    cache = PIPE / f"cache_{csv_name.replace('.csv', '')}_pf.npz"
    if cache.exists() and csv_name == "train.csv":
        pack = np.load(cache, allow_pickle=True)
        cols = pack["cols"].tolist()
        meta = {k: pack[k] for k in ("breath_id", "id", "y", "u_out", "R", "C", "u_in", "time_step")}
        print(f"  features cached {cache.name} X={pack['X'].shape}", flush=True)
        return pack["X"], cols, meta
    df = pd.read_csv(DATA / csv_name)
    X, cols, meta = pf_feature_matrix(df)
    if csv_name == "train.csv":
        PIPE.mkdir(exist_ok=True)
        np.savez_compressed(cache, X=X, cols=np.array(cols), **meta)
        print(f"  wrote {cache.name}", flush=True)
    return X, cols, meta


def run_probe(n_breaths=8000, epochs=3, width=PF_TINY, batch_size=256):
    gpu = setup_cuda()
    print(f"PROBE GPU={gpu} n={n_breaths} ep={epochs} width={width}", flush=True)
    X, cols, meta = load_pack("train.csv")
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(X), size=min(n_breaths, len(X)), replace=False)
    X, y, u = X[idx], meta["y"][idx], meta["u_out"][idx]
    R, C = meta["R"][idx], meta["C"][idx]
    bids = meta["breath_id"][idx]
    tr_ids, va_ids = breath_split(np.unique(bids), 0.2, SEED)
    tr = np.flatnonzero(np.isin(bids, list(tr_ids)))
    va = np.flatnonzero(np.isin(bids, list(va_ids)))
    med, iqr = robust_fit(X[tr])
    Xtr, Xva = robust_apply(X[tr], med, iqr), robust_apply(X[va], med, iqr)
    set_global_seed(SEED)
    model = PulpFictionNet(Xtr.shape[-1], width).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    with torch.no_grad():
        _ = model(torch.from_numpy(Xtr[:2]).to(DEVICE))
    print(f"  n_feat={Xtr.shape[-1]} params={n_params/1e6:.2f}M forward_ok", flush=True)
    with keep_awake():
        res = train_pf(
            model, Xtr, y[tr], u[tr], Xva, y[va], u[va], R[va], C[va],
            epochs=epochs, batch_size=batch_size, label="PROBE", log_every=1, patience=99,
        )
    sec = res["sec_per_epoch"]
    n_full = 54324  # ~80% of 67905
    est_holdout_min = (sec * n_full / max(len(tr), 1) * 60) / 60.0
    est = {
        "sec_per_epoch_subset": sec,
        "subset_train_breaths": int(len(tr)),
        "est_sec_per_epoch_full": float(sec * n_full / max(len(tr), 1)),
        "est_holdout_60ep_min": float(sec * n_full / max(len(tr), 1) * 60 / 60.0),
        "params_M": n_params / 1e6,
        "n_feat": int(Xtr.shape[-1]),
        "width": list(width),
        "best_insp_subset": res["best_val"],
        "gpu": gpu,
    }
    (PIPE / "stage1_probe.json").write_text(json.dumps(est, indent=2, default=float), encoding="utf-8")
    print("PROBE", json.dumps(est, indent=2, default=float), flush=True)
    return est


def run_holdout(width=PF_SMALL, epochs=80, batch_size=256, seed=SEED, tag="pf_small"):
    gpu = setup_cuda()
    print(f"HOLDOUT GPU={gpu} width={width} ep={epochs} tag={tag}", flush=True)
    X, cols, meta = load_pack("train.csv")
    y, u, R, C, bids = meta["y"], meta["u_out"], meta["R"], meta["C"], meta["breath_id"]
    tr_ids, va_ids = breath_split(np.unique(bids), VAL_FRAC, SEED)
    tr = np.flatnonzero(np.isin(bids, list(tr_ids)))
    va = np.flatnonzero(np.isin(bids, list(va_ids)))
    med, iqr = robust_fit(X[tr])
    Xtr, Xva = robust_apply(X[tr], med, iqr), robust_apply(X[va], med, iqr)
    set_global_seed(seed)
    model = PulpFictionNet(Xtr.shape[-1], width)
    with keep_awake():
        res = train_pf(
            model, Xtr, y[tr], u[tr], Xva, y[va], u[va], R[va], C[va],
            epochs=epochs, batch_size=batch_size, label=f"PF:{tag}:s{seed}", log_every=5,
        )
    pred = res["pred_va"]
    rec = {
        "status": "holdout_done",
        "tag": tag,
        "width": list(width),
        "seed": seed,
        "protocol": f"holdout {VAL_FRAC} split_seed={SEED} train_seed={seed}",
        "val_insp_mae": float(res["best_val"]),
        "val_full_mae": res["best_full"],
        "hard50_10_mae": res["best_hard"],
        "best_epoch": res["best_epoch"],
        "epochs_ran": res["epochs_ran"],
        "sec_per_epoch": res["sec_per_epoch"],
        "n_feat": int(Xtr.shape[-1]),
        "n_tr": int(len(tr)),
        "n_va": int(len(va)),
        "delta_vs_e1": float(E1_HOLD - res["best_val"]),
        "delta_vs_e3_screen": float(E3_HOLD - res["best_val"]),
        "gpu": gpu,
    }
    np.savez_compressed(
        PIPE / f"stage1_{tag}_s{seed}_holdout.npz",
        breath_id=bids[va], pred=pred.astype(np.float32), y=y[va], u_out=u[va],
        R=R[va], C=C[va],
    )
    wdir = PIPE / "weights"
    wdir.mkdir(exist_ok=True)
    torch.save(
        {"state": res["model"].state_dict(), "med": med, "iqr": iqr, "width": list(width), "cols": cols},
        wdir / f"{tag}_s{seed}.pt",
    )
    (PIPE / f"stage1_{tag}_s{seed}.json").write_text(json.dumps(rec, indent=2, default=float), encoding="utf-8")
    update_track("pf_neural", rec)
    print("HOLDOUT", json.dumps(rec, indent=2, default=float), flush=True)
    return rec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["probe", "holdout"], required=True)
    p.add_argument("--width", choices=["tiny", "small"], default="small")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--probe-breaths", type=int, default=8000)
    args = p.parse_args()
    width = PF_TINY if args.width == "tiny" else PF_SMALL
    PIPE.mkdir(exist_ok=True)
    if args.mode == "probe":
        run_probe(n_breaths=args.probe_breaths, epochs=args.epochs or 3, width=width, batch_size=args.batch_size)
    else:
        run_holdout(width=width, epochs=args.epochs or 80, batch_size=args.batch_size, seed=args.seed, tag=f"pf_{args.width}")


if __name__ == "__main__":
    main()
