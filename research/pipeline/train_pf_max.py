"""Largest Pulp Fiction net that fits on RTX 4060. Resume-safe. Notebook training recipe.

300-ep budget, ReduceLROnPlateau 0.85/7, ES patience 30 (notebook), Adam, full-seq MAE.
Select checkpoint on inspiratory MAE (competition metric).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from pf_features import robust_apply, robust_fit
from pf_model import WIDTH_LADDER, PulpFictionNet
from pipe_common import DEVICE, PIPE, SEED, VAL_FRAC, insp_mae, insp_mae_hard, keep_awake
from stage1_pf import load_pack, predict_np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from batch2_common import breath_split
from batch3_common import set_global_seed

CKPT = PIPE / "ckpt_pf_max"


def try_cudnn(on: bool) -> None:
    torch.backends.cudnn.benchmark = on
    torch.backends.cudnn.enabled = on


def vram_probe(n_feat: int, width, batch: int, checkpoint: bool) -> bool:
    torch.cuda.empty_cache()
    try:
        model = PulpFictionNet(n_feat, width, checkpoint=checkpoint).to(DEVICE)
        x = torch.randn(batch, 80, n_feat, device=DEVICE)
        y = torch.randn(batch, 80, device=DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler("cuda")
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            pred = model(x)
            loss = (pred - y).abs().mean()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        npar = sum(p.numel() for p in model.parameters())
        mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"  FIT width={width} bs={batch} ckpt={checkpoint} params={npar/1e6:.1f}M vram={mem:.2f}GB", flush=True)
        del model, x, y, opt, scaler, pred, loss
        torch.cuda.empty_cache()
        return True
    except torch.cuda.OutOfMemoryError:
        print(f"  OOM width={width} bs={batch} ckpt={checkpoint}", flush=True)
        torch.cuda.empty_cache()
        return False


def pick_config(n_feat: int):
    try_cudnn(True)
    for width in WIDTH_LADDER:
        for ckpt in (True, False):
            for bs in (32, 16, 8, 4):
                if vram_probe(n_feat, width, bs, ckpt):
                    return width, bs, ckpt, True
    try_cudnn(False)
    for width in WIDTH_LADDER:
        for ckpt in (True, False):
            for bs in (16, 8, 4):
                if vram_probe(n_feat, width, bs, ckpt):
                    return width, bs, ckpt, False
    raise SystemExit("no PF width fits this GPU")


def train(args):
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    torch.cuda.empty_cache()
    X, cols, meta = load_pack("train.csv")
    y, u, R, C, bids = meta["y"], meta["u_out"], meta["R"], meta["C"], meta["breath_id"]
    tr_ids, va_ids = breath_split(np.unique(bids), VAL_FRAC, SEED)
    tr = np.flatnonzero(np.isin(bids, list(tr_ids)))
    va = np.flatnonzero(np.isin(bids, list(va_ids)))
    med, iqr = robust_fit(X[tr])
    Xtr, Xva = robust_apply(X[tr], med, iqr), robust_apply(X[va], med, iqr)
    ytr, yva, utr, uva = y[tr], y[va], u[tr], u[va]
    n_feat = Xtr.shape[-1]

    cfg_path = CKPT / "config.json"
    CKPT.mkdir(exist_ok=True)
    if cfg_path.exists() and not args.reprobe:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        width, bs, ckpt, cudnn = tuple(cfg["width"]), cfg["bs"], cfg["checkpoint"], cfg["cudnn"]
        try_cudnn(cudnn)
        print("resume config", cfg, flush=True)
    else:
        print("VRAM probe...", flush=True)
        width, bs, ckpt, cudnn = pick_config(n_feat)
        cfg = {"width": list(width), "bs": bs, "checkpoint": ckpt, "cudnn": cudnn, "n_feat": n_feat}
        cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    set_global_seed(SEED)
    model = PulpFictionNet(n_feat, width, checkpoint=ckpt).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda")
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.85, patience=7, min_lr=1e-6)
    start_ep = 1
    best = {"mae": float("inf"), "epoch": 0}
    hist = []
    last = CKPT / "last.pt"
    if last.exists() and args.resume:
        pack = torch.load(last, map_location=DEVICE, weights_only=False)
        model.load_state_dict(pack["model"])
        opt.load_state_dict(pack["opt"])
        scaler.load_state_dict(pack["scaler"])
        start_ep = pack["epoch"] + 1
        best = pack.get("best", best)
        hist = pack.get("hist", [])
        print(f"resumed ep={start_ep} best_insp={best['mae']}", flush=True)

    loader = DataLoader(
        TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
        batch_size=bs, shuffle=True, pin_memory=True, drop_last=False,
    )
    patience = 30
    stale = 0
    with keep_awake():
        for ep in range(start_ep, args.epochs + 1):
            t0 = time.time()
            model.train()
            for xb, yb in loader:
                xb = xb.to(DEVICE, non_blocking=True)
                yb = yb.to(DEVICE, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda"):
                    pred = model(xb)
                    loss = (pred - yb).abs().mean()
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            pred_va = predict_np(model, Xva, bs=max(bs, 32))
            va_insp = insp_mae(yva, pred_va, uva)
            va_full = float(np.mean(np.abs(yva - pred_va)))
            hard = insp_mae_hard(yva, pred_va, uva, R[va], C[va])
            sched.step(va_insp)
            rec = {"epoch": ep, "insp": va_insp, "full": va_full, "hard": hard, "lr": opt.param_groups[0]["lr"], "sec": time.time() - t0}
            hist.append(rec)
            improved = va_insp < best["mae"] - 1e-5
            if improved:
                best = {"mae": va_insp, "epoch": ep, "full": va_full, "hard": hard, "pred": pred_va.astype(np.float32)}
                stale = 0
                torch.save({"model": model.state_dict(), "width": list(width), "med": med, "iqr": iqr, "cols": cols}, CKPT / "best.pt")
                np.savez_compressed(CKPT / "best_val.npz", pred=pred_va.astype(np.float32), y=yva, u_out=uva, R=R[va], C=C[va], breath_id=bids[va])
            else:
                stale += 1
            torch.save(
                {"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                 "epoch": ep, "best": {k: v for k, v in best.items() if k != "pred"}, "hist": hist,
                 "width": list(width), "med": med, "iqr": iqr},
                last,
            )
            print(
                f"  [PFMAX] ep={ep:03d} insp={va_insp:.4f} full={va_full:.4f} hard={hard:.4f} "
                f"best={best['mae']:.4f}@{best['epoch']} stale={stale} sec={rec['sec']:.1f}",
                flush=True,
            )
            (CKPT / "history.json").write_text(json.dumps({"best": {k: v for k, v in best.items() if k != "pred"}, "hist": hist, "cfg": cfg}, indent=2, default=float), encoding="utf-8")
            if stale >= patience:
                print(f"early stop at {ep} (notebook patience 30)", flush=True)
                break
    print("DONE best_insp", best["mae"], "ep", best["epoch"], flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--reprobe", action="store_true")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
