"""Batch 3 shared: k-fold OOF, models, analysis."""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold

import json
from pathlib import Path

from batch2_common import (
    DEVICE,
    OUT,
    SEQ_COLS_RICH,
    BiLSTMReg,
    add_features,
    apply_norm,
    breath_split,
    insp_mae,
    insp_mae_by_rc,
    norm_fit,
    predict_np,
    reshape_sequences,
    train_model,
    masked_loss,
)
from batch3_state import CKPT_CV_PATH, load_state, record_screening, screening_done
from batch3_screen_policy import ScreenStopConfig, screening_leader_mae

PRESSURE_STEP = 0.07030214544382634
BATCH2_OOF = 0.2506
BATCH2_OOF_STD = 0.0194

AUG_NONE = "none"
AUG_UIN = "uin_noise"
AUG_MASK = "feat_mask"


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_folds(bids: np.ndarray, n_splits: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    gkf = GroupKFold(n_splits=n_splits)
    order = np.argsort(bids)
    bids_sorted = bids[order]
    splits = []
    for tr, va in gkf.split(np.arange(len(bids_sorted)), groups=bids_sorted):
        splits.append((order[tr], order[va]))
    return splits


def snap_lattice(pred: np.ndarray, p0: float, step: float = PRESSURE_STEP) -> np.ndarray:
    return p0 + np.round((pred - p0) / step) * step


def snap_nearest(pred: np.ndarray, uniq: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(uniq, pred)
    idx = np.clip(idx, 1, len(uniq) - 1)
    left, right = uniq[idx - 1], uniq[idx]
    return np.where(np.abs(pred - left) <= np.abs(pred - right), left, right)


@dataclass
class TrainRecipe:
    name: str = "bilstm_rich"
    hidden: int = 256
    layers: int = 3
    lr: float = 3e-3
    loss: str = "mse"
    y_scale: bool = True
    epochs: int = 90
    augment: str = AUG_NONE
    multitask: bool = False
    residual: bool = False
    hybrid_physics: bool = False
    aux_dp: float = 0.2
    aux_cum: float = 0.2
    seeds: list[int] = field(default_factory=lambda: [42])
    use_amp: bool = True


class ResidualBiLSTM(nn.Module):
    """Project input -> hidden, add to LSTM output before head."""
    def __init__(self, n_feat, hidden=256, layers=3, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(n_feat, hidden * 2)
        self.lstm = nn.LSTM(
            n_feat, hidden, num_layers=layers, batch_first=True,
            bidirectional=True, dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, 1),
        )

    def forward(self, x):
        h, _ = self.lstm(x)
        h = h + self.proj(x)
        return self.head(h).squeeze(-1)


class MultiTaskBiLSTM(nn.Module):
    def __init__(self, n_feat, hidden=256, layers=3, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            n_feat, hidden, num_layers=layers, batch_first=True,
            bidirectional=True, dropout=dropout if layers > 1 else 0.0,
        )
        d = hidden * 2
        self.head_p = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))
        self.head_dp = nn.Sequential(nn.Linear(d, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))
        self.head_cum = nn.Sequential(nn.Linear(d, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))

    def forward(self, x):
        h, _ = self.lstm(x)
        p = self.head_p(h).squeeze(-1)
        dp = self.head_dp(h).squeeze(-1)
        cum = self.head_cum(h).squeeze(-1)
        return p, dp, cum


def physics_pressure(vol_over_c, vol_r_over_c, u_r_over_c, u_in):
    """Fixed coefficients from global insp fit (weak learner); used as hybrid prior."""
    return (
        0.64 * vol_over_c + 0.12 * vol_r_over_c
        + 0.08 * u_r_over_c + 0.02 * u_in + 2.5
    ).astype(np.float32)


def train_multitask(
    model, X_tr, y_tr, u_tr, X_va, y_va, u_va,
    R_va=None, C_va=None,
    *, lr=3e-3, epochs=90, y_scale=True, aux_dp=0.2, aux_cum=0.2, seed=42, label="",
):
    set_global_seed(seed)
    model = model.to(DEVICE)
    dy_tr = np.zeros_like(y_tr)
    dy_tr[:, 1:] = y_tr[:, 1:] - y_tr[:, :-1]
    cum_tr = np.cumsum(y_tr, axis=1)
    dy_va = np.zeros_like(y_va)
    dy_va[:, 1:] = y_va[:, 1:] - y_va[:, :-1]
    cum_va = np.cumsum(y_va, axis=1)

    if y_scale:
        ym = float(y_tr[u_tr == 0].mean())
        ys = max(float(y_tr[u_tr == 0].std()), 1e-6)
        y_tr_t = (y_tr - ym) / ys
        y_va_t = (y_va - ym) / ys
        dy_tr_t, dy_va_t = dy_tr / ys, dy_va / ys
        cum_tr_t, cum_va_t = (cum_tr - ym) / ys, (cum_va - ym) / ys
    else:
        ym, ys = 0.0, 1.0
        y_tr_t, y_va_t = y_tr, y_va
        dy_tr_t, dy_va_t = dy_tr, dy_va
        cum_tr_t, cum_va_t = cum_tr, cum_va

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=10, min_lr=1e-5)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(X_tr),
            torch.from_numpy(y_tr_t.astype(np.float32)),
            torch.from_numpy(dy_tr_t.astype(np.float32)),
            torch.from_numpy(cum_tr_t.astype(np.float32)),
            torch.from_numpy(u_tr),
        ),
        batch_size=256, shuffle=True,
    )
    best = {"mae": float("inf"), "state": None, "pred": None, "ep": 0}
    stale, stop_patience = 0, 25
    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb, db, cb, ub in loader:
            xb, yb, db, cb, ub = [t.to(DEVICE) for t in (xb, yb, db, cb, ub)]
            p, dp, cum = model(xb)
            m = ub == 0
            lp = masked_loss(p, yb, ub, kind="mse")
            ld = (torch.abs(dp - db) * m.float()).sum() / m.float().sum().clamp_min(1.0)
            lc = (torch.abs(cum - cb) * m.float()).sum() / m.float().sum().clamp_min(1.0)
            loss = lp + aux_dp * ld + aux_cum * lc
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        p_va = predict_multitask_p(model, X_va) * ys + ym
        va_mae = insp_mae(y_va.reshape(-1), p_va.reshape(-1), u_va.reshape(-1))
        sched.step(va_mae)
        if va_mae < best["mae"] - 1e-5:
            best = {"mae": va_mae, "state": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                    "pred": p_va.copy(), "ep": ep}
            stale = 0
        else:
            stale += 1
        if ep % 15 == 0 or ep == 1:
            print(f"  [{label}] ep={ep} val={va_mae:.4f}", flush=True)
        if stale >= stop_patience:
            break
    model.load_state_dict(best["state"])
    return {"best_val": best["mae"], "pred_va": best["pred"], "best_epoch": best["ep"]}


def predict_multitask_p(model, X):
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), 512):
            xb = torch.from_numpy(X[i : i + 512]).to(DEVICE)
            p, _, _ = model(xb)
            outs.append(p.cpu().numpy())
    return np.concatenate(outs, axis=0)


def augment_batch(xb: torch.Tensor, kind: str, u_idx: int, rng: np.random.Generator) -> torch.Tensor:
    if kind == AUG_NONE:
        return xb
    xb = xb.clone()
    if kind == AUG_UIN:
        noise = torch.from_numpy(rng.normal(0, 0.8, size=(xb.size(0), xb.size(1))).astype(np.float32)).to(xb.device)
        xb[:, :, u_idx] = xb[:, :, u_idx] + noise
    elif kind == AUG_MASK:
        mask = torch.from_numpy(rng.random((xb.size(0), xb.size(1), xb.size(2))) > 0.12).to(xb.device)
        xb = xb * mask.float()
    return xb


def train_with_augment(model, X_tr, y_tr, u_tr, X_va, y_va, u_va, u_idx, augment, seed, **kwargs):
    """Wrap train_model with optional augmentation via custom loop - fall back to train_model if none."""
    if augment == AUG_NONE:
        set_global_seed(seed)
        return train_model(model, X_tr, y_tr, u_tr, X_va, y_va, u_va, **kwargs)
    # minimal augmented training loop reusing hyperparams from kwargs
    set_global_seed(seed)
    model = model.to(DEVICE)
    y_scale = kwargs.get("y_scale", True)
    lr = kwargs.get("lr", 3e-3)
    epochs = kwargs.get("epochs", 90)
    loss_kind = kwargs.get("loss_kind", "mse")
    label = kwargs.get("label", "")
    if y_scale:
        ym = float(y_tr[u_tr == 0].mean())
        ys = max(float(y_tr[u_tr == 0].std()), 1e-6)
        y_tr_t = (y_tr - ym) / ys
        y_va_t = (y_va - ym) / ys
    else:
        ym, ys = 0.0, 1.0
        y_tr_t, y_va_t = y_tr, y_va
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(X_tr), torch.from_numpy(y_tr_t.astype(np.float32)), torch.from_numpy(u_tr),
        ),
        batch_size=256, shuffle=True,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=10, min_lr=1e-5)
    rng = np.random.default_rng(seed)
    best = {"mae": float("inf"), "state": None, "pred": None, "ep": 0}
    stale = 0
    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb, ub in loader:
            xb, yb, ub = xb.to(DEVICE), yb.to(DEVICE), ub.to(DEVICE)
            xb = augment_batch(xb, augment, u_idx, rng)
            pred = model(xb)
            loss = masked_loss(pred, yb, ub, kind=loss_kind)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        pred_va_s = predict_np(model, X_va)
        va_mae = insp_mae(y_va.reshape(-1), (pred_va_s * ys + ym).reshape(-1), u_va.reshape(-1))
        sched.step(va_mae)
        if va_mae < best["mae"] - 1e-5:
            best = {"mae": va_mae, "ep": ep,
                    "state": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                    "pred": (pred_va_s * ys + ym).copy()}
            stale = 0
        else:
            stale += 1
        if ep % 15 == 0 or ep == 1:
            print(f"  [{label}] ep={ep} val={va_mae:.4f}", flush=True)
        if stale >= 25:
            break
    model.load_state_dict(best["state"])
    return {"best_val": best["mae"], "pred_va": best["pred"], "best_epoch": best["ep"]}


def assert_gpu_training(model, label: str = "") -> None:
    if DEVICE.type != "cuda":
        print(f"WARNING [{label}]: training on CPU", flush=True)
        return
    p = next(model.parameters())
    if p.device.type != "cuda":
        raise RuntimeError(f"Model not on GPU for {label}")


def train_recipe_on_split(
    X, y, u, R, C,
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    recipe: TrainRecipe,
    seed: int,
    cols=SEQ_COLS_RICH,
    label: str = "",
    screen_stop: ScreenStopConfig | None = None,
    screen_leader_mae: float | None = None,
):
    """Train one recipe on a breath-index split; return val preds + metrics."""
    set_global_seed(seed)
    u_idx = cols.index("u_in")
    Xtr, Xva = X[tr_idx], X[va_idx]
    ytr, yva = y[tr_idx], y[va_idx]
    utr, uva = u[tr_idx], u[va_idx]
    mean, std = norm_fit(Xtr)
    Xtrn, Xvan = apply_norm(Xtr, mean, std), apply_norm(Xva, mean, std)

    if recipe.hybrid_physics:
        raise NotImplementedError("hybrid physics: use screen_e5_hybrid in batch3_run")

    if recipe.multitask:
        model = MultiTaskBiLSTM(Xtrn.shape[-1], recipe.hidden, recipe.layers, 0.1)
        res = train_multitask(
            model, Xtrn, ytr, utr, Xvan, yva, uva,
            R_va=R[va_idx], C_va=C[va_idx],
            lr=recipe.lr, epochs=recipe.epochs, y_scale=recipe.y_scale,
            aux_dp=recipe.aux_dp, aux_cum=recipe.aux_cum, seed=seed, label=label,
        )
        pred = res["pred_va"]
        ep = res["best_epoch"]
    elif recipe.residual:
        model = ResidualBiLSTM(Xtrn.shape[-1], recipe.hidden, recipe.layers, 0.1)
        res = train_model(
            model, Xtrn, ytr, utr, Xvan, yva, uva,
            R_va=R[va_idx], C_va=C[va_idx],
            epochs=recipe.epochs, lr=recipe.lr, loss_kind=recipe.loss,
            schedule="plateau", y_scale=recipe.y_scale, label=label, log_every=10,
            use_amp=recipe.use_amp,
            screen_stop=screen_stop, screen_leader_mae=screen_leader_mae,
        )
        assert_gpu_training(res["model"], label)
        pred, ep = res["pred_va"], res["best_epoch"]
    elif recipe.augment != AUG_NONE:
        model = BiLSTMReg(Xtrn.shape[-1], recipe.hidden, recipe.layers, 0.1)
        res = train_with_augment(
            model, Xtrn, ytr, utr, Xvan, yva, uva, u_idx, recipe.augment, seed,
            epochs=recipe.epochs, lr=recipe.lr, loss_kind=recipe.loss,
            y_scale=recipe.y_scale, label=label,
        )
        pred, ep = res["pred_va"], res["best_epoch"]
    else:
        model = BiLSTMReg(Xtrn.shape[-1], recipe.hidden, recipe.layers, 0.1)
        res = train_model(
            model, Xtrn, ytr, utr, Xvan, yva, uva,
            R_va=R[va_idx], C_va=C[va_idx],
            epochs=recipe.epochs, lr=recipe.lr, loss_kind=recipe.loss,
            schedule="plateau", y_scale=recipe.y_scale, label=label, log_every=10,
            use_amp=recipe.use_amp,
            screen_stop=screen_stop, screen_leader_mae=screen_leader_mae,
        )
        assert_gpu_training(res["model"], label)
        pred, ep = res["pred_va"], res["best_epoch"]

    val_mae = insp_mae(yva.reshape(-1), pred.reshape(-1), uva.reshape(-1))
    hard = insp_mae_by_rc(
        yva.reshape(-1), pred.reshape(-1), uva.reshape(-1),
        R[va_idx].reshape(-1), C[va_idx].reshape(-1),
    )
    del model, res
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "val_mae": float(val_mae),
        "hard_mae": float(hard),
        "best_epoch": int(ep),
        "pred_va": pred,
        "stopping_reason": res.get("stopping_reason"),
        "epochs_ran": res.get("epochs_ran"),
        "history": res.get("history"),
    }


def screen_recipe(
    feat_df,
    recipe: TrainRecipe,
    state_key: str,
    *,
    screen_seed: int = 42,
    val_frac: float = 0.2,
    use_screen_stop: bool = False,
) -> dict:
    """Single holdout screen; skips if already in batch3_state.json."""
    state = load_state()
    if screening_done(state, state_key):
        sc = state["screening"][state_key]
        print(f"  [SCREEN {state_key}] cached val={sc['val_insp_mae']:.4f}", flush=True)
        return sc

    X, y, u, R, C, bids = reshape_sequences(feat_df, SEQ_COLS_RICH)
    tr_ids, va_ids = breath_split(np.unique(bids), val_frac=val_frac, seed=screen_seed)
    tr_idx = np.flatnonzero(np.isin(bids, list(tr_ids)))
    va_idx = np.flatnonzero(np.isin(bids, list(va_ids)))
    train_seed = screen_seed
    stop_cfg = ScreenStopConfig() if use_screen_stop else None
    leader = screening_leader_mae(state) if use_screen_stop else None
    out = train_recipe_on_split(
        X, y, u, R, C, tr_idx, va_idx, recipe, train_seed,
        label=f"SCREEN:{state_key}",
        screen_stop=stop_cfg,
        screen_leader_mae=leader,
    )
    recipe_meta = {
        "name": recipe.name, "multitask": recipe.multitask, "residual": recipe.residual,
        "augment": recipe.augment, "hybrid_physics": recipe.hybrid_physics,
        "epochs": recipe.epochs, "lr": recipe.lr, "loss": recipe.loss,
    }
    curve_path = OUT / f"batch3_curve_{state_key}.json"
    curve_path.write_text(
        json.dumps({"history": out.get("history") or []}, indent=2, default=float),
        encoding="utf-8",
    )
    less_budget = bool(recipe.epochs and out.get("epochs_ran", recipe.epochs) < recipe.epochs)
    record_screening(
        state, state_key,
        val_mae=out["val_mae"], hard_mae=out["hard_mae"], best_epoch=out["best_epoch"],
        recipe=recipe_meta,
        extra={
            "stopping_reason": out.get("stopping_reason"),
            "epochs_ran": out.get("epochs_ran"),
            "epochs_budget": recipe.epochs,
            "less_than_full_budget": less_budget,
            "learning_curve_file": curve_path.name,
            "screen_stop_enabled": use_screen_stop,
        },
    )
    print(
        f"  [SCREEN {state_key}] val={out['val_mae']:.4f} hard50_10={out['hard_mae']:.4f} "
        f"(holdout {val_frac} seed={screen_seed})",
        flush=True,
    )
    return load_state()["screening"][state_key]


def subgroup_analysis(y, pred, u, R, C, u_out_flat=None):
    """MAE by (R,C), insp step, and contribution to total weighted MAE."""
    y_f, p_f = y.reshape(-1), pred.reshape(-1)
    u_f, r_f, c_f = u.reshape(-1), R.reshape(-1), C.reshape(-1)
    m = u_f == 0
    err = np.abs(y_f[m] - p_f[m])
    total = err.mean()
    rows = []
    for r in [5, 20, 50]:
        for c in [10, 20, 50]:
            mm = m & (r_f == r) & (c_f == c)
            if mm.sum() == 0:
                continue
            ma = np.abs(y_f[mm] - p_f[mm]).mean()
            sub_err = np.abs(y_f[mm] - p_f[mm]).sum()
            rows.append({
                "R": r, "C": c, "mae": float(ma), "count": int(mm.sum()),
                "contrib_pct": float(100.0 * sub_err / err.sum()) if err.sum() > 0 else 0.0,
            })
    # by insp step
    step_mae = {}
    bids_steps = u.reshape(-1, 80)
    for step in range(32):
        mm = m & ((np.arange(len(y_f)) % 80) == step)
        if mm.sum() > 0:
            step_mae[step] = float(np.abs(y_f[mm] - p_f[mm]).mean())
    return {"overall": float(total), "by_rc": rows, "by_step": step_mae}


def kfold_oof(
    feat_df,
    recipe: TrainRecipe,
    folds: list,
    seeds: list[int],
    cols=SEQ_COLS_RICH,
    verbose=True,
    checkpoint_key: str | None = None,
    checkpoint_path: Path | None = None,
) -> dict:
    X, y, u, R, C, bids = reshape_sequences(feat_df, cols)
    u_idx = cols.index("u_in")
    ckpt_path = checkpoint_path or CKPT_CV_PATH
    ckpt = {}
    if checkpoint_key and ckpt_path.exists():
        try:
            ckpt = json.loads(ckpt_path.read_text(encoding="utf-8"))
        except Exception:
            ckpt = {}
    oof_per_seed = []
    all_fold_scores = []
    for seed in seeds:
        set_global_seed(seed)
        oof = np.zeros_like(y)
        fold_scores = []
        seed_key = f"{checkpoint_key}_s{seed}" if checkpoint_key else None
        cached = ckpt.get(seed_key) if seed_key else None
        if cached and cached.get("complete_oof") and (OUT / f"batch3_oof_{checkpoint_key}.npz").exists():
            loaded = np.load(OUT / f"batch3_oof_{checkpoint_key}.npz")
            oof = loaded["oof"].astype(np.float32)
            fold_scores = cached.get("folds", [])
            oof_per_seed.append({
                "seed": seed, "oof_mae": float(cached["oof_mae"]),
                "oof": oof, "folds": fold_scores, "from_cache": True, "breath_id": bids,
            })
            all_fold_scores.extend(fold_scores)
            if verbose:
                print(f"  seed {seed} OOF={cached['oof_mae']:.4f} (cached)", flush=True)
            continue
        for fi, (tr_idx, va_idx) in enumerate(folds, 1):
            fold_npz = OUT / "batch3_cv_folds" / f"{seed_key}_f{fi}.npz" if seed_key else None
            if fold_npz is not None and fold_npz.exists():
                pack = np.load(fold_npz)
                oof[va_idx] = pack["pred"]
                fm = float(pack["mae"]) if "mae" in pack else insp_mae(
                    y[va_idx].reshape(-1), oof[va_idx].reshape(-1), u[va_idx].reshape(-1)
                )
                fold_scores.append(fm)
                if verbose:
                    print(f"    seed={seed} fold={fi} val={fm:.4f} (cached npz)", flush=True)
                continue
            Xtr, Xva = X[tr_idx], X[va_idx]
            ytr, yva = y[tr_idx], y[va_idx]
            utr, uva = u[tr_idx], u[va_idx]
            mean, std = norm_fit(Xtr)
            Xtrn, Xvan = apply_norm(Xtr, mean, std), apply_norm(Xva, mean, std)

            if recipe.hybrid_physics:
                vol = Xtr[:, :, cols.index("vol_over_c")]
                volr = Xtr[:, :, cols.index("vol_r_over_c")]
                urc = Xtr[:, :, cols.index("u_r_over_c")]
                uin = Xtr[:, :, u_idx]
                phys_tr = physics_pressure(vol, volr, urc, uin)
                vol_va = Xva[:, :, cols.index("vol_over_c")]
                phys_va = physics_pressure(
                    vol_va,
                    Xva[:, :, cols.index("vol_r_over_c")],
                    Xva[:, :, cols.index("u_r_over_c")],
                    Xva[:, :, u_idx],
                )
                ytr_res = ytr - phys_tr
                yva_res = yva - phys_va
                model = BiLSTMReg(Xtrn.shape[-1], recipe.hidden, recipe.layers, 0.1)
                res = train_model(
                    model, Xtrn, ytr_res, utr, Xvan, yva_res, uva,
                    R_va=R[va_idx], C_va=C[va_idx],
                    epochs=recipe.epochs, lr=recipe.lr, loss_kind=recipe.loss,
                    schedule="plateau", y_scale=recipe.y_scale,
                    label=f"{recipe.name}_s{seed}_f{fi}", log_every=20,
                )
                pred = res["pred_va"] + phys_va
            elif recipe.multitask:
                model = MultiTaskBiLSTM(Xtrn.shape[-1], recipe.hidden, recipe.layers, 0.1)
                res = train_multitask(
                    model, Xtrn, ytr, utr, Xvan, yva, uva,
                    R_va=R[va_idx], C_va=C[va_idx],
                    lr=recipe.lr, epochs=recipe.epochs, y_scale=recipe.y_scale,
                    aux_dp=recipe.aux_dp, aux_cum=recipe.aux_cum,
                    seed=seed, label=f"{recipe.name}_s{seed}_f{fi}",
                )
                pred = res["pred_va"]
            elif recipe.residual:
                model = ResidualBiLSTM(Xtrn.shape[-1], recipe.hidden, recipe.layers, 0.1)
                res = train_model(
                    model, Xtrn, ytr, utr, Xvan, yva, uva,
                    R_va=R[va_idx], C_va=C[va_idx],
                    epochs=recipe.epochs, lr=recipe.lr, loss_kind=recipe.loss,
                    schedule="plateau", y_scale=recipe.y_scale,
                    label=f"{recipe.name}_s{seed}_f{fi}", log_every=20,
                )
                pred = res["pred_va"]
            elif recipe.augment != AUG_NONE:
                model = BiLSTMReg(Xtrn.shape[-1], recipe.hidden, recipe.layers, 0.1)
                res = train_with_augment(
                    model, Xtrn, ytr, utr, Xvan, yva, uva, u_idx, recipe.augment, seed,
                    epochs=recipe.epochs, lr=recipe.lr, loss_kind=recipe.loss,
                    y_scale=recipe.y_scale, label=f"{recipe.name}_s{seed}_f{fi}",
                )
                pred = res["pred_va"]
            else:
                model = BiLSTMReg(Xtrn.shape[-1], recipe.hidden, recipe.layers, 0.1)
                res = train_model(
                    model, Xtrn, ytr, utr, Xvan, yva, uva,
                    R_va=R[va_idx], C_va=C[va_idx],
                    epochs=recipe.epochs, lr=recipe.lr, loss_kind=recipe.loss,
                    schedule="plateau", y_scale=recipe.y_scale,
                    label=f"{recipe.name}_s{seed}_f{fi}", log_every=20,
                )
                pred = res["pred_va"]

            oof[va_idx] = pred
            fm = insp_mae(yva.reshape(-1), pred.reshape(-1), uva.reshape(-1))
            fold_scores.append(fm)
            if verbose:
                print(f"    seed={seed} fold={fi} val={fm:.4f}", flush=True)
            if seed_key:
                wdir = OUT / "batch3_cv_weights"
                wdir.mkdir(exist_ok=True)
                torch.save(model.state_dict(), wdir / f"{seed_key}_f{fi}.pt")
            del model, res
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if seed_key:
                pdir = OUT / "batch3_cv_folds"
                pdir.mkdir(exist_ok=True)
                np.savez_compressed(
                    pdir / f"{seed_key}_f{fi}.npz",
                    pred=pred.astype(np.float32),
                    va_idx=va_idx.astype(np.int32),
                    breath_id=bids[va_idx],
                    mae=np.array(fm),
                )
                entry = ckpt.get(seed_key, {})
                partial = entry.get("partial_mae", {})
                partial[str(fi)] = fm
                ckpt[seed_key] = {"seed": seed, "partial_mae": partial, "folds_done": list(partial.keys())}
                ckpt_path.write_text(json.dumps(ckpt, indent=2, default=float), encoding="utf-8")

        oof_mae = insp_mae(y.reshape(-1), oof.reshape(-1), u.reshape(-1))
        oof_per_seed.append({"seed": seed, "oof_mae": oof_mae, "oof": oof.copy(), "folds": fold_scores, "breath_id": bids})
        if seed_key:
            ckpt[seed_key] = {
                "seed": seed, "oof_mae": oof_mae, "folds": fold_scores,
                "complete_oof": True,
                "oof_npz": f"batch3_oof_{checkpoint_key}.npz",
            }
            ckpt_path.write_text(json.dumps(ckpt, indent=2, default=float), encoding="utf-8")
        all_fold_scores.extend(fold_scores)
        if verbose:
            print(f"  seed {seed} OOF={oof_mae:.4f}", flush=True)

    pooled = np.mean([s["oof"] for s in oof_per_seed], axis=0) if len(oof_per_seed) > 1 else oof_per_seed[0]["oof"]
    pooled_mae = insp_mae(y.reshape(-1), pooled.reshape(-1), u.reshape(-1))
    return {
        "per_seed": oof_per_seed,
        "pooled_oof": pooled,
        "pooled_mae": pooled_mae,
        "seed_maes": [s["oof_mae"] for s in oof_per_seed],
        "seed_mean": float(np.mean([s["oof_mae"] for s in oof_per_seed])),
        "seed_std": float(np.std([s["oof_mae"] for s in oof_per_seed])),
        "fold_mean": float(np.mean(all_fold_scores)),
        "fold_std": float(np.std(all_fold_scores)),
        "y": y, "u": u, "R": R, "C": C,
    }
