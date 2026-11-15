"""
Batch 3 — Phase 1: holdout screening → promotions → STOP for approval.
Phase 2: 5-fold GroupKFold, seed 42 only (--run-cv-approved).
Multi-seed CV: --multi-seed-final <config_key> (final candidate confirmation only).

Model selection: OOF only. One gym submit; no leaderboard tuning.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch

from batch2_common import (
    DATA,
    OUT,
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
from batch3_baseline import document_baseline_audit, replay_batch2_holdout
from batch3_common import (
    AUG_MASK,
    AUG_NONE,
    AUG_UIN,
    BATCH2_OOF,
    TrainRecipe,
    kfold_oof,
    make_folds,
    screen_recipe,
    set_global_seed,
    snap_lattice,
    SEQ_COLS_RICH,
)
from batch3_state import (
    CV_CONTROLS,
    CV_MULTI_SEED,
    CV_SEED_DEFAULT,
    PRIMARY_SCREEN_KEYS,
    PROMOTE_DELTA,
    TOP_K_SCREEN,
    decide_promotions,
    load_state,
    record_cv,
    save_state,
    set_cv_approved,
    write_screening_report,
)

lines: list[str] = []
FALLBACK = {"name": "batch2_bilstm_rich", "oof_mae": BATCH2_OOF}
SCREEN_SEED = 42
VAL_FRAC = 0.2


def log(msg: str = "") -> None:
    print(msg, flush=True)
    lines.append(msg)


def fit_physics_fold(Xtr, ytr, utr, cols):
    idx_v, idx_vr, idx_ur, idx_u = (
        cols.index("vol_over_c"), cols.index("vol_r_over_c"),
        cols.index("u_r_over_c"), cols.index("u_in"),
    )
    flat = utr.reshape(-1) == 0
    Xf = np.stack([
        Xtr.reshape(-1, Xtr.shape[-1])[flat, idx_v],
        Xtr.reshape(-1, Xtr.shape[-1])[flat, idx_vr],
        Xtr.reshape(-1, Xtr.shape[-1])[flat, idx_ur],
        Xtr.reshape(-1, Xtr.shape[-1])[flat, idx_u],
        np.ones(flat.sum()),
    ], axis=1)
    coef, *_ = np.linalg.lstsq(Xf, ytr.reshape(-1)[flat], rcond=None)

    def predict(X):
        A = np.stack([
            X[:, :, idx_v].reshape(-1),
            X[:, :, idx_vr].reshape(-1),
            X[:, :, idx_ur].reshape(-1),
            X[:, :, idx_u].reshape(-1),
            np.ones(X.shape[0] * X.shape[1]),
        ], axis=1)
        return (A @ coef).reshape(X.shape[0], X.shape[1]).astype(np.float32)

    return predict


def screen_e5_hybrid(feat, screen_seed=SCREEN_SEED, val_frac=VAL_FRAC, use_screen_stop: bool = False):
    key = "e5_hybrid"
    state = load_state()
    if state.get("screening", {}).get(key, {}).get("done"):
        return state["screening"][key]
    X, y, u, R, C, bids = reshape_sequences(feat, SEQ_COLS_RICH)
    tr_ids, va_ids = breath_split(np.unique(bids), val_frac, screen_seed)
    tr_idx = np.flatnonzero(np.isin(bids, list(tr_ids)))
    va_idx = np.flatnonzero(np.isin(bids, list(va_ids)))
    pred_fn = fit_physics_fold(X[tr_idx], y[tr_idx], u[tr_idx], SEQ_COLS_RICH)
    phys_va = pred_fn(X[va_idx])
    phys_mae = insp_mae(y[va_idx].reshape(-1), phys_va.reshape(-1), u[va_idx].reshape(-1))
    mean, std = norm_fit(X[tr_idx])
    Xtrn = apply_norm(X[tr_idx], mean, std)
    Xvan = apply_norm(X[va_idx], mean, std)
    ytr_res = y[tr_idx] - pred_fn(X[tr_idx])
    set_global_seed(screen_seed)
    model = BiLSTMReg(Xtrn.shape[-1], 256, 3, 0.1)
    from batch3_screen_policy import ScreenStopConfig, screening_leader_mae
    stop_cfg = ScreenStopConfig() if use_screen_stop else None
    leader = screening_leader_mae(state) if use_screen_stop else None
    res = train_model(
        model, Xtrn, ytr_res, u[tr_idx], Xvan, y[va_idx] - phys_va, u[va_idx],
        R_va=R[va_idx], C_va=C[va_idx],
        epochs=75, lr=3e-3, loss_kind="mse", schedule="plateau", y_scale=True,
        label="SCREEN:e5_hybrid", log_every=10,
        screen_stop=stop_cfg, screen_leader_mae=leader,
    )
    pred = phys_va + res["pred_va"]
    val_mae = insp_mae(y[va_idx].reshape(-1), pred.reshape(-1), u[va_idx].reshape(-1))
    hard = insp_mae_by_rc(
        y[va_idx].reshape(-1), pred.reshape(-1), u[va_idx].reshape(-1),
        R[va_idx].reshape(-1), C[va_idx].reshape(-1),
    )
    from batch3_state import record_screening
    record_screening(
        state, key, val_mae=val_mae, hard_mae=float(hard), best_epoch=res["best_epoch"],
        recipe={"hybrid_physics": True},
    )
    state = load_state()
    state["screening"][key]["phys_only_mae"] = float(phys_mae)
    save_state(state)
    log(f"  [SCREEN {key}] hybrid={val_mae:.4f} hard50_10={hard:.4f} phys_only={phys_mae:.4f}")
    return load_state()["screening"][key]


def run_phase2_cv(
    feat, folds, key_to_recipe, cv_seeds: list[int],
    multi_key: str | None = None,
    queue_override: list[str] | None = None,
):
    state = load_state()
    if queue_override:
        queue = list(queue_override)
    elif multi_key:
        queue = [multi_key]
    else:
        queue = state.get("cv_queue") or []
    if multi_key and multi_key not in state.get("screening", {}):
        raise SystemExit(f"Unknown config {multi_key}")
    log("\n" + "=" * 72)
    label = "MULTI_SEED_CONFIRM" if multi_key else "GROUP_KFOLD_CV"
    log(f"PHASE 2 — {label} (K=5, seeds={cv_seeds})")
    log("=" * 72)
    for key in queue:
        ck = f"{key}_multiseed" if multi_key else key
        prev = state.get("cv_results", {}).get(ck, {})
        if prev.get("complete") and prev.get("seeds") == cv_seeds:
            log(f"  [CV {ck}] cached OOF={prev['oof_mae']:.4f}")
            continue
        recipe = key_to_recipe.get(key)
        if recipe is None:
            log(f"  skip {key}: no recipe")
            continue
        log(f"\n--- CV: {key} ---")
        out = kfold_oof(feat, recipe, folds, cv_seeds, verbose=True, checkpoint_key=ck)
        oof_npz = OUT / f"batch3_oof_{ck}.npz"
        X, y, u, R, C, bids = reshape_sequences(feat, SEQ_COLS_RICH)
        oof = out["per_seed"][0]["oof"] if len(cv_seeds) == 1 else out["pooled_oof"]
        np.savez_compressed(
            oof_npz,
            breath_id=bids,
            oof=oof.astype(np.float32),
            y=y.astype(np.float32),
            u_out=u.astype(np.float32),
            R=R.astype(np.float32),
            C=C.astype(np.float32),
            **{f"seed_{s['seed']}": s["oof"].astype(np.float32) for s in out["per_seed"]},
        )
        payload = {
            "complete": True,
            "oof_mae": out["pooled_mae"],
            "seed_mean": out["seed_mean"],
            "seed_std": out["seed_std"],
            "fold_mean": out["fold_mean"],
            "fold_std": out["fold_std"],
            "per_seed_mae": out["seed_maes"],
            "seeds": cv_seeds,
            "oof_npz": str(oof_npz.name),
            "fold_scores": out["per_seed"][0]["folds"] if out["per_seed"] else [],
            "n_folds": 5,
            "selection_note": "Compare OOF vs Batch2 0.2506; do not use leaderboard for selection",
        }
        if len(cv_seeds) == 1:
            payload["oof_mae_primary"] = out["per_seed"][0]["oof_mae"]
        state = load_state()
        record_cv(state, ck, payload)
        log(f"  [CV {ck}] OOF={out['pooled_mae']:.4f} per_seed={out['seed_maes']}")


def main():
    parser = argparse.ArgumentParser(description="Batch 3 ventilator pressure")
    parser.add_argument(
        "--run-cv-approved",
        action="store_true",
        help="Phase 2: 5-fold × seed 42 for promoted configs (requires explicit flag)",
    )
    parser.add_argument(
        "--skip-screening",
        action="store_true",
        help="Only promotions/report/CV (use when screening already in state)",
    )
    parser.add_argument(
        "--baseline-replay",
        action="store_true",
        help="After e1 screen: optional Batch2-matched holdout (90 ep, no AMP)",
    )
    parser.add_argument(
        "--multi-seed-final",
        metavar="KEY",
        default=None,
        help=f"Multi-seed CV for one finalist only (e.g. e1_baseline); seeds {CV_MULTI_SEED}",
    )
    parser.add_argument(
        "--only-screen",
        action="append",
        default=[],
        help="Run only these screen keys (e.g. e5_hybrid); others skip if cached",
    )
    parser.add_argument(
        "--cv-keys",
        nargs="+",
        default=None,
        help="Limit Phase 2 to these configs (e.g. e1_baseline e3_aug_uin)",
    )
    args = parser.parse_args()
    only = set(args.only_screen)

    if args.multi_seed_final and not args.run_cv_approved:
        log("ERROR: --multi-seed-final requires --run-cv-approved")
        raise SystemExit(1)

    t0 = time.time()
    log("BATCH 3 (screen → promote → [await approval] → CV)")
    log(f"Batch2 reference OOF {BATCH2_OOF} | holdout ~0.239 (see batch3_baseline_audit.json)")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        # cuDNN LSTM has aborted twice on this GPU; native CUDA LSTM is slower but stable
        torch.backends.cudnn.enabled = False
        torch.cuda.empty_cache()
        log(f"GPU: {torch.cuda.get_device_name(0)} (cudnn disabled for LSTM stability)")
    else:
        log("WARNING: CUDA not available")

    state = load_state()
    state["protocol_notes"] = {
        "promote_delta": PROMOTE_DELTA,
        "top_k_explore": TOP_K_SCREEN,
        "cv_controls": list(CV_CONTROLS),
        "phase2_seeds_default": CV_SEED_DEFAULT,
        "no_auto_multi_seed": True,
        "final_model_by_oof_only": True,
        "single_gym_submit": True,
    }
    state["screen_split"] = {"val_frac": VAL_FRAC, "seed": SCREEN_SEED}
    save_state(state)

    train = pd.read_csv(DATA / "train.csv")
    feat = add_features(train)
    folds = make_folds(reshape_sequences(feat, SEQ_COLS_RICH)[-1], 5)

    screen_specs = [
        ("e1_baseline", TrainRecipe(name="baseline", epochs=75, lr=3e-3, loss="mse", y_scale=True)),
        ("e2_multitask", TrainRecipe(name="multitask", multitask=True, epochs=75, aux_dp=0.25, aux_cum=0.15)),
        ("e3_aug_none", TrainRecipe(name="aug_none", augment=AUG_NONE, epochs=75)),
        ("e3_aug_uin", TrainRecipe(name="aug_uin", augment=AUG_UIN, epochs=75)),
        ("e3_aug_mask", TrainRecipe(name="aug_mask", augment=AUG_MASK, epochs=75)),
        ("e4_residual", TrainRecipe(name="residual", residual=True, epochs=85)),
    ]
    key_to_recipe = dict(screen_specs)
    key_to_recipe["e5_hybrid"] = TrainRecipe(name="hybrid", hybrid_physics=True, epochs=75, lr=3e-3, y_scale=True)

    if not args.skip_screening and not args.run_cv_approved:
        log("\n" + "=" * 72)
        log("PHASE 1 — SCREENING (holdout 20%, split_seed=42, train_seed=42)")
        log("=" * 72)
        if not only:
            for key, recipe in screen_specs:
                screen_recipe(
                    feat, recipe, key, screen_seed=SCREEN_SEED, val_frac=VAL_FRAC,
                    use_screen_stop=False,
                )
            screen_e5_hybrid(feat, use_screen_stop=True)
        else:
            for key, recipe in screen_specs:
                if key in only:
                    screen_recipe(
                        feat, recipe, key, screen_seed=SCREEN_SEED, val_frac=VAL_FRAC,
                        use_screen_stop=False,
                    )
            if "e5_hybrid" in only:
                screen_e5_hybrid(feat, use_screen_stop=True)

    state = load_state()
    audit = document_baseline_audit(state)
    if audit["batch3_e1_screen"].get("val_insp_mae") is not None:
        d = audit["known_deltas_vs_batch2_holdout"]["holdout_mae"]
        log(f"\nBaseline check: e1 screen Δ holdout vs Batch2 = {d:+.4f} (AMP on, ep75)")
    if args.baseline_replay:
        log("\nBaseline replay (Batch2 recipe: 90 ep, no AMP)...")
        rep = replay_batch2_holdout(feat)
        log(f"  replay holdout={rep['val_insp_mae']:.4f} Δ vs 0.239 = {rep.get('delta_vs_batch2_holdout', 0):+.4f}")

    queue = decide_promotions(load_state(), cv_seeds=CV_SEED_DEFAULT)
    report = write_screening_report(load_state())
    log("\n" + report)

    if args.run_cv_approved:
        set_cv_approved(load_state(), True)
        seeds = CV_MULTI_SEED if args.multi_seed_final else CV_SEED_DEFAULT
        cv_keys = args.cv_keys
        log(f"\nStarting GROUP_KFOLD_CV keys={cv_keys or queue} seeds={seeds}")
        run_phase2_cv(
            feat, folds, key_to_recipe, seeds,
            multi_key=args.multi_seed_final,
            queue_override=cv_keys,
        )
        state = load_state()
        log("\nGROUP_KFOLD_CV results:")
        for k, cv in state.get("cv_results", {}).items():
            log(f"  {k}: OOF={cv.get('oof_mae')} fold_std={cv.get('fold_std')} seeds={cv.get('seeds')}")
    else:
        log(f"\nCV NOT started (await your approval). Planned runs: {load_state()['cv_plan'].get('estimated_training_runs')}")
        log("\nTo run Phase 2 after review: python -u research/batch3_run.py --run-cv-approved --skip-screening")

    state = load_state()
    out = {
        "fallback_batch2": FALLBACK,
        "screening": state.get("screening", {}),
        "baseline_audit": state.get("baseline_audit", {}),
        "cv_plan": state.get("cv_plan", {}),
        "cv_results": state.get("cv_results", {}),
        "promotions": state.get("promotions", {}),
        "primary_screen_keys": list(PRIMARY_SCREEN_KEYS),
        "elapsed_sec": time.time() - t0,
    }
    (OUT / "batch3_results.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    (OUT / "batch3_report.txt").write_text("\n".join(lines), encoding="utf-8")
    log("Wrote batch3_screening_report.txt + batch3_results.json")


if __name__ == "__main__":
    main()