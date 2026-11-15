"""
Small-scale inverse P-controller proof of concept (CPU).
Uses only authorized train.csv. No GPU; does not touch Batch 3 processes.
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
    CLIP_EPS,
    PRESSURE_STEP,
    P_COEF_GRID,
    P_STAR_GRID,
    clip_mask,
    lattice_residual,
    linear_p_fit,
    p_params_from_linear,
    snap_lattice,
)

OUT = Path(__file__).resolve().parent
LATTICE_TOL = PRESSURE_STEP * 0.51
MAX_BREATHS = 4000  # PoC subset
SEED = 42
VAL_FRAC = 0.2


def breath_arrays(df: pd.DataFrame, bids: np.ndarray) -> dict:
    sub = df[df["breath_id"].isin(bids)].sort_values(["breath_id", "time_step"])
    n = len(bids)
    y = sub["pressure"].to_numpy(np.float32).reshape(n, 80)
    uin = sub["u_in"].to_numpy(np.float32).reshape(n, 80)
    uout = sub["u_out"].to_numpy(np.float32).reshape(n, 80)
    R = sub["R"].to_numpy(np.float32).reshape(n, 80)[:, 0]
    C = sub["C"].to_numpy(np.float32).reshape(n, 80)[:, 0]
    return {"y": y, "uin": uin, "uout": uout, "R": R, "C": C, "ids": bids}


def identify_p_one_breath(
    uin_1d: np.ndarray,
    y_1d: np.ndarray,
    uout_1d: np.ndarray,
) -> dict:
    insp = uout_1d == 0
    u = uin_1d[insp]
    p = y_1d[insp]
  # exclude clips
    m = ~clip_mask(u)
    u, p = u[m], p[m]
    n_insp = int(insp.sum())
    n_use = len(u)
    if n_use < 5:
        return {
            "ok": False, "reason": "too_few_steps", "n_insp": n_insp, "n_use": n_use,
            "linear_r2": 0.0, "grid_hit_frac": 0.0,
        }

    # Linear fit diagnostic (P: negative slope expected)
    a, b, r2 = linear_p_fit(u, p)
    p_coef_lr, p_star_lr = p_params_from_linear(a, b)

    # Vectorized grid: p_hat = p_star - u/p_coef
    best = {"hits": 0, "mae": np.inf, "p_coef": None, "p_star": None}
    for pc in P_COEF_GRID:
        if pc < 1e-9:
            continue
        p_hat = P_STAR_GRID[:, None] - u[None, :] / pc  # (6, n_use)
        resid = lattice_residual(p_hat)
        hits = (resid <= LATTICE_TOL).sum(axis=1)
        mae_axis = np.mean(np.abs(p_hat - p[None, :]), axis=1)
        for j in range(len(P_STAR_GRID)):
            if hits[j] > best["hits"] or (hits[j] == best["hits"] and mae_axis[j] < best["mae"]):
                best = {
                    "hits": int(hits[j]),
                    "mae": float(mae_axis[j]),
                    "p_coef": float(pc),
                    "p_star": float(P_STAR_GRID[j]),
                    "hit_frac": float(hits[j] / n_use),
                }

    p_hat_best = best["p_star"] - u / best["p_coef"]
    p_snap = snap_lattice(p_hat_best)
    step_ok = lattice_residual(p_hat_best) <= LATTICE_TOL
    recon_mae = float(np.mean(np.abs(p_snap[step_ok] - p[step_ok]))) if step_ok.any() else np.nan

    return {
        "ok": best["hits"] >= n_use * 0.5,
        "n_insp": n_insp,
        "n_use": n_use,
        "clip_frac": float(1.0 - n_use / max(n_insp, 1)),
        "linear_r2": r2,
        "linear_p_coef": p_coef_lr,
        "linear_p_star": p_star_lr,
        "grid_p_coef": best["p_coef"],
        "grid_p_star": best["p_star"],
        "grid_hit_frac": best.get("hit_frac", 0.0),
        "grid_mae_all": best["mae"],
        "recon_mae_lattice_ok": recon_mae,
    }


def predict_p_inverse(uin_1d: np.ndarray, uout_1d: np.ndarray, p_coef: float, p_star: float) -> tuple[np.ndarray, np.ndarray]:
    """Return full-length pred and mask where inverse trusted (insp, not clipped, lattice ok)."""
    pred = np.full_like(uin_1d, np.nan, dtype=np.float64)
    trusted = np.zeros_like(uin_1d, dtype=bool)
    insp = uout_1d == 0
    for i in np.where(insp)[0]:
        u = uin_1d[i]
        if clip_mask(np.array([u]))[0]:
            continue
        ph = p_star - u / p_coef
        if lattice_residual(np.array([ph]))[0] <= LATTICE_TOL:
            pred[i] = snap_lattice(np.array([ph]))[0]
            trusted[i] = True
    return pred, trusted


def main():
    t0 = time.time()
    log_lines = []
    log_lines.append("INVERSE P-CONTROLLER PoC (subset)")
    log_lines.append(f"max_breaths={MAX_BREATHS} split seed={SEED}")

    df = pd.read_csv(DATA / "train.csv")
    all_bids = df["breath_id"].unique()
    rng = np.random.default_rng(SEED)
    if len(all_bids) > MAX_BREATHS:
        all_bids = rng.choice(all_bids, size=MAX_BREATHS, replace=False)
    tr_set, va_set = breath_split(all_bids, val_frac=VAL_FRAC, seed=SEED)
    tr_ids = np.array(sorted(tr_set))
    va_ids = np.array(sorted(va_set))
    tr = breath_arrays(df, tr_ids)
    va = breath_arrays(df, va_ids)

    # --- Identify on TRAIN (labels allowed for ID quality) ---
    tr_stats = []
    for i in range(len(tr_ids)):
        st = identify_p_one_breath(tr["uin"][i], tr["y"][i], tr["uout"][i])
        st["breath_id"] = int(tr_ids[i])
        st["R"], st["C"] = float(tr["R"][i]), float(tr["C"][i])
        tr_stats.append(st)
    tr_ok = [s for s in tr_stats if s.get("ok")]
    log_lines.append(f"Train breaths: {len(tr_ids)}; P-grid ok (≥50% lattice hits): {len(tr_ok)}")

    def grid_identify_uin_only(uin_1d: np.ndarray, uout_1d: np.ndarray) -> tuple[float, float, np.ndarray, np.ndarray]:
        """Test-time rule: u_in only + lattice (writeup-style; can false-positive)."""
        insp = uout_1d == 0
        u = uin_1d[insp]
        m = ~clip_mask(u)
        u = u[m]
        best_hits, best_pc, best_ps = 0, P_COEF_GRID[1], P_STAR_GRID[0]
        for pc in P_COEF_GRID:
            if pc < 1e-9:
                continue
            p_hat = P_STAR_GRID[:, None] - u[None, :] / pc
            hits = (lattice_residual(p_hat) <= LATTICE_TOL).sum(axis=1)
            j = int(np.argmax(hits))
            if int(hits[j]) > best_hits:
                best_hits, best_pc, best_ps = int(hits[j]), float(pc), float(P_STAR_GRID[j])
        pred, trusted = predict_p_inverse(uin_1d, uout_1d, best_pc, best_ps)
        return best_pc, best_ps, pred, trusted

    def grid_identify_oracle(uin_1d: np.ndarray, y_1d: np.ndarray, uout_1d: np.ndarray) -> tuple[float, float, np.ndarray, np.ndarray]:
        """Eval-only: pick (p_coef, p_star) minimizing MAE vs labels on non-clipped insp steps."""
        st = identify_p_one_breath(uin_1d, y_1d, uout_1d)
        if st.get("grid_p_coef") is None:
            pred = np.full_like(y_1d, np.nan)
            return 1.0, 20.0, pred, np.zeros_like(uout_1d, dtype=bool)
        pred, trusted = predict_p_inverse(uin_1d, uout_1d, st["grid_p_coef"], st["grid_p_star"])
        return st["grid_p_coef"], st["grid_p_star"], pred, trusted

    val_preds_u = np.full_like(va["y"], np.nan)
    val_trusted_u = np.zeros_like(va["y"], dtype=bool)
    val_preds_o = np.full_like(va["y"], np.nan)
    val_trusted_o = np.zeros_like(va["y"], dtype=bool)
    for i in range(len(va_ids)):
        uin, uout, y = va["uin"][i], va["uout"][i], va["y"][i]
        _, _, pred_u, trust_u = grid_identify_uin_only(uin, uout)
        val_preds_u[i], val_trusted_u[i] = pred_u, trust_u
        _, _, pred_o, trust_o = grid_identify_oracle(uin, y, uout)
        val_preds_o[i], val_trusted_o[i] = pred_o, trust_o

    def val_metrics(pred_2d, trust_2d, tag: str) -> dict:
        y_flat = va["y"].reshape(-1)
        uout_flat = va["uout"].reshape(-1)
        pred_flat = pred_2d.reshape(-1)
        trust_flat = trust_2d.reshape(-1)
        insp = uout_flat == 0
        trusted_insp = insp & trust_flat & np.isfinite(pred_flat)
        n_insp = int(insp.sum())
        n_trusted = int(trusted_insp.sum())
        coverage_pct = 100.0 * n_trusted / max(n_insp, 1)
        mae_trusted = (
            float(np.mean(np.abs(y_flat[trusted_insp] - pred_flat[trusted_insp])))
            if n_trusted > 0
            else float("nan")
        )
        mean_insp = float(np.mean(y_flat[insp]))
        hybrid = np.where(trusted_insp, pred_flat, mean_insp)
        hybrid_mae = insp_mae(y_flat, hybrid, uout_flat)
        full_mae_trusted_rows = float(insp_mae(y_flat, pred_flat, uout_flat)) if n_trusted else float("nan")
        return {
            "tag": tag,
            "insp_step_coverage_pct": coverage_pct,
            "mae_on_trusted_steps": mae_trusted,
            "hybrid_mean_fallback_mae": hybrid_mae,
            "all_insp_mae_where_pred_finite": full_mae_trusted_rows,
            "n_trusted_steps": n_trusted,
            "n_insp_steps": n_insp,
        }

    val_u = val_metrics(val_preds_u, val_trusted_u, "uin_only_lattice")
    val_o = val_metrics(val_preds_o, val_trusted_o, "oracle_label_id_eval_only")

    # Train: oracle eval if we trust grid on train
    tr_cov, tr_mae = [], []
    for i, st in enumerate(tr_stats):
        if not st.get("grid_p_coef"):
            continue
        pred, trusted = predict_p_inverse(tr["uin"][i], tr["uout"][i], st["grid_p_coef"], st["grid_p_star"])
        m = (tr["uout"][i] == 0) & trusted
        if m.any():
            tr_mae.append(float(np.mean(np.abs(tr["y"][i][m] - pred[m]))))
            tr_cov.append(m.sum() / (tr["uout"][i] == 0).sum())
    train_coverage = float(np.mean(tr_cov)) if tr_cov else 0.0
    train_mae_trusted = float(np.mean(tr_mae)) if tr_mae else float("nan")

    elapsed = time.time() - t0
    results = {
        "hypothesis": "inverse P controller (AmbrosM); grids from writeup",
        "subset": {"n_breaths": len(all_bids), "train": len(tr_ids), "val": len(va_ids)},
        "train_ident": {
            "p_grid_ok_breaths": len(tr_ok),
            "frac_ok": len(tr_ok) / len(tr_ids),
            "mean_hit_frac_ok": float(np.mean([s["grid_hit_frac"] for s in tr_ok])) if tr_ok else None,
            "mean_linear_r2": float(np.nanmean([s["linear_r2"] for s in tr_stats])),
            "coverage_trusted_steps": train_coverage,
            "mae_on_trusted_steps": train_mae_trusted,
        },
        "val_uin_only": val_u,
        "val_oracle_id": val_o,
        "warning": "uin_only lattice self-check can false-positive; use oracle block as ID ceiling",
        "vs_neural_baseline_note": "Batch2 OOF 0.2506; this PoC is NOT comparable (subset, P-only, mean fallback)",
        "runtime_sec": elapsed,
        "next_pi": "deferred pending review; T=0.5 hypothesis from writeup",
    }

    OUT.joinpath("poc_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    log_lines.extend([
        "",
        f"Val ORACLE ID — coverage {val_o['insp_step_coverage_pct']:.1f}%, "
        f"MAE trusted {val_o['mae_on_trusted_steps']:.4f}, hybrid {val_o['hybrid_mean_fallback_mae']:.4f}",
        f"Val U_IN only — coverage {val_u['insp_step_coverage_pct']:.1f}%, "
        f"MAE trusted {val_u['mae_on_trusted_steps']:.4f} (unreliable if MAE huge)",
        f"Train P-ID ok breaths: {len(tr_ok)}/{len(tr_ids)}",
        f"Runtime: {elapsed:.1f}s",
    ])
    OUT.joinpath("poc_report.txt").write_text("\n".join(log_lines), encoding="utf-8")

    append_log = OUT / "experiment_log.md"
    if not append_log.exists():
        append_log.write_text("# Inverse control experiment log\n\n", encoding="utf-8")
    append_log.write_text(
        f"\n## PoC-P1 ({time.strftime('%Y-%m-%d %H:%M')})\n"
        f"- Hypothesis: discrete P-controller inverse matches lattice\n"
        f"- Val oracle: cov {val_o['insp_step_coverage_pct']:.1f}%, "
        f"MAE {val_o['mae_on_trusted_steps']:.4f}, hybrid {val_o['hybrid_mean_fallback_mae']:.4f}\n"
        f"- Runtime: {elapsed:.1f}s\n",
        encoding="utf-8",
    )
    print("\n".join(log_lines))
    return results


if __name__ == "__main__":
    main()