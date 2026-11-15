"""Autonomous 3-stage runner: probe → holdout PF → (optional 2nd seed) → ensemble → PI.

Does not touch Batch 2/3 artifacts. Gym submit is a separate explicit step.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipe_common import E1_HOLD, PIPE, load_scoreboard
from pf_model import PF_SMALL, PF_TINY
from stage1_pf import run_holdout, run_probe
from stage2_ensemble import eval_ensembles
from stage3_pi import run_overlay
from pipe_common import update_track, insp_mae
import numpy as np


def maybe_write_ensemble_from_holdouts():
    files = sorted(PIPE.glob("stage1_*_holdout.npz"))
    if len(files) < 2:
        print(f"ensemble: {len(files)} holdout npz (need ≥2) — skip")
        return None
    packs = [np.load(f) for f in files]
    # align on breath_id
    bids0 = packs[0]["breath_id"]
    preds = []
    for p in packs:
        if not np.array_equal(p["breath_id"], bids0):
            print("ensemble: breath_id mismatch — skip")
            return None
        preds.append(p["pred"])
    preds = np.stack(preds, axis=0)
    y, u, R, C = packs[0]["y"], packs[0]["u_out"], packs[0]["R"], packs[0]["C"]
    np.savez_compressed(PIPE / "ensemble_inputs.npz", preds=preds, y=y, u_out=u, R=R, C=C, breath_id=bids0)
    res = eval_ensembles(preds, y, u, R, C)
    (PIPE / "stage2_ensemble.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    update_track("neural_ensemble", {"status": "done", "n_models": int(preds.shape[0]), **res})
    print("ensemble best", res["best"], res["best_mae"])
    return res


def main():
    PIPE.mkdir(exist_ok=True)
    probe_path = PIPE / "stage1_probe.json"
    if not probe_path.exists():
        print("=== STAGE 1 PROBE ===", flush=True)
        est = run_probe(n_breaths=6000, epochs=2, width=PF_TINY, batch_size=256)
    else:
        est = json.loads(probe_path.read_text(encoding="utf-8"))
        print("probe cached", est.get("est_holdout_60ep_min"), "min (batch-64 tiny; holdout uses 256+AMP)", flush=True)

    # 4060: tiny + batch 256 + AMP. Full PF_SMALL is ~4× and only if tiny holdout is strong.
    width, epochs, tag, bs = PF_TINY, 40, "pf_tiny", 256
    print(f"holdout plan: {tag} ep={epochs} bs={bs}", flush=True)

    h1 = PIPE / f"stage1_{tag}_s42.json"
    if not h1.exists():
        print(f"=== STAGE 1 HOLDOUT {tag} ===", flush=True)
        rec = run_holdout(width=width, epochs=epochs, batch_size=bs, seed=42, tag=tag)
    else:
        rec = json.loads(h1.read_text(encoding="utf-8"))
        print("holdout cached", rec.get("val_insp_mae"), flush=True)

    mae = rec["val_insp_mae"]
    # second seed only if clearly better than e1
    if mae <= E1_HOLD - 0.005:
        h2 = PIPE / f"stage1_{tag}_s0.json"
        if not h2.exists():
            print("=== STAGE 1 second seed 0 ===", flush=True)
            run_holdout(width=width, epochs=epochs, batch_size=bs, seed=0, tag=tag)
        maybe_write_ensemble_from_holdouts()
    else:
        print(f"PF holdout {mae:.4f} did not beat e1-0.005 — skip extra PF seeds", flush=True)
        update_track(
            "neural_ensemble",
            {"status": "skipped", "reason": "only one PF model; adding seeds not justified", "pf_holdout": mae},
        )

    npz = PIPE / f"stage1_{tag}_s42_holdout.npz"
    # PI: 400-breath cost gate first, then scale if fast
    costp = PIPE / "pi_cost_pi400_cost_gate.json"
    overlay_json = PIPE / "pi_s42_overlay.json"
    if npz.exists() and not overlay_json.exists():
        print("=== STAGE 3 PI cost sample (400 breaths, p0=40) ===", flush=True)
        sample = run_overlay(str(npz), p0_k=40, max_breaths=400, tag="pi_cost400")
        med_ms = sample.get("med_ms_per_breath") or 1e9
        n_va = rec.get("n_va", 13581)
        est_min = med_ms * n_va / 1000 / 60
        print(f"PI full-val estimate {est_min:.1f} min", flush=True)
        if est_min <= 90:
            print("=== STAGE 3 PI full holdout ===", flush=True)
            run_overlay(str(npz), p0_k=80, max_breaths=None, tag="pi_s42")
        else:
            print("PI full val too slow — keep 400-breath measurement", flush=True)
            update_track(
                "inverse_pi_neural_fallback",
                {**sample, "note": f"full val est {est_min:.0f} min; kept 400-breath sample"},
            )

    sb = load_scoreboard()
    (PIPE / "PIPELINE_REPORT.txt").write_text(json.dumps(sb, indent=2, default=float), encoding="utf-8")
    print("=== SCOREBOARD ===")
    print(json.dumps(sb["tracks"], indent=2, default=float))


if __name__ == "__main__":
    main()
