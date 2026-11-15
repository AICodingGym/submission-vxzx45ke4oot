# Batch 3 closed — stop for review

**Validated neural best (this batch):** `e3_aug_uin` GroupKFold OOF **0.2519** (seed 42, K=5).  
**Control:** `e1_baseline` OOF **0.2760** (same protocol, AMP-era).  
**Historical Batch 2 fallback:** rich BiLSTM OOF **0.2506** (90 ep, no AMP).  
**Selection rule:** OOF only. No gym submit from this wrap-up. No extra GPU training after this file.

## Phase 1 screening (holdout 80/20, seed 42) — complete

| Config | Holdout MAE | Hard 50/10 |
|--------|-------------|------------|
| e1_baseline | 0.2808 | 0.3119 |
| e2_multitask | 0.3531 | 0.4154 |
| e3_aug_none | 0.2808 | 0.3119 |
| **e3_aug_uin** | **0.2616** | **0.2837** |
| e3_aug_mask | 0.3444 | 0.3676 |
| e4_residual | 0.3370 | 0.4002 |
| e5_hybrid | 0.3308 | 0.3998 |

`e3_aug_uin` beat `e1_baseline` by **0.0192** overall / **0.0282** hard. e1 holdout **0.2808 ≠ 0.25**, so AMP-era control CV was required.

## Phase 2 — 5-fold GroupKFold, seed 42 only

| | e1_baseline (control) | e3_aug_uin |
|--|----------------------|------------|
| Fold 1 | 0.2875 | 0.2448 |
| Fold 2 | 0.2516 | 0.2526 |
| Fold 3 | 0.2950 | 0.2554 |
| Fold 4 | 0.2883 | 0.2562 |
| Fold 5 | 0.2576 | 0.2502 |
| **OOF** | **0.2760** | **0.2519** |
| fold mean ± std | 0.2760 ± **0.0178** | 0.2518 ± **0.0041** |

vs Batch 2 OOF **0.2506**: `e3_aug_uin` is **+0.0013** (within fold noise of Batch 2 ±0.019; **not** a clear OOF win). vs this-batch control **0.2760**: **−0.0241** (clear win on the AMP 75-ep protocol).

**Preserve for hybrid fallback:** `research/batch3_oof_e3_aug_uin.npz` (`breath_id`, `oof`, `y`, `u_out`, `R`, `C`). Control: `batch3_oof_e1_baseline.npz`. Fold weights: `research/batch3_cv_weights/`.

If one model must be “the” Batch 3 neural: **`e3_aug_uin`**. If matching Batch 2’s published 0.2506 exactly, keep Batch 2 as the named historical fallback; they are essentially tied on OOF.

## Not started (by design)

- Multi-seed CV
- CV for e2 / e3_aug_none / e4 / e5 / mask
- Gym submit
- Inverse PI scale-up / Pulp Fiction GPU ablations
