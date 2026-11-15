# Three-stage 2nd-place pipeline (own gym data)

Published 2nd-place MAE was ~0.07. We reimplemented the **method** on gym `train.csv` / `test.csv` only (no third-party CSVs or weights).

## Validation tracks (inspiratory MAE)

| Track | Protocol | MAE | Notes |
|-------|----------|-----|-------|
| Best original BiLSTM | GroupKFold K=5 seed 42 | **0.2519** | `e3_aug_uin` (Batch 2 fallback 0.2506) |
| PF-inspired neural | holdout 80/20 seed 42 | **0.4772** | Tiny-width 5×BiLSTM+GRU, 40 ep. Still falling at ep 40; 4060 cannot host 768-wide × 300 ep × 7 folds. **Pruned.** |
| Neural ensemble | GroupKFold OOF, e1+e3 | **0.2393** | inv-MAE weights beat mean/median/`better_than_median` slightly (0.2399) |
| Inverse PI + neural fallback | GroupKFold OOF, tight ID | **0.2452** | vs e3 0.2519; 3.6% insp coverage; trusted MAE 0.0093 |
| **Full hybrid** | ensemble + lattice snap + tight PI | **0.2322** | **Best OOF.** Coverage 3.6%; hard 50/10 0.257 |

## What was pruned

- Full Pulp Fiction (7 × TPU 768-wide, 300 ep): infeasible. Tiny clone underfits (0.47 vs 0.25).
- Loose PI (p1∧p2 + 8 lattice hits): trusted MAE 1.59, **hurts** NN. Tightened to 8/12 early hits and ≥75% lattice fraction.
- Extra PF seeds / GRU-only / 9h 950-p0 search.

## Submission

`predictions.csv`: e1+e3 5-fold means, inv-MAE weights (0.477 / 0.523), unique-pressure snap, tight PI overlay on 282/7545 test breaths.
