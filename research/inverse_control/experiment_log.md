
## PoC-P1 (2026-11-13 19:01)
- Hypothesis: discrete P-controller inverse matches lattice
- Val oracle: cov 83.7%, MAE 3.4248, hybrid 4.4341
- Runtime: 9.5s

## PoC-P2-tight + PI (2026-11-13 21:06)
- Unique-P tol=0.001 (fixed arithmetic-lattice false positives)
- P val oracle cov=12.5% MAE=7.1828948062565985
- PI val oracle cov=15.9% MAE=0.10874863128125001
- PI val u_in-only cov=8.5% MAE=0.0013717491804878074
- Runtime 792.6s

## PF Batch D started (2026-11-14)
- Holdout screens only, e1 protocol (75ep AMP, seed 42). Not a 7-model clone.
- Queue: pf_lagback (u_in future lags 1–2) then pf_rc_onehot.
- Control: e1_baseline holdout 0.2808.

## PF ablations (2026-11-14 13:04)
- pf_lagback: val=0.2881 hard=0.3165 Δe1=-0.0074 ep=69 sec=4814
- pf_rc_onehot: val=0.2605 hard=0.2847 Δe1=+0.0202 ep=66 sec=4839

## 3-stage pipeline (2026-11-15)
- PF tiny net holdout 0.4772 (underfit; pruned vs e3 0.2519 OOF)
- Ensemble e1+e3 OOF inv-MAE **0.2393** (btm/mean 0.2399); snap **0.2384**
- Tight PI on e3: cov 3.6%, trusted MAE 0.0093, hybrid 0.2452
- Full hybrid ensemble+snap+PI OOF **0.2322**
- Submit: fold-mean e1+e3 weighted + snap + PI on test (282/7545 breaths)
