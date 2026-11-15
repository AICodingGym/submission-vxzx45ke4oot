# Inverse-controller first investigation (stop for review)

**Track:** `research/inverse_control/` only. Batch 3 artifacts untouched. No gym submit. No third-party CSVs.

## Project state at this writeup

- **Validated neural best:** Batch 2 rich BiLSTM **GroupKFold OOF 0.2506**. Batch 3 has **no CV yet** (screening only). Screen leader `e3_aug_uin` holdout **0.2616** is not OOF and is worse than Batch 2 holdout **~0.239**.
- **Batch 3 CV:** not started (awaits approval).
- **Sources:** AmbrosM writeup (full), arXiv:2102.06779 (abstract), Pulp Fiction from **local** `C:\Users\argsw\Downloads\gb-vpp-pulp-fiction.ipynb`.

## What we ran (PoC only)

| Exp | Data | Runtime | Result |
|-----|------|---------|--------|
| PoC-P1 (arithmetic lattice, tol 0.51×step) | 4k breaths | ~10s | **Invalid** — tol > half-step ⇒ almost every float “matches” |
| PoC-P2 (950 unique pressures, tol 1e-3) | 1.2k breaths | part of 13 min | P rarely identifies; oracle trusted MAE **7.18** |
| PoC-PI (T=0.5 hyp., 53 frequent p0) | 400 breaths | ~13 min total | Oracle 26/320 train MAE<0.15; val **u_in-only 8.5% coverage, MAE 0.0014** on accepted steps |

## Seven-point report

1. **Controller structure in our data**  
   950 unique pressures (matches writeup). Linear u↔p R² ~0.43 (not a clean P law). A **minority** of breaths fit AmbrosM **PI with T=0.5** well (train: 26/320 with recon MAE 0.037 among those). Most breaths do **not** invert as P.

2. **Inverse P vs PI**  
   **P is not viable** once unique-sensor matching is used. **PI is viable on a subset**: when test-time ID succeeds, reconstruction MAE is ~**0.001** (sensor lattice). Coverage is the bottleneck, not precision on accepted steps.

3. **Held-out inspiratory reconstruction** (PI, 80-breath val, inference rules only)  
   **~8.5%** of insp steps trusted (7/80 breaths). Oracle (labels for ID only) ~16% coverage, MAE 0.11. Not yet AmbrosM-scale coverage (they searched all 950 p[0]; we used 53 frequent p0).

4. **Hybrid vs neural**  
   **Not measured with NN fallback.** Mean-pressure fallback MAE ~6 (useless). Do **not** replace the Batch 2 model. A hybrid is only worth a holdout screen **after** we attach NN preds to the same breaths. Naive arithmetic: replacing 8.5% of 0.25-error steps with 0.001 could shave ~0.02 if ID has few false matches — unproven.

5. **Pulp Fiction worth testing (from notebook, not writeup inference)**  
   Highest-leverage vs our BiLSTM: **within-breath `u_in` lag_back**, **R/C one-hot**, not the 5-stack TPU LSTM. One ablation at a time. See `pulp_fiction_notes.md`.

6. **Bottlenecks**  
   Naive PI ID ~21k loops/breath (our reduced p0 set) ⇒ ~6s/val breath in Python. Full 950×20×20×6 as in the writeup is ~9h locally — **not started**. Vectorize p_coef/i_coef; CPU parallelize breaths before scaling.

7. **Next three experiments (need your OK before large runs)**  
   1. Holdout hybrid: Batch 2-style NN preds + replace only PI-trusted steps; report **total insp MAE** (screen, not OOF).  
   2. Widen p0 search on a **200-breath** subset; log coverage vs time.  
   3. Single neural ablation: `u_in` lag_back 1–2 on our rich BiLSTM, same Batch 3 holdout.

**Do not** start full-data inversion, 7-fold Pulp Fiction clones, or Batch 3 multi-seed CV from this track.
