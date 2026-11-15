# Batch 3 — hypotheses (target: insp MAE ≤ 0.20; stretch 0.1364)

**Fallback best:** Batch 2 rich BiLSTM — GroupKFold OOF **0.2506 ± 0.0194**

| ID | Hypothesis | Test |
|----|------------|------|
| **E1** | Batch 2 peak is reproducible; seed noise is smaller than fold noise (~±0.019), so claims need multi-seed GroupKFold. | Same 5-fold splits; seeds {42,0,1}; report per-seed OOF and pooled stats |
| **E2** | Auxiliary **Δp** and **intra-breath pressure cumsum** sharpen representation and lower insp MAE. | Multi-task BiLSTM; ablate aux weights |
| **E3** | Light **u_in jitter** or **feature masking** helps generalization; mixup is risky. | Each aug vs baseline on **identical folds** (one seed) |
| **E4** | **Residual BiLSTM** + longer plateau training closes train–val gap without overfitting. | vs baseline; 500-breath memorization spot-check |
| **E5** | **Physics linear map on V/C, R·V/C** explains weak u→P at R50C10; hybrid residual NN beats pure LSTM. | Per-fold physics fit + NN on residual or stacked hybrid |
| **E6** | Multi-seed OOF **mean/median** and **lattice snap** help as errors approach 0.07 step. | Ensemble E1 checkpoints; snap harm/benefit breakdown |

Confirm gains with **same GroupKFold protocol** as Batch 2. Flag holdout-only vs OOF.

## Protocol (Batch 3 run)

1. **SCREENING** — single holdout (20%, seed 42), train seed 42; all configs.
2. **Baseline audit** — `batch3_baseline_audit.json` vs Batch 2 holdout 0.239 / OOF 0.2506; optional `--baseline-replay` (90 ep, no AMP).
3. **Promotion** — Δ≥0.005 vs e1, hard50_10 exception, top-3 exploratory, controls **e1** + **e2** always CV.
4. **STOP** — `batch3_screening_report.txt`; no CV until `--run-cv-approved`.
5. **Phase 2 CV** — GroupKFold × **seed 42 only**; multi-seed via `--multi-seed-final <key>` on one finalist.
6. **Submit** — pick winner by **OOF only**; one gym submit; no leaderboard tuning.
