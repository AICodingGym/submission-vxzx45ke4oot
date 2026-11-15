# Research backlog — ventilator pressure prediction

Primary metric: **inspiratory MAE** (`u_out == 0`).  
Final protocol: **GroupKFold on `breath_id` (K=5)** when claiming a best-known model.

## Best-known validated model (independent of experiments)

| Model | Metric | Value | Protocol |
|-------|--------|-------|----------|
| **BiLSTM + rich feats, MSE + y-scale, LR=3e-3, ReduceLROnPlateau** | **GroupKFold OOF insp MAE** | **0.2506 ± 0.0194** | Batch 2 E5 finalist |
| Same (single 80/20 screen) | holdout insp MAE | 0.2389 | Batch 2 E3 |
| Prior sanity BiLSTM 60ep | single 80/20 | 0.3747 | superseded as best-known |

Hard-regime (R=50,C=10) OOF MAE under finalist: **0.2706**.

## Batch 2 evidence (short)

- **E1:** Cosine LR decay was blocking memorization. Fixed high LR + target scaling → train MAE **0.155** on 500 breaths (still not ~0). Scale/LR matter; residual gap remains.
- **E2:** Underfit hypothesis confirmed. Best screen **0.2489** (scaled MSE, plateau, lr=3e-3) vs prior 0.375.
- **E3:** Richer history features help (**0.2389**). Conv/Attn did **not** beat plain BiLSTM+rich on this budget. Hard-weighting mixed.
- **E4:** R=50,C=10 is physically stiff/resistive: low `u_in`, high `|ΔP|`, near-zero corr(u,P), strong corr(V/C,P), large linear residual; late-step errors dominate.
- **E5:** Tree≪NN (OOF 0.74 vs 0.25). Error corr only 0.32 but blends **hurt** because tree is too weak. Best blend = pure NN.

## Ranked open questions

| Rank | Question | Status |
|------|----------|--------|
| 1 | Can we close the R=50,C=10 gap further (specialists, phase-aware loss, better volume/flow state)? | **Next** |
| 2 | Longer/fuller training of rich BiLSTM (folds still improving near ep80)? | Next |
| 3 | Why can’t LSTM memorize 500 breaths to ~0 — need breath embeddings / deeper head / different inductive bias? | Diagnostic |
| 4 | Stronger second model for ensembling (deep tree with more rounds, or second NN seed)? | Later |
| 5 | Submit-ready full-train inference + postprocess | Later |

## Batch log

- Batch 0–1 + sanity: see prior reports
- **Batch 2 complete** (`batch2_report.txt`, `batch2_results.json`, `batch2_hypotheses.md`) — awaiting review before Batch 3
