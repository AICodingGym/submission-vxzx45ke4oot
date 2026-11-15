# Batch 2 — hypotheses

Best-known at start: BiLSTM full-data 60ep, val insp MAE **0.3747** (held fixed until beaten under comparable protocol).

| ID | Hypothesis | Test |
|----|------------|------|
| **E1** | Memorization fails because **targets are unscaled** and/or LR is poorly matched to raw cmH2O MAE/MSE — not because of a data-pipeline bug. Fixing target scaling + LR should drive 500-breath train insp MAE near 0, and the same fix should improve full-data training. | 500-breath overfit matrix: ±scale × {MAE,MSE,Huber} × LR; then A/B full-data with/without winning recipe |
| **E2** | The full model is **underfit**; longer training + patient ReduceLROnPlateau + loss choice + larger width/depth will beat 0.375. | Screen on fixed 80/20 breath split; track train vs val |
| **E3** | **Local conv / attention** and richer cumulative features help especially on high-R low-C trajectories where error grows over the breath. | Conv-BiLSTM, Attn-BiLSTM, richer hist feats; report overall + R=50,C=10 MAE |
| **E4** | R=50,C=10 is physically distinct (stiff, resistive): sharper ΔP for given volume proxy, more nonlinear u_in→P maps, late-breath accumulation — explaining growing late errors. | Comparative EDA of controls, pressure dynamics, repeats, late-step residuals |
| **E5** | Tree and strong LSTM errors are only partly correlated; **mean/median/weighted** blends can beat either alone. | Same-split preds; corr; blend grid; grouped K-fold for the chosen finalist |

**Final protocol:** anything labeled final uses **GroupKFold on `breath_id` (K=5)**, metric = inspiratory MAE.
