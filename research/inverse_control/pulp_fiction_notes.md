# GB VPP Pulp Fiction — notes from local notebook

Source: `C:\Users\argsw\Downloads\gb-vpp-pulp-fiction.ipynb` (our copy; not executed in repo).

## Architecture (documented in notebook)

- **Input:** `(80, n_features)` after `RobustScaler` + reshape.
- **Stack:** 5× `Bidirectional(LSTM)` — 768 → 512 → 384 → 256 → 128 (all `return_sequences=True`).
- **Parallel GRU branch:** from LSTM layer 2 onward, **Multiply** LSTM×GRU intermediates + `BatchNormalization`, BiGRU 384→256→128→64.
- **Head:** `Concatenate` [x5, z2, z3, z4, z5] → `Dense(128, selu)` → `Dense(1)` per timestep.
- **“7 models”:** `KFold(n_splits=7, shuffle=True, seed=2021)` — one checkpoint per fold; test prediction = list of 7 fold models (AmbrosM ensembled these).

## Features (documented)

- Lags **forward and backward** on `u_in` / `u_out` (lags 1–4 + `lag_back` diffs) — **future** valve settings within breath.
- Breath-level u_in max/mean, diffs to max/mean, cumsums, `ewm(halflife=9)`, rolling u_in window 15.
- `area` = cumsum(`time_step * u_in`), crosses with `u_out`.
- **R/C:** string `R__C` then **`pd.get_dummies` on entire dataframe** → one-hot for R, C, and RC pairs (high dim).
- **Not in notebook:** explicit `vol/C`, `u_r/C`, or physics-style terms (we have these in `SEQ_COLS_RICH`).

## Training (documented)

- `loss="mae"`, `Adam`, **all 80 timesteps** in loss (no `u_out==0` mask in compile).
- Up to **300 epochs**, `EarlyStopping(patience=30)`, `ReduceLROnPlateau(factor=0.85, patience=7)`.
- **TPU** `BATCH_SIZE = replicas * 64` (metadata said ~7h on TPU v3-8).

## vs our Batch 2 rich BiLSTM

| Aspect | Pulp Fiction | Ours (Batch 2/3) |
|--------|--------------|-------------------|
| Depth/width | 5 BiLSTM + 4 BiGRU + fusion | 3-layer BiLSTM h=256 |
| Fusion | LSTM×GRU Multiply + concat | — |
| Future inputs | u_in/u_out lag_back 1–4 | No future leaks (causal) |
| R/C | one-hot dummies | numeric R, C, products |
| Scaling | RobustScaler | z-score per fold |
| Loss | full-sequence MAE | **insp-only** masked MAE |
| CV | 7-fold shuffle KFold | **GroupKFold** on breath_id |
| Hardware | TPU large batch | GPU RTX 4060 |

## Highest-value ablations for us (1–2 at a time)

1. **Backward lags within breath** on holdout (screen only) — tests non-causal hint vs our EDA causality story.
2. **Insp-only vs full-sequence MAE** on same arch — aligns metric with competition.
3. **R/C one-hot** vs numeric (lightweight, no 5-stack LSTM).
4. **GRU multiply branch** on top of our BiLSTM — one fusion block, not full PF clone.

Do **not** replicate 7-fold × 300ep × huge LSTM on 4060 without screening.

## Batch D screen results (holdout 80/20 seed 42, e1 recipe, 2026-11-14)

Same protocol as Batch 3 e1: 75 ep, AMP, MSE + y-scale, BiLSTM 256×3. Control **e1 = 0.2808** / hard **0.3119**. e3_aug_uin screen **0.2616** is a different recipe (u_in noise).

| Config | Holdout MAE | Hard 50/10 | Δ vs e1 | Best ep |
|--------|-------------|------------|---------|---------|
| e1_baseline (control) | 0.2808 | 0.3119 | — | 72 |
| **pf_lagback** (u_in future lags 1–2 + diffs) | **0.2881** | 0.3165 | **−0.0074** | 69 |
| **pf_rc_onehot** (R/C/RC dummies; drop numeric R,C) | **0.2605** | 0.2847 | **+0.0202** | 66 |
| e3_aug_uin (reference, not this run) | 0.2616 | 0.2837 | +0.0192 | 64 |

- **lag_back did not help.** We already have `u_in_backcum`; local future lags added noise, not signal. Drop it.
- **R/C one-hot tied e3_aug_uin on holdout** (~0.26) and beat e1 by 0.020. **Holdout only — not OOF.** No CV until approved.
- Not started: GRU×LSTM multiply, full-sequence MAE, stacking one-hot + u_in aug, 7-fold PF clone.