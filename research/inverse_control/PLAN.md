# Experimental plan — inverse controller + hybrid (90 min budget)

## Project state

- **Batch 3** Phase 1 **complete** (screen leader `e3_aug_uin` 0.2616). **No GroupKFold / no new OOF.** CV awaits approval.
- **Neural reference:** Batch 2 BiLSTM rich OOF **0.2506**, holdout **~0.239**.
- **This track:** `research/inverse_control/` only. First investigation: `FIRST_INVESTIGATION.md`.

## Literature → our hypotheses

| Idea | Source | Our test |
|------|--------|----------|
| `u_in` from feedback on pressure | AmbrosM writeup | Linear u vs p per breath; grid (p_coef, p_star) |
| Inverse P: `p = p_star - u/p_coef` | AmbrosM | Lattice snap @ 0.0703 step |
| Clip breaks invertibility | AmbrosM | Mask u≈0 or u≈100 |
| PI + state q, T≈0.5 | AmbrosM (hypothesis) | PoC-PI after P viability |
| 950 discrete pressures | EDA batch0 | Same tolerance as writeup |
| NN ensemble + PID overlay | AmbrosM final pipeline | Hybrid after coverage analysis |
| Pulp Fiction 7 models | Notebook (metadata only) | Manual diff vs our BiLSTM features/arch |

Pulp Fiction architecture: read from local notebook (`pulp_fiction_notes.md`).

## Batch A — Inverse P (now)

1. `inv_p_poc.py` — 4k breaths, breath holdout 20%/seed 42.
2. Metrics: lattice hit rate, insp step **coverage**, MAE on trusted steps.
3. **Inference rule (PoC):** trust iff non-clipped + lattice (no label at test).

## Batch B — PI (if P coverage > 0)

1. Small subset, T=0.5 grid + i_coef from writeup.
2. Fit q0 from first step; verify p[1],p[2] on lattice (train labels for eval only).

## Batch C — Hybrid screen (holdout)

1. Replace trusted steps with inverse; neural fallback elsewhere (Batch 2 weights when available).
2. Tune **coverage threshold on train** only; report val insp MAE.

## Batch D — Pulp Fiction delta (1–2 ablations)

1. After notebook manual review: candidate changes (e.g. unidirectional LSTM, feature set).
2. Single holdout screen; no 7-model ensemble.

## Stop rules

- P coverage < 5% insp on val → document failure, keep NN.
- PoC runtime > 90 min → stop for review.
- No full-dataset 9h inversion without approval.

## Deliverables

- `experiment_log.md`, `poc_results.json`, `poc_report.txt`
- End of investigation: 7-point summary per assignment brief.