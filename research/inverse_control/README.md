# Inverse controller identification (DSC 102 / PA3)

**Isolated from Batch 1–3.** Does not modify `batch3_*` checkpoints or training.

## References (our reading status)

| Source | Access | Notes |
|--------|--------|--------|
| [AmbrosM writeup — inverse PID](https://www.kaggle.com/competitions/ventilator-pressure-prediction/writeups/ambrosm-2-solution-the-inverse-of-a-pid-controller) | OK | P/PI inverse, discrete grids, lattice checks, clip handling |
| [arXiv:2102.06779](https://arxiv.org/abs/2102.06779) | Abstract only via fetch | Google/princeton ventilator ML control; PID baseline; **not** the Kaggle discrete inverse recipe |
| [Pulp Fiction notebook](https://www.kaggle.com/code/dlaststark/gb-vpp-pulp-fiction) | Metadata only | Runtime ~7h TPU; Apache 2.0; **architecture not in fetch** — defer to manual notebook review |

## Best neural reference (unchanged)

- **Batch 2** rich BiLSTM GroupKFold OOF **0.2506** (authoritative until Batch 3 completes CV).
- **Batch 3** Phase 1 screening only (in progress): e1 holdout **0.2808** (not comparable to OOF).

## Run PoC

```bash
python research/inverse_control/inv_p_poc.py
```

Outputs: `research/inverse_control/poc_results.json`, `poc_report.txt`, `experiment_log.md`