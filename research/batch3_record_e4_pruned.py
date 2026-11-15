"""Record e4_residual screen from interrupted run (best checkpoint ep 70 from logs)."""
from __future__ import annotations

import json
from pathlib import Path

from batch2_common import OUT
from batch3_state import load_state, record_screening

CURVE = [
    {"epoch": 1, "val": 1.2129, "hard": 1.6138},
    {"epoch": 10, "val": 0.5679, "hard": 0.7034},
    {"epoch": 20, "val": 0.5017, "hard": 0.5536},
    {"epoch": 30, "val": 0.4275, "hard": 0.5261},
    {"epoch": 40, "val": 0.5215, "hard": 0.6235},
    {"epoch": 50, "val": 0.3869, "hard": 0.4428},
    {"epoch": 60, "val": 0.3709, "hard": 0.4498},
    {"epoch": 70, "val": 0.3602, "hard": 0.4187},
    {"epoch": 80, "val": 0.4137, "hard": 0.4677},
]

def main():
    state = load_state()
    if state.get("screening", {}).get("e4_residual", {}).get("done"):
        print("e4_residual already recorded")
        return
    curve_path = OUT / "batch3_curve_e4_residual.json"
    curve_path.write_text(
        json.dumps({"history": CURVE, "note": "from terminal log; best ckpt ep 70"}, indent=2),
        encoding="utf-8",
    )
    record_screening(
        state,
        "e4_residual",
        val_mae=0.3602,
        hard_mae=0.4187,
        best_epoch=70,
        recipe={
            "name": "residual", "residual": True, "epochs": 85, "lr": 0.003, "loss": "mse",
        },
        extra={
            "stopping_reason": "manual_pruned_far_from_leader_after_ep70",
            "epochs_ran": 80,
            "epochs_budget": 85,
            "less_than_full_budget": True,
            "borderline_rerun_needed": False,
            "learning_curve_file": curve_path.name,
            "screen_stop_enabled": False,
        },
    )
    print("Recorded e4_residual pruned at best val 0.3602")


if __name__ == "__main__":
    main()