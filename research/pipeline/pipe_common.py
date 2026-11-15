"""Shared paths, metrics, keep-awake, scoreboard for the 3-stage PF pipeline."""
from __future__ import annotations

import ctypes
import json
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
RESEARCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESEARCH))
sys.path.insert(0, str(RESEARCH / "inverse_control"))

DATA = RESEARCH.parent / "data" / "ventilator-pressure-prediction"
PIPE = Path(__file__).resolve().parent
INV = RESEARCH / "inverse_control"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
VAL_FRAC = 0.2

# Locked references (do not overwrite prior artifacts)
BILSTM_OOF = 0.2519  # batch3 e3_aug_uin GroupKFold
BILSTM_OOF_B2 = 0.2506  # batch2 rich BiLSTM
E1_HOLD = 0.2808
E3_HOLD = 0.2616

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002


def setup_cuda() -> str:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for PF neural stage")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.cuda.empty_cache()
    return torch.cuda.get_device_name(0)


@contextmanager
def keep_awake():
    k32 = ctypes.windll.kernel32
    k32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED)
    try:
        yield
    finally:
        k32.SetThreadExecutionState(ES_CONTINUOUS)


def insp_mae(y, pred, u_out) -> float:
    m = np.asarray(u_out).reshape(-1) == 0
    return float(np.mean(np.abs(np.asarray(y).reshape(-1)[m] - np.asarray(pred).reshape(-1)[m])))


def insp_mae_hard(y, pred, u_out, R, C, r=50, c=10) -> float:
    m = (np.asarray(u_out).reshape(-1) == 0) & (np.asarray(R).reshape(-1) == r) & (np.asarray(C).reshape(-1) == c)
    if m.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(np.asarray(y).reshape(-1)[m] - np.asarray(pred).reshape(-1)[m])))


def scoreboard_path() -> Path:
    return PIPE / "scoreboard.json"


def load_scoreboard() -> dict:
    p = scoreboard_path()
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {
        "tracks": {
            "original_bilstm": {
                "protocol": "GroupKFold K=5 seed 42",
                "oof_insp_mae": BILSTM_OOF,
                "name": "e3_aug_uin",
                "batch2_fallback_oof": BILSTM_OOF_B2,
            },
            "pf_neural": {"status": "pending"},
            "neural_ensemble": {"status": "pending"},
            "inverse_pi_neural_fallback": {"status": "pending"},
            "full_hybrid": {"status": "pending"},
        }
    }


def save_scoreboard(sb: dict) -> None:
    scoreboard_path().write_text(json.dumps(sb, indent=2, default=float), encoding="utf-8")


def update_track(name: str, payload: dict) -> dict:
    sb = load_scoreboard()
    sb.setdefault("tracks", {})[name] = payload
    save_scoreboard(sb)
    return sb
