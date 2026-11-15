"""Shared constants and helpers for inverse ventilator controller ID."""
from __future__ import annotations

import numpy as np

PRESSURE_STEP = 0.07030214544382634

# Hypothesis grids from AmbrosM writeup (verify on data; not hard-coded truths)
P_COEF_GRID = np.array([
    0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
], dtype=np.float64)
P_STAR_GRID = np.array([10, 15, 20, 25, 30, 35], dtype=np.float64)
I_COEF_GRID = P_COEF_GRID.copy()  # 1st place notes same set as p_coef per comments

U_CLIP_LO = 0.0
U_CLIP_HI = 100.0
CLIP_EPS = 0.05


# AmbrosM: match one of 950 discrete sensor values, "very small tolerance".
# HALF_STEP ≈ 0.035; a looser threshold falsely accepts almost every float.
UNIQUE_P_TOL = 1e-3
T_PI_HYPOTHESIS = 0.5  # AmbrosM writeup; treat as hypothesis to verify


def snap_lattice(p: np.ndarray, step: float = PRESSURE_STEP) -> np.ndarray:
    """Snap to nearest arithmetic lattice point (legacy; prefer unique-set snap)."""
    return np.round(p / step) * step


def lattice_residual(p: np.ndarray, step: float = PRESSURE_STEP) -> np.ndarray:
    """Distance to nearest arithmetic lattice point."""
    s = snap_lattice(p, step)
    return np.abs(p - s)


def unique_pressure_residual(p: np.ndarray, uniq: np.ndarray) -> np.ndarray:
    """Distance to nearest observed unique pressure (950-value sensor set)."""
    p = np.asarray(p, dtype=np.float64)
    idx = np.searchsorted(uniq, p)
    idx = np.clip(idx, 1, len(uniq) - 1)
    left, right = uniq[idx - 1], uniq[idx]
    return np.minimum(np.abs(p - left), np.abs(p - right))


def on_unique_pressure(p: np.ndarray, uniq: np.ndarray, tol: float = UNIQUE_P_TOL) -> np.ndarray:
    return unique_pressure_residual(p, uniq) <= tol


def snap_unique(p: np.ndarray, uniq: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    idx = np.searchsorted(uniq, p)
    idx = np.clip(idx, 1, len(uniq) - 1)
    left, right = uniq[idx - 1], uniq[idx]
    return np.where(np.abs(p - left) <= np.abs(p - right), left, right)


def clip_mask(u_in: np.ndarray) -> np.ndarray:
    return (u_in <= U_CLIP_LO + CLIP_EPS) | (u_in >= U_CLIP_HI - CLIP_EPS)


def linear_p_fit(u_in: np.ndarray, pressure: np.ndarray) -> tuple[float, float, float]:
    """Fit u_in ≈ a + b*p on 1D arrays; return a,b,r2."""
    if u_in.size < 3:
        return 0.0, 0.0, 0.0
    p = pressure.astype(np.float64)
    u = u_in.astype(np.float64)
    var_p = np.var(p)
    if var_p < 1e-12:
        return float(u.mean()), 0.0, 0.0
    b, a = np.polyfit(p, u, 1)
    pred = a + b * p
    ss_res = np.sum((u - pred) ** 2)
    ss_tot = np.sum((u - u.mean()) ** 2) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    return float(a), float(b), float(r2)


def p_params_from_linear(a: float, b: float) -> tuple[float | None, float | None]:
    """If u = p_coef*(p_star - p) = -p_coef*p + p_coef*p_star, then b=-p_coef, a=p_coef*p_star."""
    if abs(b) < 1e-9:
        return None, None
    p_coef = -b
    p_star = a / p_coef
    return float(p_coef), float(p_star)