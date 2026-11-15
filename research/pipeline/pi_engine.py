"""Vectorized inverse PI identification (test-time: u_in / dt / lattice only).

Replaces neural preds only on steps that pass strict unique-pressure checks.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from numba import njit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inverse_control"))

from inv_common import (
    I_COEF_GRID,
    P_COEF_GRID,
    P_STAR_GRID,
    T_PI_HYPOTHESIS,
    UNIQUE_P_TOL,
    U_CLIP_HI,
    U_CLIP_LO,
    CLIP_EPS,
    on_unique_pressure,
    snap_unique,
)

MIN_HITS = 8


@njit(cache=True)
def _reconstruct(uin, dt, pc, ic, ps, q0, T):
    n = uin.shape[0]
    ph = np.empty(n, dtype=np.float64)
    q = q0
    ph[0] = ps - (uin[0] - ic * q) / pc
    for i in range(1, n):
        s = dt[i] / (dt[i] + T)
        k = pc + ic * s
        if abs(k) < 1e-12:
            for j in range(i, n):
                ph[j] = np.nan
            break
        ph[i] = ps - (uin[i] - ic * (1.0 - s) * q) / k
        e = ps - ph[i]
        q = (1.0 - s) * q + s * e
    return ph


@njit(cache=True)
def _searchsorted_near(uniq, val, tol):
    lo = 0
    hi = uniq.shape[0]
    while lo < hi:
        mid = (lo + hi) // 2
        if uniq[mid] < val:
            lo = mid + 1
        else:
            hi = mid
    best = 1e9
    for j in (lo - 1, lo, lo + 1):
        if 0 <= j < uniq.shape[0]:
            d = abs(uniq[j] - val)
            if d < best:
                best = d
    return best <= tol


@njit(cache=True)
def identify_pi_uin(
    uin, uout, dt, uniq,
    p0_cands, pc_grid, ic_grid, ps_grid,
    T, tol, clip_lo, clip_hi, clip_eps, min_hits, min_frac,
):
    """Return best (pc, ic, ps, q0, hits) or hits=-1 if none."""
    u0, u1, u2 = uin[0], uin[1], uin[2]
    s1 = dt[1] / (dt[1] + T)
    s2 = dt[2] / (dt[2] + T)
    best_hits = -1
    best_pc = 0.0
    best_ic = 0.0
    best_ps = 0.0
    best_q0 = 0.0
    for ip in range(p0_cands.shape[0]):
        p0 = p0_cands[ip]
        for ipc in range(pc_grid.shape[0]):
            pc = pc_grid[ipc]
            if pc < 1e-9:
                continue
            for iic in range(ic_grid.shape[0]):
                ic = ic_grid[iic]
                if abs(ic) < 1e-9:
                    continue
                for ips in range(ps_grid.shape[0]):
                    ps = ps_grid[ips]
                    q0 = (u0 + pc * (p0 - ps)) / ic
                    k1 = pc + ic * s1
                    if abs(k1) < 1e-12:
                        continue
                    p1 = ps - (u1 - ic * (1.0 - s1) * q0) / k1
                    if not _searchsorted_near(uniq, p1, tol):
                        continue
                    e1 = ps - p1
                    q1 = (1.0 - s1) * q0 + s1 * e1
                    k2 = pc + ic * s2
                    if abs(k2) < 1e-12:
                        continue
                    p2 = ps - (u2 - ic * (1.0 - s2) * q1) / k2
                    if not _searchsorted_near(uniq, p2, tol):
                        continue
                    ph = _reconstruct(uin, dt, pc, ic, ps, q0, T)
                    hits = 0
                    n_use = 0
                    hits12 = 0
                    n12 = 0
                    nlim = 12 if uin.shape[0] >= 12 else uin.shape[0]
                    for t in range(uin.shape[0]):
                        if uout[t] != 0.0:
                            continue
                        if uin[t] <= clip_lo + clip_eps or uin[t] >= clip_hi - clip_eps:
                            continue
                        if not np.isfinite(ph[t]):
                            continue
                        n_use += 1
                        on = _searchsorted_near(uniq, ph[t], tol)
                        if on:
                            hits += 1
                        if t < nlim:
                            n12 += 1
                            if on:
                                hits12 += 1
                    if n12 < 6 or hits12 < min_hits:
                        continue
                    if n_use < min_hits:
                        continue
                    frac = hits / n_use
                    if frac < min_frac:
                        continue
                    if hits > best_hits:
                        best_hits = hits
                        best_pc = pc
                        best_ic = ic
                        best_ps = ps
                        best_q0 = q0
    return best_pc, best_ic, best_ps, best_q0, best_hits


def dt_from_t(t: np.ndarray) -> np.ndarray:
    dt = np.diff(t, prepend=t[0])
    dt[0] = 0.0
    return dt.astype(np.float64)


def trust_mask(uin, uout, ph, uniq) -> np.ndarray:
    insp = uout == 0
    clipped = (uin <= U_CLIP_LO + CLIP_EPS) | (uin >= U_CLIP_HI - CLIP_EPS)
    ok = insp & ~clipped & np.isfinite(ph) & on_unique_pressure(ph, uniq, UNIQUE_P_TOL)
    return ok


def identify_breath(uin, uout, t, uniq, p0_cands) -> dict | None:
    dt = dt_from_t(t)
    pc, ic, ps, q0, hits = identify_pi_uin(
        uin.astype(np.float64), uout.astype(np.float64), dt, uniq.astype(np.float64),
        p0_cands.astype(np.float64),
        P_COEF_GRID, I_COEF_GRID, P_STAR_GRID,
        T_PI_HYPOTHESIS, UNIQUE_P_TOL, U_CLIP_LO, U_CLIP_HI, CLIP_EPS, MIN_HITS, 0.75,
    )
    if hits < MIN_HITS:
        return None
    ph = _reconstruct(uin.astype(np.float64), dt, pc, ic, ps, q0, T_PI_HYPOTHESIS)
    m = trust_mask(uin, uout, ph, uniq)
    if int(m.sum()) < MIN_HITS:
        return None
    pred = np.full_like(ph, np.nan)
    pred[m] = snap_unique(ph[m], uniq)
    return {
        "p_coef": float(pc), "i_coef": float(ic), "p_star": float(ps),
        "q0": float(q0), "hits": int(hits), "pred": pred, "trust": m,
    }


def overlay_nn(nn_pred: np.ndarray, pi_pred: np.ndarray, trust: np.ndarray) -> np.ndarray:
    out = nn_pred.copy()
    out[trust] = pi_pred[trust]
    return out


def frequent_p0(y0: np.ndarray, uniq: np.ndarray, k: int = 80) -> np.ndarray:
    """Train-only: most common snapped p[0] values. Safe at test (no test labels)."""
    snapped = snap_unique(y0.astype(np.float64), uniq)
    vals, cnt = np.unique(snapped, return_counts=True)
    order = np.argsort(-cnt)
    return vals[order[:k]]
