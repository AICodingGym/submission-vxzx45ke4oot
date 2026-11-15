"""Batch 0 / EDA: structure of ventilator pressure data."""
from __future__ import annotations

from math import gcd
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(
    r"C:\Users\argsw\Downloads\dsc102\ventilator-pressure-prediction"
    r"\data\ventilator-pressure-prediction"
)
OUT = Path(
    r"C:\Users\argsw\Downloads\dsc102\ventilator-pressure-prediction\research"
)
OUT.mkdir(parents=True, exist_ok=True)

print("Loading...")
train = pd.read_csv(DATA / "train.csv")
test = pd.read_csv(DATA / "test.csv")

lines: list[str] = []


def log(s: str = "") -> None:
    print(s)
    lines.append(s)


# ---------------------------------------------------------------------------
# 1. Natural unit of prediction / temporal structure
# ---------------------------------------------------------------------------
log("=" * 72)
log("1. NATURAL UNIT & TEMPORAL STRUCTURE")
log("=" * 72)
log(f"train rows={len(train):,} breaths={train['breath_id'].nunique():,}")
log(f"test  rows={len(test):,} breaths={test['breath_id'].nunique():,}")
log(f"steps/breath train unique sizes: {train.groupby('breath_id').size().unique()}")
log(f"steps/breath test unique sizes: {test.groupby('breath_id').size().unique()}")

dt = train.groupby("breath_id")["time_step"].diff().dropna()
log(
    f"dt median={dt.median():.6f} mean={dt.mean():.6f} std={dt.std():.6f} "
    f"q01={dt.quantile(0.01):.6f} q99={dt.quantile(0.99):.6f}"
)

# Inspiration / expiration lengths
insp_len = train.groupby("breath_id")["u_out"].apply(lambda s: int((s == 0).sum()))
exp_len = 80 - insp_len
log(f"inspiratory steps: min={insp_len.min()} max={insp_len.max()} mean={insp_len.mean():.2f} "
    f"median={insp_len.median()} unique={sorted(insp_len.unique())[:20]}... nunique={insp_len.nunique()}")
log(f"expiratory steps: min={exp_len.min()} max={exp_len.max()} mean={exp_len.mean():.2f}")
log(f"scored fraction (u_out==0): {(train['u_out']==0).mean():.4f}")

# Is u_out always a single switch 0->1?
def n_switches(s: pd.Series) -> int:
    return int((s.diff().fillna(0) != 0).sum())


sw = train.groupby("breath_id")["u_out"].apply(n_switches)
log(f"u_out switches/breath: value_counts={sw.value_counts().to_dict()}")

# R,C constant within breath?
rc_var = train.groupby("breath_id")[["R", "C"]].nunique().max()
log(f"max unique R/C within breath: {rc_var.to_dict()}")

# ---------------------------------------------------------------------------
# 2. Pressure discretization
# ---------------------------------------------------------------------------
log()
log("=" * 72)
log("2. PRESSURE DISCRETIZATION")
log("=" * 72)
p = train["pressure"].to_numpy()
uniq = np.unique(p)
log(f"n unique pressure values: {len(uniq)}")
diffs = np.diff(np.sort(uniq))
pos = diffs[diffs > 0]
log(f"min positive unique-diff: {pos.min():.10f}")
log(f"median/mean unique-diff: {np.median(pos):.10f} / {np.mean(pos):.10f}")

# Look for lattice step
scale = 1_000_000
ints = np.round(p * scale).astype(np.int64)
u_ints = np.unique(ints)
d = np.diff(u_ints)
d = d[d > 0]
# Use many diffs for gcd
sample = d[:5000].tolist()
g = reduce(gcd, sample)
log(f"approx lattice step via gcd: {g}/{scale} = {g/scale:.10f}")
# Count how many unique values fall on that lattice from min
step = g / scale
idx = np.round((uniq - uniq.min()) / step)
on_lattice = np.max(np.abs(uniq - (uniq.min() + idx * step))) < 1e-8
log(f"all unique on estimated lattice? {bool(np.all(np.abs(uniq - (uniq.min() + np.round((uniq-uniq.min())/step)*step)) < 1e-6))}")
log(f"pressure min/max: {p.min():.6f} / {p.max():.6f}")
log(f"top 10 most common pressures:\n{train['pressure'].value_counts().head(10).to_string()}")

# ---------------------------------------------------------------------------
# 3. Predictability: current inputs vs history
# ---------------------------------------------------------------------------
log()
log("=" * 72)
log("3. PREDICTABILITY: CURRENT vs HISTORY (inspiratory MAE)")
log("=" * 72)

# Breath-level CV split (no leakage)
breath_ids = train["breath_id"].unique()
rng = np.random.default_rng(42)
rng.shuffle(breath_ids)
n_val = int(0.2 * len(breath_ids))
val_ids = set(breath_ids[:n_val])
tr_ids = set(breath_ids[n_val:])

tr = train[train["breath_id"].isin(tr_ids)].copy()
va = train[train["breath_id"].isin(val_ids)].copy()
mask_va = va["u_out"] == 0


def insp_mae(y_true, y_pred, mask) -> float:
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


# Baseline: global inspiratory mean
mu = tr.loc[tr["u_out"] == 0, "pressure"].mean()
log(f"B0 global insp mean MAE: {insp_mae(va['pressure'].to_numpy(), np.full(len(va), mu), mask_va.to_numpy()):.4f}")

# Baseline: mean by (R,C)
rc_mean = tr.loc[tr["u_out"] == 0].groupby(["R", "C"])["pressure"].mean()
pred = va.set_index(["R", "C"]).index.map(rc_mean)
pred = pred.to_numpy(dtype=float)
# fill missing with global
pred = np.where(np.isnan(pred), mu, pred)
log(f"B1 mean by (R,C) MAE: {insp_mae(va['pressure'].to_numpy(), pred, mask_va.to_numpy()):.4f}")

# Current-step only: groupby (R,C,u_out,u_in_bin)
for bins in [10, 20, 50]:
    tr2 = tr.copy()
    va2 = va.copy()
    edges = np.linspace(0, 100, bins + 1)
    tr2["u_bin"] = pd.cut(tr2["u_in"], edges, include_lowest=True, labels=False)
    va2["u_bin"] = pd.cut(va2["u_in"], edges, include_lowest=True, labels=False)
    means = tr2.groupby(["R", "C", "u_out", "u_bin"], observed=True)["pressure"].mean()
    pred = va2.set_index(["R", "C", "u_out", "u_bin"]).index.map(means).to_numpy(dtype=float)
    pred = np.where(np.isnan(pred), mu, pred)
    log(f"B2 current (R,C,u_out,u_in/{bins}bins) MAE: {insp_mae(va2['pressure'].to_numpy(), pred, (va2['u_out']==0).to_numpy()):.4f}")

# Add time_step bin (still current, no history of u_in)
tr2 = tr.copy()
va2 = va.copy()
tr2["t_bin"] = (tr2["time_step"] / 0.1).astype(int)
va2["t_bin"] = (va2["time_step"] / 0.1).astype(int)
tr2["u_bin"] = (tr2["u_in"] / 5).astype(int)
va2["u_bin"] = (va2["u_in"] / 5).astype(int)
means = tr2.groupby(["R", "C", "u_out", "t_bin", "u_bin"])["pressure"].mean()
pred = va2.set_index(["R", "C", "u_out", "t_bin", "u_bin"]).index.map(means).to_numpy(dtype=float)
pred = np.where(np.isnan(pred), mu, pred)
log(f"B3 current+timebin (R,C,u_out,t,u) MAE: {insp_mae(va2['pressure'].to_numpy(), pred, (va2['u_out']==0).to_numpy()):.4f}")

# History features: cumulative u_in, lag u_in
def add_hist(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["breath_id", "time_step"]).copy()
    g = df.groupby("breath_id", sort=False)
    df["u_in_lag1"] = g["u_in"].shift(1).fillna(0)
    df["u_in_lag2"] = g["u_in"].shift(2).fillna(0)
    df["u_in_cumsum"] = g["u_in"].cumsum()
    df["u_in_cummax"] = g["u_in"].cummax()
    df["step"] = g.cumcount()
    df["du"] = df["u_in"] - df["u_in_lag1"]
    return df


trh = add_hist(tr)
vah = add_hist(va)

# Simple linear regression on handcrafted features (inspiratory only fit)
from numpy.linalg import lstsq

feat_cols = ["R", "C", "u_in", "u_out", "time_step", "u_in_lag1", "u_in_lag2",
             "u_in_cumsum", "u_in_cummax", "step", "du"]


def fit_predict_lin(tr_df, va_df, cols, insp_only_train: bool):
    if insp_only_train:
        fit = tr_df[tr_df["u_out"] == 0]
    else:
        fit = tr_df
    X = fit[cols].to_numpy(dtype=float)
    # add intercept + interactions R*u_in, C*u_in, R*cumsum, C*cumsum
    extra = np.column_stack([
        np.ones(len(fit)),
        fit["R"] * fit["u_in"],
        fit["C"] * fit["u_in"],
        fit["R"] * fit["u_in_cumsum"],
        fit["C"] * fit["u_in_cumsum"],
        fit["u_in"] / (fit["C"] + 1e-6),
        fit["u_in_cumsum"] / (fit["C"] + 1e-6),
        fit["u_in"] * fit["R"] / (fit["C"] + 1e-6),
    ])
    X = np.hstack([X, extra])
    y = fit["pressure"].to_numpy(dtype=float)
    coef, *_ = lstsq(X, y, rcond=None)

    Xv = va_df[cols].to_numpy(dtype=float)
    extrav = np.column_stack([
        np.ones(len(va_df)),
        va_df["R"] * va_df["u_in"],
        va_df["C"] * va_df["u_in"],
        va_df["R"] * va_df["u_in_cumsum"],
        va_df["C"] * va_df["u_in_cumsum"],
        va_df["u_in"] / (va_df["C"] + 1e-6),
        va_df["u_in_cumsum"] / (va_df["C"] + 1e-6),
        va_df["u_in"] * va_df["R"] / (va_df["C"] + 1e-6),
    ])
    Xv = np.hstack([Xv, extrav])
    return Xv @ coef


pred_curr = fit_predict_lin(
    trh.assign(**{c: trh[c] for c in feat_cols}),
    vah,
    ["R", "C", "u_in", "u_out", "time_step"],
    insp_only_train=True,
)
# recompute properly for current-only
def lin_mae(cols, insp_only_train=True):
    pred = fit_predict_lin(trh, vah, cols, insp_only_train)
    return insp_mae(vah["pressure"].to_numpy(), pred, (vah["u_out"] == 0).to_numpy())


log(f"L1 linear current-only (R,C,u_in,u_out,t) MAE: {lin_mae(['R','C','u_in','u_out','time_step']):.4f}")
log(f"L2 linear +lags MAE: {lin_mae(['R','C','u_in','u_out','time_step','u_in_lag1','u_in_lag2','du']):.4f}")
log(f"L3 linear +cumsum/cummax MAE: {lin_mae(feat_cols):.4f}")
log(f"L3b linear +hist trained on ALL phases MAE: {lin_mae(feat_cols, insp_only_train=False):.4f}")

# Lag-1 pressure oracle (upper bound on autoregression usefulness; not usable at test)
vah2 = vah.copy()
vah2["p_lag1"] = vah2.groupby("breath_id")["pressure"].shift(1)
# Only where lag exists and insp
m = (vah2["u_out"] == 0) & vah2["p_lag1"].notna()
log(f"Oracle: predict p_lag1 (persistence) insp MAE: {np.mean(np.abs(vah2.loc[m,'pressure']-vah2.loc[m,'p_lag1'])):.4f}")

# ---------------------------------------------------------------------------
# 4. Consistency across comparable trajectories
# ---------------------------------------------------------------------------
log()
log("=" * 72)
log("4. RESPONSE CONSISTENCY FOR SIMILAR CONTROLS")
log("=" * 72)

# Fingerprint: R,C + rounded first 30 u_in values during inspiration-ish
def fingerprint(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for bid, g in df.groupby("breath_id"):
        g = g.sort_values("time_step")
        u = g["u_in"].to_numpy()[:30]
        # coarse fingerprint
        key = (int(g["R"].iloc[0]), int(g["C"].iloc[0]), tuple(np.round(u, 1)))
        rows.append({"breath_id": bid, "R": key[0], "C": key[1], "fp": key[2],
                     "p_insp": g.loc[g["u_out"] == 0, "pressure"].to_numpy()})
    return pd.DataFrame(rows)


# Sample for speed: 8000 breaths
sample_ids = rng.choice(list(tr_ids), size=min(8000, len(tr_ids)), replace=False)
sub = train[train["breath_id"].isin(sample_ids)]
fp = fingerprint(sub)
# Find duplicated fingerprints
vc = fp["fp"].value_counts()
dups = vc[vc >= 2]
log(f"fingerprints with >=2 breaths (among {len(fp)} sampled): {len(dups)}")
if len(dups) > 0:
    # For each dup group, MAE of each breath vs group-mean pressure trajectory
    maes = []
    for fval in dups.index[:200]:
        group = fp[fp["fp"] == fval]
        # align lengths
        ps = [p for p in group["p_insp"] if len(p) > 0]
        if len(ps) < 2:
            continue
        L = min(map(len, ps))
        arr = np.stack([p[:L] for p in ps])
        mean_traj = arr.mean(axis=0)
        for i in range(arr.shape[0]):
            maes.append(np.mean(np.abs(arr[i] - mean_traj)))
    if maes:
        log(f"within-fingerprint pressure MAE to mean traj: mean={np.mean(maes):.4f} "
            f"median={np.median(maes):.4f} p90={np.quantile(maes,0.9):.4f} n={len(maes)}")

# Also: identical rounded full u_in sequences
log("Checking exact duplicate control sequences (rounded u_in to 3dp, same R,C)...")
# vectorized-ish via hashing first 80 u_in
recs = []
for bid, g in sub.groupby("breath_id"):
    g = g.sort_values("time_step")
    key = (int(g.R.iloc[0]), int(g.C.iloc[0]), tuple(np.round(g.u_in.to_numpy(), 3)))
    recs.append((bid, key, g.loc[g.u_out == 0, "pressure"].to_numpy()))
from collections import defaultdict
buckets = defaultdict(list)
for bid, key, p_insp in recs:
    buckets[key].append(p_insp)
multi = {k: v for k, v in buckets.items() if len(v) >= 2}
log(f"exact (R,C,u_in@3dp) duplicates: {len(multi)} groups")
maes = []
for ps in list(multi.values())[:300]:
    L = min(map(len, ps))
    arr = np.stack([p[:L] for p in ps])
    mean_traj = arr.mean(0)
    for i in range(len(arr)):
        maes.append(np.mean(np.abs(arr[i] - mean_traj)))
if maes:
    log(f"exact-dup consistency MAE: mean={np.mean(maes):.4f} median={np.median(maes):.4f} n={len(maes)}")
else:
    log("No exact duplicates found in sample.")

# ---------------------------------------------------------------------------
# 5. Simple physical / dynamical model
# ---------------------------------------------------------------------------
log()
log("=" * 72)
log("5. SIMPLIFIED PHYSICAL / DYNAMICAL MODELS")
log("=" * 72)

# Discrete integrator: volume ~= cumsum(u_in * dt * (1-u_out?)) ; P ~= V/C + R*flow
# Use linear model with physics-inspired features only


def physics_features(df: pd.DataFrame) -> np.ndarray:
    df = df.sort_values(["breath_id", "time_step"]).copy()
    g = df.groupby("breath_id", sort=False)
    dt = g["time_step"].diff().fillna(0.033)
    # approximate flow related to u_in when u_out=0; leak when u_out=1
    flow_in = df["u_in"]  # proportional unknown
    # volume proxy
    vol = (flow_in * dt * (1 - df["u_out"] * 0.0)).groupby(df["breath_id"]).cumsum()  # keep all for now
    # better: accumulate u_in*dt always, subtract when u_out
    vol2 = (df["u_in"] * dt).groupby(df["breath_id"]).cumsum()
    # exponential decay volume when u_out
    # simple linear features:
    feats = np.column_stack([
        np.ones(len(df)),
        df["u_in"].to_numpy(),
        (df["u_in"] / df["C"]).to_numpy(),
        (df["u_in"] * df["R"] / df["C"]).to_numpy(),
        (vol2 / df["C"]).to_numpy(),
        (vol2 * df["R"] / df["C"]).to_numpy(),
        df["u_out"].to_numpy(),
        df["time_step"].to_numpy(),
        (df["R"]).to_numpy(),
        (df["C"]).to_numpy(),
        ((df["u_in"] - g["u_in"].shift(1).fillna(0)) / df["C"]).to_numpy(),
    ])
    return feats


# Need aligned features for trh/vah with same order
tr_sorted = trh.sort_values(["breath_id", "time_step"])
va_sorted = vah.sort_values(["breath_id", "time_step"])
Xtr = physics_features(tr_sorted)
ytr = tr_sorted["pressure"].to_numpy()
# fit on inspiratory
mtr = tr_sorted["u_out"].to_numpy() == 0
coef, *_ = lstsq(Xtr[mtr], ytr[mtr], rcond=None)
Xva = physics_features(va_sorted)
pred = Xva @ coef
log(f"P1 physics-inspired linear (insp-train) MAE: "
    f"{insp_mae(va_sorted['pressure'].to_numpy(), pred, (va_sorted['u_out']==0).to_numpy()):.4f}")

# First-order recursive: p_t = a*p_{t-1} + b*u_in + ... fit per (R,C) on train
# Teacher-forced fit; evaluate open-loop from p0


def fit_arx_by_rc(tr_df, va_df):
    results = {}
    preds = np.zeros(len(va_df))
    va_df = va_df.copy()
    va_df["_i"] = np.arange(len(va_df))
    for (r, c), gtr in tr_df.groupby(["R", "C"]):
        gtr = gtr.sort_values(["breath_id", "time_step"])
        # build teacher-forced design
        rows = []
        ys = []
        for _, gb in gtr.groupby("breath_id"):
            gb = gb.sort_values("time_step")
            p = gb["pressure"].to_numpy()
            u = gb["u_in"].to_numpy()
            uo = gb["u_out"].to_numpy()
            t = gb["time_step"].to_numpy()
            for i in range(1, len(gb)):
                if uo[i] != 0:
                    continue  # fit on insp only
                rows.append([p[i - 1], u[i], u[i - 1], uo[i], t[i], 1.0])
                ys.append(p[i])
        if len(ys) < 100:
            continue
        A = np.asarray(rows)
        coef, *_ = lstsq(A, np.asarray(ys), rcond=None)
        results[(r, c)] = coef
    # open-loop predict on val
    for (r, c), gva in va_df.groupby(["R", "C"]):
        coef = results.get((r, c))
        if coef is None:
            continue
        for _, gb in gva.groupby("breath_id"):
            gb = gb.sort_values("time_step")
            idx = gb["_i"].to_numpy()
            u = gb["u_in"].to_numpy()
            uo = gb["u_out"].to_numpy()
            t = gb["time_step"].to_numpy()
            p_true = gb["pressure"].to_numpy()
            p_hat = np.zeros(len(gb))
            p_hat[0] = p_true[0]  # use first true pressure as init (slight leakage of init condition)
            # Better: use train mean initial pressure for (R,C)
            for i in range(1, len(gb)):
                x = np.array([p_hat[i - 1], u[i], u[i - 1], uo[i], t[i], 1.0])
                p_hat[i] = x @ coef
            preds[idx] = p_hat
    # init with mean first pressure of train for that R,C to reduce leakage
    return preds


# Init-free version: set p0 = mean first insp pressure by R,C
init_p = tr.groupby(["R", "C", "breath_id"]).first().reset_index().groupby(["R", "C"])["pressure"].mean()


def fit_arx_openloop(tr_df, va_df):
    preds = np.zeros(len(va_df))
    va_df = va_df.copy()
    va_df["_i"] = np.arange(len(va_df))
    coefs = {}
    for (r, c), gtr in tr_df.groupby(["R", "C"]):
        rows, ys = [], []
        for _, gb in gtr.groupby("breath_id"):
            gb = gb.sort_values("time_step")
            p = gb["pressure"].to_numpy()
            u = gb["u_in"].to_numpy()
            uo = gb["u_out"].to_numpy()
            t = gb["time_step"].to_numpy()
            for i in range(1, len(gb)):
                if uo[i] != 0:
                    continue
                rows.append([p[i - 1], u[i], u[i - 1], u[i] - u[i - 1], uo[i], 1.0])
                ys.append(p[i])
        A = np.asarray(rows, dtype=float)
        coef, *_ = lstsq(A, np.asarray(ys, dtype=float), rcond=None)
        coefs[(r, c)] = coef
    for (r, c), gva in va_df.groupby(["R", "C"]):
        coef = coefs[(r, c)]
        p0 = float(init_p.loc[(r, c)])
        for _, gb in gva.groupby("breath_id"):
            gb = gb.sort_values("time_step")
            idx = gb["_i"].to_numpy()
            u = gb["u_in"].to_numpy()
            uo = gb["u_out"].to_numpy()
            p_hat = np.zeros(len(gb))
            p_hat[0] = p0
            for i in range(1, len(gb)):
                x = np.array([p_hat[i - 1], u[i], u[i - 1], u[i] - u[i - 1], uo[i], 1.0])
                p_hat[i] = x @ coef
            preds[idx] = p_hat
    return preds


pred_arx = fit_arx_openloop(tr, va)
log(f"P2 open-loop ARX per (R,C) MAE: "
    f"{insp_mae(va['pressure'].to_numpy(), pred_arx, (va['u_out']==0).to_numpy()):.4f}")

# ---------------------------------------------------------------------------
# 6. Difficulty by R/C and phase
# ---------------------------------------------------------------------------
log()
log("=" * 72)
log("6. DIFFICULTY BY R/C AND BREATH PHASE")
log("=" * 72)

# Use L3 predictions
pred_l3 = fit_predict_lin(trh, vah, feat_cols, insp_only_train=True)
vah = vah.copy()
vah["pred"] = pred_l3
vah["abs_err"] = np.abs(vah["pressure"] - vah["pred"])
insp = vah[vah["u_out"] == 0]
log("MAE by (R,C) for L3 linear+hist:")
tab = insp.groupby(["R", "C"])["abs_err"].mean().unstack()
log(tab.to_string())

# By step within breath
log("MAE by step index (inspiratory only), selected steps:")
insp2 = insp.copy()
insp2["step"] = insp2.groupby("breath_id").cumcount()
by_step = insp2.groupby("step")["abs_err"].mean()
for s in [0, 1, 2, 5, 10, 15, 20, 25, 30]:
    if s in by_step.index:
        log(f"  step {s}: MAE={by_step.loc[s]:.4f}")

# Early vs late inspiration
med_insp = int(insp_len.median())
insp2["phase"] = np.where(insp2["step"] < med_insp / 2, "early_insp", "late_insp")
log(f"early vs late insp MAE: {insp2.groupby('phase')['abs_err'].mean().to_dict()}")

# u_in magnitude difficulty
insp2["u_q"] = pd.qcut(insp2["u_in"], 5, duplicates="drop")
log(f"MAE by u_in quintile:\n{insp2.groupby('u_q', observed=True)['abs_err'].mean().to_string()}")

# Pressure delta magnitude
insp2["dp"] = insp2.groupby("breath_id")["pressure"].diff().abs()
log(f"corr(|err|, |dp|): {insp2['abs_err'].corr(insp2['dp']):.4f}")

# ---------------------------------------------------------------------------
# Save artifacts
# ---------------------------------------------------------------------------
report = "\n".join(lines)
(OUT / "eda_batch0_report.txt").write_text(report, encoding="utf-8")
log()
log(f"Wrote {OUT / 'eda_batch0_report.txt'}")
