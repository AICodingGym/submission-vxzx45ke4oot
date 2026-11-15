"""Stronger BiLSTM memorization probe — clarify whether overfit failure is a bug."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from batch1_experiments import BiLSTM, add_features, insp_mae, reshape_sequences

DATA = Path(__file__).resolve().parents[1] / "data" / "ventilator-pressure-prediction"
SEED = 0


def predict(model, X, device):
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            outs.append(model(torch.from_numpy(X[i : i + 256]).to(device)).cpu().numpy())
    return np.concatenate(outs, axis=0)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = pd.read_csv(DATA / "train.csv")
    rng = np.random.default_rng(SEED)
    ids = rng.choice(train["breath_id"].unique(), size=500, replace=False)
    feat = add_features(train[train["breath_id"].isin(ids)])
    cols = [
        "u_in", "u_out", "R", "C", "time_step", "R_div_C", "u_over_c",
        "u_in_lag1", "u_in_cumsum", "u_in_cummax", "du", "insp_len",
        "steps_to_exp", "u_in_backcum", "vol_over_c",
    ]
    X, y, u = reshape_sequences(feat, cols)
    mean, std = X.reshape(-1, X.shape[-1]).mean(0), X.reshape(-1, X.shape[-1]).std(0)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    Xn = ((X - mean.astype(np.float32)) / std).astype(np.float32)

    print(f"device={device} breaths={len(X)} feat={X.shape[-1]}", flush=True)

    def run(loss_name, epochs, hidden=256, layers=3, lr=2e-3):
        model = BiLSTM(Xn.shape[-1], hidden=hidden, layers=layers, dropout=0.0).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loader = DataLoader(
            TensorDataset(torch.from_numpy(Xn), torch.from_numpy(y), torch.from_numpy(u)),
            batch_size=64, shuffle=True,
        )
        t0 = time.time()
        for ep in range(1, epochs + 1):
            model.train()
            for xb, yb, ub in loader:
                xb, yb, ub = xb.to(device), yb.to(device), ub.to(device)
                pred = model(xb)
                m = ub == 0
                if loss_name == "mae":
                    loss = torch.abs(pred - yb)[m].mean()
                else:
                    loss = ((pred - yb) ** 2)[m].mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            if ep % 20 == 0 or ep == 1 or ep == epochs:
                pred = predict(model, Xn, device)
                mae = insp_mae(y.reshape(-1), pred.reshape(-1), u.reshape(-1))
                print(f"[{loss_name}] ep={ep:03d} train_insp_mae={mae:.4f}", flush=True)
        print(f"[{loss_name}] done sec={time.time()-t0:.1f}", flush=True)

    run("mae", epochs=150, lr=2e-3)
    run("mse", epochs=150, lr=2e-3)


if __name__ == "__main__":
    main()
