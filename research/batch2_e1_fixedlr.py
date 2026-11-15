"""E1 follow-up: fixed high LR (no decay) + target scale — can we memorize 500 breaths?"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch.nn as nn

from batch2_common import (
    DATA, SEQ_COLS_BASE, BiLSTMReg, add_features, apply_norm, norm_fit,
    reshape_sequences, train_model,
)


def main():
    train = pd.read_csv(DATA / "train.csv")
    feat = add_features(train)
    rng = np.random.default_rng(0)
    ids = rng.choice(feat["breath_id"].unique(), size=500, replace=False)
    sub = feat[feat["breath_id"].isin(ids)]
    X, y, u, R, C, _ = reshape_sequences(sub, SEQ_COLS_BASE)
    Xn = apply_norm(X, *norm_fit(X))

    for label, y_scale, loss, lr, epochs, hidden, layers in [
        ("fix_mse_lr1e2_h384", True, "mse", 1e-2, 300, 384, 3),
        ("fix_mae_lr1e2_h384", True, "mae", 1e-2, 300, 384, 3),
        ("fix_mse_lr2e2_h512", True, "mse", 2e-2, 300, 512, 2),
        # identity overfit control: predict with a huge MLP on flattened insp feats
    ]:
        model = BiLSTMReg(Xn.shape[-1], hidden=hidden, layers=layers, dropout=0.0)
        res = train_model(
            model, Xn, y, u, Xn, y, u,
            R_tr=R, C_tr=C, R_va=R, C_va=C,
            epochs=epochs, lr=lr, batch_size=64, loss_kind=loss,
            schedule="none", y_scale=y_scale, patience=50,
            label=label, log_every=50, min_lr=lr,
        )
        print(f"RESULT {label}: best_train={res['best_train']:.4f} ep={res['best_epoch']}", flush=True)

    # MLP memorization control on flattened sequences
    print("MLP flatten control...", flush=True)
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from batch2_common import DEVICE, insp_mae, predict_np

    class FlatMLP(nn.Module):
        def __init__(self, n_in, n_out=80):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_in, 2048), nn.GELU(),
                nn.Linear(2048, 2048), nn.GELU(),
                nn.Linear(2048, n_out),
            )

        def forward(self, x):
            return self.net(x.reshape(x.size(0), -1))

    y_mean, y_std = float(y[u == 0].mean()), float(y[u == 0].std())
    yt = (y - y_mean) / y_std
    model = FlatMLP(Xn.shape[-1] * 80).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(Xn), torch.from_numpy(yt.astype(np.float32)), torch.from_numpy(u)),
        batch_size=64, shuffle=True,
    )
    best = 1e9
    for ep in range(1, 401):
        model.train()
        for xb, yb, ub in loader:
            xb, yb, ub = xb.to(DEVICE), yb.to(DEVICE), ub.to(DEVICE)
            pred = model(xb)
            m = ub == 0
            loss = ((pred - yb) ** 2)[m].mean()
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 50 == 0 or ep == 1:
            pred = predict_np(model, Xn) * y_std + y_mean
            mae = insp_mae(y.reshape(-1), pred.reshape(-1), u.reshape(-1))
            best = min(best, mae)
            print(f"  [mlp_flat] ep={ep} train_insp_mae={mae:.4f}", flush=True)
    print(f"RESULT mlp_flat: best_train={best:.4f}", flush=True)


if __name__ == "__main__":
    main()
