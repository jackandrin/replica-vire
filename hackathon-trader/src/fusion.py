"""Fusion network (level 2): 32 inputs -> 64 -> 32 -> 16 -> (mu, log-variance) for each of 6 horizons.

Inputs = 15 out-of-fold regressor predictions + 4 sentiment features + 13 context features.
Loss = Gaussian negative log-likelihood summed over horizons (rows with a missing label are masked).
Attribution = integrated gradients (in-house, no captum dependency) for the dashboard's "why".
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .regressors import Prep


class FusionNet(nn.Module):
    def __init__(self, d_in: int, n_out: int, hidden=(64, 32, 16), dropout=0.2):
        super().__init__()
        layers, d = [], d_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            d = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(d, 2 * n_out)
        self.n_out = n_out

    def forward(self, x):
        o = self.head(self.body(x))
        return o[:, : self.n_out], o[:, self.n_out:].clamp(-10, 6)  # mu, log-variance (scaled units)


def gaussian_nll(mu, lv, y, mask):
    per = 0.5 * (lv + (y - mu) ** 2 * torch.exp(-lv))
    return (per * mask).sum() / mask.sum().clamp(min=1)


class FusionModel:
    def __init__(self, inputs: list, horizons: list, hidden=(64, 32, 16), dropout=0.2, lr=1e-3,
                 weight_decay=1e-4, batch_size=1024, epochs=200, patience=15, seed=42, device=None):
        self.inputs, self.horizons = list(inputs), list(horizons)
        self.hp = dict(hidden=tuple(hidden), dropout=dropout)
        self.lr, self.wd, self.bs, self.epochs, self.patience, self.seed = lr, weight_decay, batch_size, epochs, patience, seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.history = []

    def _y(self, Y: np.ndarray):
        Yc = np.clip(Y, self.ylo, self.yhi)
        mask = np.isfinite(Yc)
        return np.where(mask, Yc, 0.0) / self.y_scale, mask.astype(np.float32)

    def fit(self, Xtr, Ytr, Xva, Yva, log=None):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        dev = torch.device(self.device)
        self.prep = Prep().fit(Xtr)
        self.ylo, self.yhi = np.nanpercentile(Ytr, 0.5, axis=0), np.nanpercentile(Ytr, 99.5, axis=0)
        self.y_scale = np.nanstd(np.clip(Ytr, self.ylo, self.yhi), axis=0)
        self.y_scale[~np.isfinite(self.y_scale) | (self.y_scale <= 0)] = 1.0
        t = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)  # noqa: E731
        Xf, Xv = t(self.prep.transform(Xtr)), t(self.prep.transform(Xva))
        yf, mf = map(t, self._y(Ytr))
        yv, mv = map(t, self._y(Yva))
        self.net = FusionNet(Xf.shape[1], len(self.horizons), **self.hp).to(dev)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=max(2, self.patience // 3))
        best, best_state, bad = np.inf, None, 0
        for ep in range(self.epochs):
            self.net.train()
            perm = torch.randperm(len(Xf), device=dev)
            tot = 0.0
            for i in range(0, len(Xf), self.bs):
                idx = perm[i:i + self.bs]
                opt.zero_grad()
                loss = gaussian_nll(*self.net(Xf[idx]), yf[idx], mf[idx])
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                tot += float(loss.detach()) * len(idx)
            self.net.eval()
            with torch.no_grad():
                v = float(gaussian_nll(*self.net(Xv), yv, mv))
            sched.step(v)
            self.history.append({"epoch": ep, "train_nll": tot / len(Xf), "val_nll": v})
            if log and ep % 10 == 0:
                log.info("epoch %3d  train_nll=%.4f  val_nll=%.4f", ep, tot / len(Xf), v)
            if v < best - 1e-5:
                best, bad, self.best_epoch = v, 0, ep
                best_state = {k: x.detach().clone() for k, x in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.net.load_state_dict(best_state)
        self.net.cpu()  # store/predict on CPU so the pickle loads anywhere
        self.device, self.val_nll = "cpu", best
        return self

    def predict(self, X: np.ndarray):
        """Returns (mu, sigma) in return units, each shape (n, n_horizons)."""
        self.net.eval()
        with torch.no_grad():
            mu, lv = self.net(torch.tensor(self.prep.transform(X), dtype=torch.float32))
        return mu.numpy() * self.y_scale, np.exp(0.5 * lv.numpy()) * self.y_scale

    def attribute(self, X: np.ndarray, horizon_idx: np.ndarray, steps: int = 32) -> np.ndarray:
        """Integrated gradients of mu[horizon] w.r.t. each standardized input, baseline = train mean (zeros).
        Returns (n, n_inputs) in return units; rows sum approximately to mu(x) - mu(baseline)."""
        self.net.eval()
        x = torch.tensor(self.prep.transform(X), dtype=torch.float32)
        hi = torch.tensor(np.array(horizon_idx, dtype=np.int64))
        total = torch.zeros_like(x)
        for a in torch.linspace(1.0 / steps, 1.0, steps):
            xi = (a * x).requires_grad_(True)
            mu, _ = self.net(xi)
            sel = mu.gather(1, hi.view(-1, 1)).sum()
            total += torch.autograd.grad(sel, xi)[0]
        attr = (x * total / steps).numpy()
        return attr * self.y_scale[np.array(horizon_idx, dtype=np.int64)][:, None]
