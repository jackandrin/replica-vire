"""Ridge baseline + tiny MLP that outputs [mu, sigma] (Gaussian NLL)."""
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn


# ---------------------------------------------------------------- ridge
class RidgeModel:
    def __init__(self, alphas):
        self.alphas = alphas
        self.pipe = None
        self.alpha = None

    def fit(self, Xfit, yfit, Xval, yval):
        best = None
        for a in self.alphas:
            p = make_pipeline(StandardScaler(), Ridge(alpha=a)).fit(Xfit, yfit)
            mse = float(np.mean((p.predict(Xval) - yval) ** 2))
            if best is None or mse < best[0]:
                best = (mse, a)
        self.alpha = best[1]
        return self

    def refit(self, X, y):
        self.pipe = make_pipeline(StandardScaler(), Ridge(alpha=self.alpha)).fit(X, y)
        return self

    def predict(self, X):
        return self.pipe.predict(X), None


# ---------------------------------------------------------------- mlp
class _Net(nn.Module):
    def __init__(self, d, h1, h2, dropout):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(d, h1), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(h1, h2), nn.ReLU(), nn.Dropout(dropout))
        self.head = nn.Linear(h2, 2)

    def forward(self, x):
        o = self.head(self.body(x))
        return o[:, 0], o[:, 1].clamp(-8, 4)  # mu, log-variance


class MLPModel:
    def __init__(self, h1=32, h2=16, dropout=0.2, lr=1e-3, weight_decay=1e-3, epochs=300, patience=20, seed=42,
                 use_gpu=False):
        self.use_gpu = use_gpu
        self.hp = dict(h1=h1, h2=h2, dropout=dropout)
        self.lr, self.wd, self.epochs, self.patience, self.seed = lr, weight_decay, epochs, patience, seed
        self.scaler = StandardScaler()
        self.net = None
        self.y_scale = 1.0
        self.best_epoch = 0

    def fit(self, Xfit, yfit, Xval, yval):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        dev = torch.device("cuda" if self.use_gpu and torch.cuda.is_available() else "cpu")
        self.device_used = str(dev)
        self.y_scale = float(np.std(yfit)) or 1.0
        Xf = torch.tensor(self.scaler.fit_transform(Xfit), dtype=torch.float32, device=dev)
        Xv = torch.tensor(self.scaler.transform(Xval), dtype=torch.float32, device=dev)
        yf = torch.tensor(yfit / self.y_scale, dtype=torch.float32, device=dev)
        yv = torch.tensor(yval / self.y_scale, dtype=torch.float32, device=dev)
        self.net = _Net(Xf.shape[1], **self.hp).to(dev)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.wd)

        def nll(mu, lv, y):
            return 0.5 * (lv + (y - mu) ** 2 * torch.exp(-lv)).mean()

        best, best_state, bad = 1e9, None, 0
        for ep in range(self.epochs):
            self.net.train()
            perm = torch.randperm(len(Xf), device=dev)
            for i in range(0, len(Xf), 256):
                idx = perm[i:i + 256]
                opt.zero_grad()
                mu, lv = self.net(Xf[idx])
                nll(mu, lv, yf[idx]).backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                v = float(nll(*self.net(Xv), yv))
            if v < best - 1e-5:
                best, bad, self.best_epoch = v, 0, ep
                best_state = {k: t.clone() for k, t in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.net.load_state_dict(best_state)
        self.net.cpu()  # always store/predict on CPU so the pickle loads anywhere
        self.val_nll = best
        return self

    def predict(self, X):
        self.net.eval()
        with torch.no_grad():
            mu, lv = self.net(torch.tensor(self.scaler.transform(X), dtype=torch.float32))
        return mu.numpy() * self.y_scale, np.exp(0.5 * lv.numpy()) * self.y_scale
