"""Ridge probes and pointwise V-information (PVI), cross-fitted with coin-held-out folds.

For item i and vintage v, PVI is the per-item Gaussian log-likelihood gain of a ridge probe on the
hidden state over the null probe that predicts the training mean, with the residual variance
estimated on the training fold:  PVI_i = [(y_i - ybar)^2 - (y_i - yhat_i)^2] / (2 s^2).
Every item is scored only by a probe that never saw its coin.
"""
from __future__ import annotations

import numpy as np

from .config import CFG, ProbeConfig


def coin_folds(coin: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    uniq = np.unique(coin)
    f = np.random.default_rng(seed).permutation(len(uniq)) % n_folds
    return f[np.searchsorted(uniq, coin)]


def _ridge_path(X: np.ndarray, y: np.ndarray, alphas) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit ridge for all alphas via one eigendecomposition; choose alpha by GCV. Returns (w, mu, ybar)."""
    mu = X.mean(0); ybar = y.mean()
    Xc = X - mu; yc = y - ybar
    n, d = Xc.shape
    if n >= d:
        evals, V = np.linalg.eigh(Xc.T @ Xc)
        evals = np.clip(evals, 0, None)
        b = V.T @ (Xc.T @ yc)
        best, best_gcv = None, np.inf
        for a in alphas:
            w = V @ (b / (evals + a))
            df = (evals / (evals + a)).sum()
            r = yc - Xc @ w
            gcv = (r @ r / n) / (1 - df / n) ** 2
            if gcv < best_gcv:
                best, best_gcv = w, gcv
    else:
        U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
        uty = U.T @ yc
        best, best_gcv = None, np.inf
        for a in alphas:
            w = Vt.T @ (s / (s ** 2 + a) * uty)
            df = (s ** 2 / (s ** 2 + a)).sum()
            r = yc - Xc @ w
            gcv = (r @ r / n) / (1 - df / n) ** 2
            if gcv < best_gcv:
                best, best_gcv = w, gcv
    return best, mu, ybar


def standardise(H: np.ndarray, tr: np.ndarray) -> np.ndarray:
    m = H[tr].mean(0); s = H[tr].std(0) + 1e-6
    return (H - m) / s


def crossfit(H: np.ndarray, y: np.ndarray, coin: np.ndarray, pc: ProbeConfig = CFG.probe,
             transform=None) -> dict:
    """Cross-fitted predictions and PVI for one representation H (n, d).
    `transform(H_train, y_train, H_all) -> H_all'` is applied inside each fold (used by the remedy)."""
    fold = coin_folds(coin, pc.n_folds, pc.seed)
    yhat = np.zeros_like(y, dtype=float)
    pvi = np.zeros_like(y, dtype=float)
    for f in range(pc.n_folds):
        tr, te = fold != f, fold == f
        Z = standardise(H.astype(np.float64), tr)
        if transform is not None:
            Z = transform(Z[tr], y[tr], Z, tr)
        w, mu, ybar = _ridge_path(Z[tr], y[tr], pc.alphas)
        s2 = np.var(y[tr] - ybar - (Z[tr] - mu) @ w)
        yh = ybar + (Z[te] - mu) @ w
        yhat[te] = yh
        pvi[te] = ((y[te] - ybar) ** 2 - (y[te] - yh) ** 2) / (2 * s2)
    return dict(yhat=yhat, pvi=pvi, fold=fold)


def pvi_matrix(Hs: list[np.ndarray], y: np.ndarray, coin: np.ndarray, pc: ProbeConfig = CFG.probe) -> np.ndarray:
    return np.stack([crossfit(H, y, coin, pc)["pvi"] for H in Hs], 1)


def selectivity(H: np.ndarray, y: np.ndarray, coin: np.ndarray, pc: ProbeConfig = CFG.probe, n_perm: int = 5) -> dict:
    """Control task (Hewitt and Liang): same probe on randomly permuted labels.
    Reports mean PVI on the real task, on shuffled labels, and their difference."""
    real = crossfit(H, y, coin, pc)["pvi"].mean()
    rng = np.random.default_rng(pc.seed)
    ctrl = [crossfit(H, rng.permutation(y), coin, pc)["pvi"].mean() for _ in range(n_perm)]
    return dict(pvi=float(real), control=float(np.mean(ctrl)), selectivity=float(real - np.mean(ctrl)))
