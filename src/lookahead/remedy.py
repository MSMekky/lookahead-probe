"""Residual LEACE (H5): erase only the part of the outcome that the clean reference cannot predict.

Standard LEACE (Belrose et al. 2023) erasing y itself would also remove legitimate predictability.
Here the erased concept is the residual r = y - yhat_clean, where yhat_clean is the clean
reference vintage's cross-fitted prediction from the same text. For a scalar concept the LEACE
eraser is rank one:  x' = x - S^{1/2} u u' S^{-1/2} (x - mu),  u = S^{-1/2} c / ||S^{-1/2} c||,
with S = Cov(x) and c = Cov(x, r). After erasure no linear probe can predict r better than a constant
on the fitting sample, while every direction uncorrelated with r is left intact.
The eraser is fitted on training folds only and applied to held-out items.
"""
from __future__ import annotations

import numpy as np

from .eventstudy import estimate, leak_gap
from .probes import crossfit


def fit_eraser(X: np.ndarray, r: np.ndarray, eps: float = 1e-6):
    mu = X.mean(0)
    Xc = X - mu
    S = Xc.T @ Xc / len(X)
    evals, V = np.linalg.eigh(S)
    evals = np.clip(evals, eps * evals.max(), None)
    W = (V / np.sqrt(evals)) @ V.T          # S^{-1/2}
    Winv = (V * np.sqrt(evals)) @ V.T       # S^{1/2}
    c = Xc.T @ (r - r.mean()) / len(X)
    u = W @ c
    u /= np.linalg.norm(u) + 1e-12
    A = Winv @ np.outer(u, u) @ W

    def apply(Z: np.ndarray) -> np.ndarray:
        return Z - (Z - mu) @ A.T
    return apply


def residual_transform(yhat_clean: np.ndarray):
    def transform(Z_tr, y_tr, Z_all, tr):
        r = y_tr - yhat_clean[tr]
        return fit_eraser(Z_tr, r)(Z_all)
    return transform


def evaluate(get_H, vintages: np.ndarray, items: dict, yhat_clean: np.ndarray, E: np.ndarray,
             before: np.ndarray) -> dict:
    """Re-score every vintage after residual erasure. Reports the event study after erasure and how much
    legitimate information survives, measured on item-vintage cells before exposure (k < 0)."""
    y, coin = items["y"], items["coin"]
    tf = residual_transform(yhat_clean)
    after = np.stack([crossfit(get_H(j), y, coin, transform=tf)["pvi"] for j in range(len(vintages))], 1)
    es = estimate(after, vintages, items["year"], E, coin, items["week"], yhat_clean, items["cov"])
    pre = (vintages[None, :] - items["year"][:, None]) < 0      # cells the model cannot have leaked into
    b, a = before[pre].mean(), after[pre].mean()
    return dict(
        event_study_after=es,
        leak_gap_before=leak_gap(before, vintages, items["year"], E),
        leak_gap_after=leak_gap(after, vintages, items["year"], E),
        mean_pvi_preexposure_before=float(b),
        mean_pvi_preexposure_after=float(a),
        # a ratio is only meaningful when there is legitimate information to retain
        retention_preexposure=float(a / b) if b > 0.005 else None,
        pvi_after=after,
    )
