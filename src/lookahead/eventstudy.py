"""Stacked event study across vintages (the H1 test).

PVI_{i,v} = a_{v,week} + m_i + sum_k b_k 1[v - y(i) = k] E_i + sum_v (g_v E_i + controls_v' x_i) + e
k = -1 is the reference. b_0 is leakage at first exposure (one-sided test).
Specification adopted after the power simulation (docs/power_analysis.md):
  - events defined on the surprise against the clean reference vintage, not on the raw return;
  - vintage-by-event-status fixed effects;
  - vintage-specific slopes on |yhat_clean|, yhat_clean^2 and log coverage.
Standard errors: clustered by coin (default) or two-way by coin and week (Cameron, Gelbach, Miller).
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from .config import CFG
from .probes import coin_folds, _ridge_path, standardise


def surprise(H_ref: np.ndarray, y: np.ndarray, coin: np.ndarray, week: np.ndarray, tail: float,
             n_folds: int = 5, seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Cross-fitted prediction from the clean reference vintage and the surprise-defined event flag."""
    fold = coin_folds(coin, n_folds, seed)
    yhat = np.zeros_like(y, dtype=float)
    for f in range(n_folds):
        tr, te = fold != f, fold == f
        Z = standardise(H_ref.astype(np.float64), tr)
        w, mu, ybar = _ridge_path(Z[tr], y[tr], CFG.probe.alphas)
        yhat[te] = ybar + (Z[te] - mu) @ w
    s = y - yhat
    E = np.zeros(len(y), bool)
    for w_ in np.unique(week):
        idx = np.nonzero(week == w_)[0]
        lo, hi = np.quantile(s[idx], [tail, 1 - tail])
        E[idx[(s[idx] <= lo) | (s[idx] >= hi)]] = True
    return yhat, E


def _demean(A: np.ndarray, week: np.ndarray) -> np.ndarray:
    """Item FE, then vintage-by-week FE. Exact for this balanced design (each item seen by every
    vintage, week nested in item)."""
    A = A - A.mean(axis=1, keepdims=True)
    out = A.copy()
    for w in np.unique(week):
        s = week == w
        out[s] -= A[s].mean(axis=0, keepdims=True)
    return out


def _meat(X: np.ndarray, e: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, int]:
    _, inv = np.unique(groups, return_inverse=True)
    G = inv.max() + 1
    S = np.zeros((G, X.shape[1]))
    np.add.at(S, inv, X * e[:, None])
    return S.T @ S, G


def estimate(pvi: np.ndarray, vintages: np.ndarray, year: np.ndarray, E: np.ndarray, coin: np.ndarray,
             week: np.ndarray, yhat_clean: np.ndarray | None = None, cov: np.ndarray | None = None,
             vintage_by_event: bool = True, k_bins=CFG.k_bins, cluster: str = "coin") -> dict:
    n, V = pvi.shape
    k = vintages[None, :] - year[:, None]
    kb = np.clip(k, min(k_bins), max(k_bins))
    X = np.stack([((kb == kk) & E[:, None]).astype(float) for kk in k_bins], -1)
    Yd = _demean(pvi.astype(float), week)
    Xd = np.stack([_demean(X[..., j], week) for j in range(X.shape[-1])], -1)
    Yd = Yd.reshape(-1); Xd = Xd.reshape(-1, Xd.shape[-1])
    if vintage_by_event:
        cols = [E.astype(float)]
        if yhat_clean is not None:
            cols += [np.abs(yhat_clean), yhat_clean ** 2]
        if cov is not None:
            cols += [np.log1p(cov)]
        eye = np.eye(V)
        Z = np.stack([_demean(np.outer(c, eye[j]), week).reshape(-1) for c in cols for j in range(V)], -1)
        coef_y = np.linalg.lstsq(Z, Yd, rcond=None)[0]
        coef_x = np.linalg.lstsq(Z, Xd, rcond=None)[0]
        Yd = Yd - Z @ coef_y
        Xd = Xd - Z @ coef_x
    XtX_inv = np.linalg.pinv(Xd.T @ Xd)
    beta = XtX_inv @ Xd.T @ Yd
    e = Yd - Xd @ beta
    coin_r = np.repeat(coin, V); week_r = np.repeat(week, V); item_r = np.repeat(np.arange(n), V)
    M, G = _meat(Xd, e, coin_r)
    if cluster == "twoway":
        Mw, Gw = _meat(Xd, e, week_r)
        Mi, _ = _meat(Xd, e, item_r)
        M = M + Mw - Mi
        G = min(G, Gw)
    Vc = (G / (G - 1)) * XtX_inv @ M @ XtX_inv
    se = np.sqrt(np.clip(np.diag(Vc), 0, None))
    t = beta / se
    i0 = list(k_bins).index(0)
    pre = [list(k_bins).index(kk) for kk in k_bins if kk < -1]
    bp = beta[pre]; Vp = Vc[np.ix_(pre, pre)]
    wald = float(bp @ np.linalg.pinv(Vp) @ bp)
    df = G - 1  # small-G t reference for the headline test
    return dict(
        k=list(k_bins), beta=beta.tolist(), se=se.tolist(), t=t.tolist(),
        beta0=float(beta[i0]), se0=float(se[i0]), t0=float(t[i0]),
        p0_one_sided=float(stats.t.sf(t[i0], df)),
        beta0_upper95=float(beta[i0] + stats.t.ppf(0.95, df) * se[i0]),
        reject_h1=bool(stats.t.sf(t[i0], df) < CFG.alpha_h1),
        pretrend_wald=wald, pretrend_p=float(stats.chi2.sf(wald, len(pre))),
        n_items=int(n), n_events=int(E.sum()), n_clusters=int(G), cluster=cluster,
        spec=dict(vintage_by_event=vintage_by_event, predictability_controls=yhat_clean is not None,
                  coverage_control=cov is not None),
    )


def leak_gap(pvi: np.ndarray, vintages: np.ndarray, year: np.ndarray, E: np.ndarray) -> float:
    """Descriptive leak size in nats: mean PVI on events at k=0 minus k=-1."""
    k = vintages[None, :] - year[:, None]
    return float(pvi[E][k[E] == 0].mean() - pvi[E][k[E] == -1].mean())
