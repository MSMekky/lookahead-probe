import numpy as np

from lookahead.probes import crossfit
from lookahead.remedy import fit_eraser, residual_transform


def test_eraser_removes_linear_information_about_the_concept():
    rng = np.random.default_rng(0)
    n, d = 4000, 20
    X = rng.normal(size=(n, d))
    r = X @ rng.normal(size=d) + rng.normal(size=n)
    Xe = fit_eraser(X, r)(X)
    c = (Xe - Xe.mean(0)).T @ (r - r.mean()) / n
    assert np.abs(c).max() < 1e-8


def test_residual_erasure_keeps_legitimate_signal_and_removes_leak():
    rng = np.random.default_rng(1)
    n, d = 6000, 16
    coin = np.repeat(np.arange(300), n // 300)
    clean = rng.normal(size=n)                    # what the clean model can predict
    leak = rng.normal(size=n)                     # what only the future can explain
    y = clean + leak + 0.5 * rng.normal(size=n)
    X = np.c_[clean, leak, rng.normal(size=(n, d - 2))] + 0.1 * rng.normal(size=(n, d))
    yhat_clean = clean.copy()
    before = crossfit(X, y, coin)
    after = crossfit(X, y, coin, transform=residual_transform(yhat_clean))
    corr = lambda a, b: np.corrcoef(a, b)[0, 1]
    assert corr(before["yhat"], leak) > 0.5
    assert abs(corr(after["yhat"], leak)) < 0.15
    assert corr(after["yhat"], clean) > 0.8
