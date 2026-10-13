import numpy as np

from lookahead import eventstudy as es
from lookahead import sim
from lookahead.probes import pvi_matrix


def _panel(seed, gamma, n_coins=60):
    P = sim.simulate_panel(np.random.default_rng(seed), n_coins, gamma, "coverage")
    yhat, E = es.surprise(P["H"][0], P["y"], P["coin"], P["week"], tail=0.05)
    pvi = pvi_matrix(P["H"], P["y"], P["coin"])
    return P, yhat, E, pvi


def test_detects_planted_leak():
    P, yhat, E, pvi = _panel(1, gamma=0.4)
    r = es.estimate(pvi, sim.VINTAGES, P["year"], E, P["coin"], P["week"], yhat, P["cov"])
    assert r["reject_h1"] and r["beta0"] > 0


def test_null_is_not_systematically_rejected():
    rej = []
    for s in range(6):
        P, yhat, E, pvi = _panel(100 + s, gamma=0.0)
        rej.append(es.estimate(pvi, sim.VINTAGES, P["year"], E, P["coin"], P["week"], yhat, P["cov"])["reject_h1"])
    assert sum(rej) <= 2


def test_twoway_runs_and_is_finite():
    P, yhat, E, pvi = _panel(2, gamma=0.2)
    r = es.estimate(pvi, sim.VINTAGES, P["year"], E, P["coin"], P["week"], yhat, P["cov"], cluster="twoway")
    assert np.isfinite(r["se"]).all()


def test_demean_removes_item_and_vintage_week_effects():
    rng = np.random.default_rng(0)
    n, V = 200, 5
    week = rng.integers(0, 20, n)
    A = rng.normal(size=(n, 1)) + rng.normal(size=(20, V))[week]
    D = es._demean(A, week)
    assert np.abs(D).max() < 1e-10
