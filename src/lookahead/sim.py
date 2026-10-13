"""Power simulation (results: results/power_table.csv, docs/power_analysis.md) for the stacked event-study leakage design.

Synthetic coin-week panel probed in every model vintage. Leakage is planted
in hidden states of vintages that have seen the outcome year; capability and
recency confounds are planted too. For each simulated panel we:
  1. train ridge probes per vintage with coin-held-out GroupKFold,
  2. compute pointwise V-information (PVI) per item on held-out folds,
  3. estimate the stacked event study with vintage-by-week and item fixed
     effects, coin-clustered SEs,
  4. record beta_0 rejection (H1) and the joint pre-trend test.
"""
import numpy as np
from scipy import stats

VINTAGES = np.arange(2018, 2025)          # 7 vintages
YEARS = np.array([2021, 2022, 2023])      # outcome years
WEEKS_PER_YEAR = 52
K_BINS = [-3, -2, 0, 1, 2]                # -1 is the reference


def simulate_panel(rng, n_coins, gamma, recency_mode="uniform", d=32, D=48,
                   presence=0.6, r2_text=0.05, event_q=0.05):
    # ---- items ---------------------------------------------------------
    n_weeks = len(YEARS) * WEEKS_PER_YEAR
    mask = rng.random((n_coins, n_weeks)) < presence
    coin, week = np.nonzero(mask)
    n = len(coin)
    year = YEARS[week // WEEKS_PER_YEAR]

    x = rng.standard_normal((n, d))
    b = rng.standard_normal(d)
    b *= np.sqrt(r2_text / (1 - r2_text)) / np.linalg.norm(b)
    y = x @ b + rng.standard_normal(n)
    # market-neutral: standardize within week
    for w in np.unique(week):
        s = week == w
        y[s] = (y[s] - y[s].mean()) / (y[s].std() + 1e-9)

    # events: both tails of the weekly cross-section
    E = np.zeros(n, bool)
    for w in np.unique(week):
        s = np.nonzero(week == w)[0]
        thr = np.quantile(np.abs(y[s]), 1 - 2 * event_q)
        E[s[np.abs(y[s]) >= thr]] = True

    # coverage: coin-level log-normal, boosted for events
    coin_cov = np.exp(rng.normal(0, 0.8, n_coins))
    cov = coin_cov[coin] * np.where(E, 3.0, 1.0)
    n_head = 1 + rng.poisson(cov)

    # ---- hidden states per vintage ------------------------------------
    W = rng.standard_normal((d, D)) / np.sqrt(d)
    u = rng.standard_normal(D); u /= np.linalg.norm(u)
    H = []
    for v in VINTAGES:
        k = v - year
        cap = 1.0 + 0.1 * (v - VINTAGES[0])               # capability gain
        rec = np.where((k >= -1) & (k <= 0), 0.3, 0.0)     # recency boost
        if recency_mode == "coverage":                     # confound tied to coverage
            rec = rec * np.log1p(cov) / np.log1p(cov).mean()
        sig = (cap * (1 + rec))[:, None] * (x @ W)
        noise = rng.standard_normal((n, D)) * (1.5 / np.sqrt(n_head))[:, None]
        leak_g = np.select([k == 0, k == 1, k >= 2], [1.0, 0.8, 0.6], 0.0)
        leak = (gamma * leak_g * np.sqrt(cov) * y)[:, None] * u[None, :]
        H.append(sig + noise + leak)
    return dict(coin=coin, week=week, year=year, y=y, E=E, cov=cov, H=H)


def pvi_per_vintage(P, n_folds=5, alpha=10.0):
    coin, y = P["coin"], P["y"]
    uniq = np.unique(coin)
    fold_of_coin = np.random.default_rng(0).permutation(len(uniq)) % n_folds
    fold = fold_of_coin[np.searchsorted(uniq, coin)]
    out = np.zeros((len(y), len(P["H"])))
    for j, H in enumerate(P["H"]):
        for f in range(n_folds):
            tr, te = fold != f, fold == f
            Xtr = H[tr]; mu = Xtr.mean(0); ym = y[tr].mean()
            Xc = Xtr - mu
            w = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(H.shape[1]), Xc.T @ (y[tr] - ym))
            res_tr = y[tr] - ym - Xc @ w
            s2 = res_tr.var()
            yhat = ym + (H[te] - mu) @ w
            # Gaussian PVI: log-lik gain over the null probe (same variance)
            out[te, j] = ((y[te] - ym) ** 2 - (y[te] - yhat) ** 2) / (2 * s2)
    return out


def event_study(P, PVI, vintage_by_event_fe=False):
    n, V = PVI.shape
    k = VINTAGES[None, :] - P["year"][:, None]
    kb = np.clip(k, -3, 2)
    X = np.stack([((kb == kk) & P["E"][:, None]).astype(float) for kk in K_BINS], -1)
    Y = PVI.copy()

    def demean(A):  # item FE then vintage-by-week FE (balanced, week nested in item)
        A = A - A.mean(axis=1, keepdims=True)
        out = A.copy()
        for w in np.unique(P["week"]):
            s = P["week"] == w
            out[s] -= A[s].mean(axis=0, keepdims=True)
        return out

    Yd = demean(Y)
    Xd = np.stack([demean(X[..., j]) for j in range(X.shape[-1])], -1)
    if vintage_by_event_fe:
        # partial out vintage x event-status dummies (after the nested FE demeaning)
        cols = [P["E"].astype(float)]
        if "yhat_clean" in P:  # vintage x observable predictability and coverage controls
            cols += [np.abs(P["yhat_clean"]), P["yhat_clean"] ** 2, np.log(P["cov"])]
        Ed = np.stack([demean(np.outer(c, np.eye(V)[j])) for c in cols for j in range(V)], -1)
        Z = Ed.reshape(-1, Ed.shape[-1])
        proj = lambda a: a - Z @ np.linalg.lstsq(Z, a, rcond=None)[0]
        Yd = proj(Yd.reshape(-1))
        Xd = np.stack([proj(Xd[..., j].reshape(-1)) for j in range(Xd.shape[-1])], -1)
    else:
        Yd = Yd.reshape(-1); Xd = Xd.reshape(-1, Xd.shape[-1])
    XtX_inv = np.linalg.pinv(Xd.T @ Xd)
    beta = XtX_inv @ Xd.T @ Yd
    e = Yd - Xd @ beta
    cl = np.repeat(P["coin"], V)
    meat = np.zeros((Xd.shape[1],) * 2)
    for c in np.unique(cl):
        s = cl == c
        g = Xd[s].T @ e[s]
        meat += np.outer(g, g)
    G = len(np.unique(cl))
    adj = G / (G - 1)
    Vc = adj * XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.diag(Vc))
    i0 = K_BINS.index(0)
    t0 = beta[i0] / se[i0]
    pre = [K_BINS.index(-3), K_BINS.index(-2)]
    bp = beta[pre]; Vp = Vc[np.ix_(pre, pre)]
    wald = bp @ np.linalg.pinv(Vp) @ bp
    return dict(beta=beta, se=se, t0=t0,
                reject_h1=t0 > stats.norm.ppf(0.95),
                reject_pre=stats.chi2.sf(wald, 2) < 0.05)


def leak_r2(P, PVI):
    """Mean PVI gap on event items: first exposed vintage vs the one before (k=0 vs k=-1)."""
    k = VINTAGES[None, :] - P["year"][:, None]
    E = P["E"]
    a = PVI[E][(k[E] == 0)].mean(); b = PVI[E][(k[E] == -1)].mean()
    return a - b


def surprise_events(P, event_q=0.05, ref=0):
    """Redefine events on |y - yhat_clean| from the oldest (clean) vintage, cross-fitted."""
    coin, y, week = P["coin"], P["y"], P["week"]
    uniq = np.unique(coin); fold = (np.random.default_rng(1).permutation(len(uniq)) % 5)[np.searchsorted(uniq, coin)]
    H = P["H"][ref]; yhat = np.zeros_like(y)
    for f in range(5):
        tr, te = fold != f, fold == f
        mu = H[tr].mean(0); Xc = H[tr] - mu; ym = y[tr].mean()
        w = np.linalg.solve(Xc.T @ Xc + 10 * np.eye(H.shape[1]), Xc.T @ (y[tr] - ym))
        yhat[te] = ym + (H[te] - mu) @ w
    s = y - yhat
    E = np.zeros(len(y), bool)
    for w_ in np.unique(week):
        idx = np.nonzero(week == w_)[0]
        thr = np.quantile(np.abs(s[idx]), 1 - 2 * event_q)
        E[idx[np.abs(s[idx]) >= thr]] = True
    P["yhat_clean"] = yhat
    return E


def one_run(seed, n_coins, gamma, recency_mode, vxe=False, surprise=False):
    rng = np.random.default_rng(seed)
    P = simulate_panel(rng, n_coins, gamma, recency_mode)
    if surprise:
        P["E"] = surprise_events(P)
    PVI = pvi_per_vintage(P)
    r = event_study(P, PVI, vintage_by_event_fe=vxe)
    return dict(n_items=len(P["y"]), reject_h1=r["reject_h1"], reject_pre=r["reject_pre"],
                beta0=r["beta"][K_BINS.index(0)], gap=leak_r2(P, PVI))
