"""End-to-end analysis on extracted features: probes, PVI, event study (H1), robustness, remedy (H5)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import eventstudy as es
from . import remedy
from .config import CFG, FEATURES, RESULTS
from .probes import crossfit, selectivity


def load_features(family: str, feat_dir: Path = FEATURES) -> tuple[np.ndarray, list[np.ndarray]]:
    files = sorted(feat_dir.glob(f"{family}_*.npy"))
    if not files:
        raise FileNotFoundError(f"no features for {family} in {feat_dir}")
    vint = np.array([int(f.stem.split("_")[-1]) for f in files])
    return vint, [np.load(f, mmap_mode="r") for f in files]


def _items_dict(items: pd.DataFrame) -> dict:
    return dict(y=items["y"].to_numpy(float), coin=items["coin"].astype("category").cat.codes.to_numpy(),
                week=items["week_idx"].to_numpy(), year=items["year"].to_numpy(), cov=items["n_head"].to_numpy(float))


def _subset(d: dict, m: np.ndarray) -> dict:
    return {k: v[m] for k, v in d.items()}


def run(items: pd.DataFrame, family: str = CFG.model.family, feat_dir: Path = FEATURES,
        out_dir: Path = RESULTS, do_remedy: bool = True) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    mc = CFG.model
    vint, F = load_features(family, feat_dir)
    li = list(mc.layers).index(mc.main_layer)
    d = _items_dict(items)
    ref = int(np.where(vint == mc.reference_vintage)[0][0]) if mc.reference_vintage in vint else 0
    print(f"vintages {vint.tolist()}, reference {vint[ref]}, items {len(items)}", flush=True)

    def H(j, layer=li):
        return np.asarray(F[j][:, layer, :], dtype=np.float32)

    # surprise-defined events from the clean reference + the failure catalogue
    yhat_clean, E_s = es.surprise(H(ref), d["y"], d["coin"], d["week"], CFG.panel.event_tail)
    E = E_s | items["catalogue"].to_numpy()
    res = dict(family=family, vintages=vint.tolist(), reference_vintage=int(vint[ref]), main_layer=mc.main_layer,
               n_items=len(items), n_events=int(E.sum()), n_coins=int(items["coin"].nunique()),
               n_weeks=int(items["week"].nunique()))

    # PVI per vintage, main layer
    print("probing main layer", flush=True)
    fits = [crossfit(H(j), d["y"], d["coin"]) for j in range(len(vint))]
    P = np.stack([f["pvi"] for f in fits], 1)
    res["mean_pvi_by_vintage"] = {int(v): float(P[:, j].mean()) for j, v in enumerate(vint)}
    res["probe_corr_by_vintage"] = {int(v): float(np.corrcoef(fits[j]["yhat"], d["y"])[0, 1]) for j, v in enumerate(vint)}
    np.save(out_dir / "pvi_main.npy", P)

    args = (vint, d["year"], E, d["coin"], d["week"])
    res["h1"] = es.estimate(P, *args, yhat_clean=yhat_clean, cov=d["cov"])
    res["h1_twoway"] = es.estimate(P, *args, yhat_clean=yhat_clean, cov=d["cov"], cluster="twoway")
    res["leak_gap_nats"] = es.leak_gap(P, vint, d["year"], E)

    # v2 specification for contrast: raw-return events, no vintage-by-event terms
    E_raw = np.zeros(len(items), bool)
    for w in np.unique(d["week"]):
        idx = np.nonzero(d["week"] == w)[0]
        lo, hi = np.quantile(d["y"][idx], [CFG.panel.event_tail, 1 - CFG.panel.event_tail])
        E_raw[idx[(d["y"][idx] <= lo) | (d["y"][idx] >= hi)]] = True
    res["naive_v2_spec"] = es.estimate(P, vint, d["year"], E_raw, d["coin"], d["week"], vintage_by_event=False)

    # cue-free subset
    m = ~items["cue"].to_numpy()
    res["h1_cue_free"] = es.estimate(P[m], vint, d["year"][m], E[m], d["coin"][m], d["week"][m],
                                     yhat_clean=yhat_clean[m], cov=d["cov"][m])

    # other layers
    res["h1_by_layer"] = {}
    for L in mc.layers:
        if L == mc.main_layer:
            continue
        l2 = list(mc.layers).index(L)
        PL = np.stack([crossfit(H(j, l2), d["y"], d["coin"])["pvi"] for j in range(len(vint))], 1)
        res["h1_by_layer"][int(L)] = es.estimate(PL, *args, yhat_clean=yhat_clean, cov=d["cov"])

    # selectivity (control task) on the reference and the latest vintage
    res["selectivity"] = {int(vint[j]): selectivity(H(j), d["y"], d["coin"], n_perm=3) for j in (ref, len(vint) - 1)}

    # descriptive: mean PVI on events and non-events by k
    k = vint[None, :] - d["year"][:, None]
    res["mean_pvi_by_k"] = {int(kk): dict(event=float(P[E][k[E] == kk].mean()) if (k[E] == kk).any() else None,
                                          nonevent=float(P[~E][k[~E] == kk].mean()) if (k[~E] == kk).any() else None)
                            for kk in np.unique(k)}

    if do_remedy:
        print("remedy", flush=True)
        r = remedy.evaluate(H, vint, d, yhat_clean, E, P)
        np.save(out_dir / "pvi_after_remedy.npy", r.pop("pvi_after"))
        res["remedy"] = r

    (out_dir / "results.json").write_text(json.dumps(res, indent=2, default=float))
    plot(res, out_dir)
    report(json.loads((out_dir / "results.json").read_text()), out_dir)
    return res


def report(res: dict, out_dir: Path) -> str:
    """Plain-text summary written next to results.json; every number comes from res."""
    h, tw, nv = res["h1"], res["h1_twoway"], res["naive_v2_spec"]
    f = lambda x: f"{x:+.4f}"
    rows = "\n".join(f"| {k} | {f(b)} | {s:.4f} | {t:+.2f} |" for k, b, s, t in zip(h["k"], h["beta"], h["se"], h["t"]))
    lay = "\n".join(f"| {L} | {f(r['beta0'])} | {r['se0']:.4f} | {r['p0_one_sided']:.3f} |"
                    for L, r in res["h1_by_layer"].items())
    sel = "\n".join(f"| {v} | {s['pvi']:.4f} | {s['control']:.4f} | {s['selectivity']:.4f} |"
                    for v, s in res["selectivity"].items())
    mv = " | ".join(f"{v}: {p:.4f}" for v, p in res["mean_pvi_by_vintage"].items())
    cf = res["h1_cue_free"]
    txt = f"""# Results: {res['family']}, layer {res['main_layer']}

Panel: {res['n_items']} coin-weeks, {res['n_coins']} coins, {res['n_weeks']} weeks, {res['n_events']} events
(surprise-defined against the {res['reference_vintage']} vintage, plus the failure catalogue). Vintages {res['vintages']}.

## H1: leakage at first exposure

beta0 = {f(h['beta0'])} nats (SE {h['se0']:.4f}, coin-clustered, {h['n_clusters']} clusters), one-sided p = {h['p0_one_sided']:.3f}.
Null rejected at 5%: **{h['reject_h1']}**. One-sided 95% upper bound on beta0: {f(h['beta0_upper95'])} nats. Two-way clustered (coin and week): beta0 {f(tw['beta0'])}, SE {tw['se0']:.4f}, p = {tw['p0_one_sided']:.3f}.
Descriptive leak gap (events, k = 0 minus k = -1): {f(res['leak_gap_nats'])} nats.
Pre-trend joint test (a diagnostic, not a gate; see docs/power_analysis.md): p = {h['pretrend_p']:.3f}.

| k | beta | SE | t |
| --- | --- | --- | --- |
{rows}

v2 specification for contrast (raw-return events, no vintage-by-event terms): beta0 {f(nv['beta0'])}, p = {nv['p0_one_sided']:.3f}.
Cue-free subset: beta0 {f(cf['beta0'])}, p = {cf['p0_one_sided']:.3f}, n = {cf['n_items']}.

## Other layers

| layer | beta0 | SE | p |
| --- | --- | --- | --- |
{lay}

## Probe quality

Mean PVI by vintage: {mv}

| vintage | PVI | control task | selectivity |
| --- | --- | --- | --- |
{sel}
"""
    rem = res.get("remedy")
    if rem:
        ea = rem["event_study_after"]
        ret = rem["retention_preexposure"]
        txt += f"""
## H5: residual erasure

Leak gap {f(rem['leak_gap_before'])} before, {f(rem['leak_gap_after'])} after. beta0 after erasure {f(ea['beta0'])} (p = {ea['p0_one_sided']:.3f}).
Mean PVI on pre-exposure cells: {f(rem['mean_pvi_preexposure_before'])} before, {f(rem['mean_pvi_preexposure_after'])} after.
Retention ratio: {'not meaningful (baseline below 0.005 nats)' if ret is None else f'{ret:.0%}'}.
"""
    (out_dir / "RESULTS.md").write_text(txt)
    return txt


def plot(res: dict, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for key, lab, off in [("h1", "adopted spec", -0.08), ("naive_v2_spec", "v2 spec", 0.08)]:
        h = res[key]
        kk = np.array(h["k"], float)
        ax[0].errorbar(kk + off, h["beta"], yerr=1.96 * np.array(h["se"]), fmt="o", capsize=3, label=lab)
    ax[0].axhline(0, color="grey", lw=0.8); ax[0].axvline(-1, color="grey", lw=0.8, ls=":")
    ax[0].set_xlabel("k = vintage - outcome year (k = -1 reference)"); ax[0].set_ylabel("PVI gap, events vs non-events (nats)")
    ax[0].legend(frameon=False); ax[0].set_title("Event study")
    ks = sorted(int(k) for k in res["mean_pvi_by_k"])
    for lab in ("event", "nonevent"):
        mk = {int(k): v for k, v in res["mean_pvi_by_k"].items()}
        ax[1].plot(ks, [np.nan if mk[k][lab] is None else mk[k][lab] for k in ks], marker="o", label=lab)
    ax[1].axvline(-0.5, color="grey", lw=0.8, ls=":"); ax[1].set_xlabel("k"); ax[1].set_ylabel("mean PVI (nats)")
    ax[1].legend(frameon=False); ax[1].set_title("Mean PVI by distance to cutoff")
    fig.tight_layout(); fig.savefig(out_dir / "event_study.png", dpi=160); plt.close(fig)
