"""Coin-week panel: one item per coin and ISO week with at least one headline and usable prices.

Target: log return over the 30 days after the week closes, in excess of BTC, rank-normalised
within the week (market-neutral). Outcome year = calendar year in which the 30-day window ends.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy import stats

from .config import CATALOGUE, CFG, CUE_WORDS, PanelConfig

_CUE = re.compile(r"\b(" + "|".join(re.escape(w) for w in CUE_WORDS) + r")", re.I)


def prompt(coin: str, monday: pd.Timestamp, titles: list[str]) -> str:
    head = f"Crypto news headlines about {coin} published in the week of {monday:%B %d, %Y}:\n"
    return head + "\n".join(f"- {t}" for t in titles)


def rank_gauss(x: pd.Series) -> pd.Series:
    r = x.rank(method="average")
    return pd.Series(stats.norm.ppf((r - 0.5) / len(x)), index=x.index)


def build(news: pd.DataFrame, prices: pd.DataFrame, pc: PanelConfig = CFG.panel) -> pd.DataFrame:
    news = news.copy()
    ts = news["ts"].dt.tz_convert("UTC").dt.tz_localize(None)
    news["monday"] = (ts.dt.normalize() - pd.to_timedelta(ts.dt.weekday, unit="D"))
    news = news[(news["monday"] >= pc.first_week) & (news["monday"] <= pc.last_week)]
    news = news[news["coin"].isin(prices.columns)]

    px = prices.copy()
    px.index = px.index.tz_localize(None) if px.index.tz is not None else px.index
    px = px.asfreq("D")

    rows = []
    for (coin, monday), g in news.sort_values("ts").groupby(["coin", "monday"], sort=False):
        titles = list(dict.fromkeys(g["title"].tolist()))
        rows.append(dict(coin=coin, week=monday, n_head=len(titles),
                         text=prompt(coin, monday, titles[: pc.max_headlines]),
                         cue=bool(_CUE.search(" ".join(titles[: pc.max_headlines]))),
                         n_sources=g["source"].nunique()))
    items = pd.DataFrame(rows)

    p0_day = items["week"] + pd.Timedelta(days=6)                       # Sunday close
    p1_day = p0_day + pd.Timedelta(days=pc.horizon_days)
    days = pd.Index(sorted(set(p0_day) | set(p1_day)))
    P = px.reindex(days)
    lp0 = np.log(P.to_numpy(dtype=float))
    col = {c: j for j, c in enumerate(P.columns)}
    row = {d: i for i, d in enumerate(P.index)}
    bj = col[pc.benchmark]
    i0 = p0_day.map(row).to_numpy(); i1 = p1_day.map(row).to_numpy(); j = items["coin"].map(col).to_numpy()
    r = lp0[i1, j] - lp0[i0, j]
    rb = lp0[i1, bj] - lp0[i0, bj]
    items["ret"] = r
    items["ret_btc"] = rb
    items["y_raw"] = r - rb
    items["outcome_end"] = p1_day
    items["year"] = items["outcome_end"].dt.year
    items = items[np.isfinite(items["y_raw"]) & (items["coin"] != pc.benchmark)]
    items = items[items["year"].isin(pc.outcome_years)]

    # winsorise extreme log returns (data errors on illiquid pairs), then rank-normalise per week
    lo, hi = items["y_raw"].quantile([0.001, 0.999])
    items["y_raw"] = items["y_raw"].clip(lo, hi)
    n_w = items.groupby("week")["coin"].transform("size")
    items = items[n_w >= pc.min_coins_per_week].copy()
    items["y"] = items.groupby("week", group_keys=False)["y_raw"].apply(rank_gauss)

    cat = pd.DataFrame(CATALOGUE, columns=["coin", "date"])
    d = pd.to_datetime(cat["date"])
    cat["week"] = d.dt.normalize() - pd.to_timedelta(d.dt.weekday, unit="D")
    key = set(zip(cat["coin"], cat["week"]))
    items["catalogue"] = [(c, w) in key for c, w in zip(items["coin"], items["week"])]

    items = items.sort_values(["week", "coin"]).reset_index(drop=True)
    items["week_idx"] = items["week"].rank(method="dense").astype(int) - 1
    items.insert(0, "item_id", np.arange(len(items)))
    return items


def summary(items: pd.DataFrame) -> dict:
    return dict(items=len(items), coins=items["coin"].nunique(), weeks=items["week"].nunique(),
                coins_per_week_median=float(items.groupby("week").size().median()),
                items_by_year=items.groupby("year").size().to_dict(),
                headlines_per_item_median=float(items["n_head"].median()),
                cue_share=float(items["cue"].mean()), catalogue_items=int(items["catalogue"].sum()))
