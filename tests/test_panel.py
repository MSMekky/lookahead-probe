import numpy as np
import pandas as pd

from lookahead import panel
from lookahead.config import PanelConfig


def _fake(n_coins=15, seed=0):
    rng = np.random.default_rng(seed)
    days = pd.date_range("2020-11-01", "2024-01-31", freq="D")
    coins = ["BTC"] + [f"C{i:02d}" for i in range(n_coins)]
    px = pd.DataFrame(np.exp(np.cumsum(rng.normal(0, 0.03, (len(days), len(coins))), 0)), index=days, columns=coins)
    rows = []
    for c in coins[1:]:
        for t in pd.date_range("2020-12-01", "2023-11-30", freq="2D"):
            rows.append(dict(ts=pd.Timestamp(t, tz="UTC") + pd.Timedelta(hours=9), coin=c,
                             title=f"{c} update {t:%Y%m%d}", source="x", url=""))
    return pd.DataFrame(rows), px


def test_targets_are_forward_excess_returns():
    news, px = _fake()
    items = panel.build(news, px, PanelConfig(min_coins_per_week=5))
    row = items.iloc[len(items) // 2]
    p0 = row["week"] + pd.Timedelta(days=6); p1 = p0 + pd.Timedelta(days=30)
    expect = np.log(px.loc[p1, row["coin"]] / px.loc[p0, row["coin"]]) - np.log(px.loc[p1, "BTC"] / px.loc[p0, "BTC"])
    assert np.isclose(row["y_raw"], expect)
    assert row["year"] == p1.year


def test_rank_normalised_within_week_and_years_bounded():
    news, px = _fake()
    items = panel.build(news, px, PanelConfig(min_coins_per_week=5))
    assert set(items["year"]) <= {2021, 2022, 2023}
    assert items.groupby("week")["y"].mean().abs().max() < 1e-9
    assert "week of" in items["text"].iloc[0]


def test_no_headline_after_week_end_enters_text():
    news, px = _fake()
    items = panel.build(news, px, PanelConfig(min_coins_per_week=5))
    for _, r in items.sample(20, random_state=0).iterrows():
        dates = [pd.Timestamp(t.split()[-1]) for t in r["text"].splitlines()[1:]]
        assert all(r["week"] <= d < r["week"] + pd.Timedelta(days=7) for d in dates)
