import pandas as pd

from lookahead.data import prices


def test_plan_splices_at_the_declared_dates(monkeypatch):
    idx = pd.date_range("2022-05-01", "2022-06-30", freq="D", tz="UTC")
    a = pd.Series(1.0, idx); b = pd.Series(2.0, idx)
    monkeypatch.setattr(prices, "_gate", lambda pair, s, e: a if pair == "LUNA_USDT" else b)
    out = prices._plan("LUNA", "2022-05-01", "2022-06-30")
    assert out.loc["2022-05-26"].item() == 1.0 and out.loc["2022-05-27"].item() == 2.0
    assert out.index.is_unique and len(out) == len(idx)
