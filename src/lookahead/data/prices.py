"""Daily close prices in USDT/USD. Binance spot first, then KuCoin, then Gate (none needs an API key).

Renamed or delisted tickers are mapped explicitly; a coin with no price source is dropped from the
panel and listed in the build log, never imputed.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from ..config import RAW
from .http import get_json

# Explicit plans for the 2022 failure coins, chosen after checking every source around May 2022
# (docs/data_sources.md). Binance's LUNAUSDT and Gate's LUNA_USDT both switch to Terra 2.0 in late
# May 2022; Gate's LUNC_USDT continues the original coin from 27 May. Bitfinex lists TerraUSD as
# TERRAUST (its "UST" is Tether). Celsius (CEL) trades only on Gate and Bitfinex among these venues.
PLANS = {
    "LUNA": [("gate", "LUNA_USDT", None, "2022-05-26"), ("gate", "LUNC_USDT", "2022-05-27", None)],
    "UST": [("bitfinex", "tTERRAUST:USD", None, None)],
    "CEL": [("gate", "CEL_USDT", None, None)],
}
BINANCE_SYMBOL = {}
STABLE = {"USDT", "USDC", "BUSD", "DAI", "TUSD", "USDP", "FDUSD"}


def _binance(sym: str, start: str, end: str) -> pd.Series:
    pair = BINANCE_SYMBOL.get(sym, f"{sym}USDT")
    t0 = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    t1 = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)
    rows = []
    while t0 < t1:
        try:
            k = get_json(f"https://api.binance.com/api/v3/klines?symbol={pair}&interval=1d&startTime={t0}&limit=1000")
        except Exception:
            return pd.Series(dtype=float)
        if not k:
            break
        rows += k
        t0 = k[-1][0] + 86_400_000
        time.sleep(0.15)
    if not rows:
        return pd.Series(dtype=float)
    s = pd.Series([float(r[4]) for r in rows], index=pd.to_datetime([r[0] for r in rows], unit="ms", utc=True))
    return s[~s.index.duplicated()].loc[:end]


def _kucoin(sym: str, start: str, end: str) -> pd.Series:
    pair = f"{sym}-USDT"
    t0 = int(pd.Timestamp(start, tz="UTC").timestamp())
    t1 = int(pd.Timestamp(end, tz="UTC").timestamp())
    rows = []
    while t0 < t1:
        hi = min(t1, t0 + 1400 * 86400)
        try:
            d = get_json(f"https://api.kucoin.com/api/v1/market/candles?type=1day&symbol={pair}&startAt={t0}&endAt={hi}")
        except Exception:
            break
        rows += d.get("data") or []      # [time, open, close, high, low, volume, turnover], newest first
        t0 = hi
        time.sleep(0.2)
    if not rows:
        return pd.Series(dtype=float)
    s = pd.Series({pd.Timestamp(int(r[0]), unit="s", tz="UTC"): float(r[2]) for r in rows}).sort_index()
    return s[s > 0]


def _gate(pair: str, start: str, end: str) -> pd.Series:
    t0 = int(pd.Timestamp(start, tz="UTC").timestamp()); t1 = int(pd.Timestamp(end, tz="UTC").timestamp())
    rows = []
    for a in range(t0, t1, 86400 * 900):
        try:
            rows += get_json(f"https://api.gateio.ws/api/v4/spot/candlesticks?currency_pair={pair}&interval=1d"
                             f"&from={a}&to={min(t1, a + 86400 * 899)}")
        except Exception:
            break
        time.sleep(0.2)
    if not rows:
        return pd.Series(dtype=float)
    s = pd.Series({pd.Timestamp(int(r[0]), unit="s", tz="UTC"): float(r[2]) for r in rows}).sort_index()
    return s[s > 0]


def _bitfinex(sym: str, start: str, end: str) -> pd.Series:
    a = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000); b = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)
    try:
        rows = get_json(f"https://api-pub.bitfinex.com/v2/candles/trade:1D:{sym}/hist?start={a}&end={b}&limit=10000&sort=1")
    except Exception:
        return pd.Series(dtype=float)
    s = pd.Series({pd.Timestamp(int(r[0]), unit="ms", tz="UTC"): float(r[2]) for r in rows}).sort_index()
    return s[s > 0]


def _plan(sym: str, start: str, end: str) -> pd.Series:
    parts = []
    for venue, pair, lo, hi in PLANS[sym]:
        f = {"gate": _gate, "bitfinex": _bitfinex}[venue]
        s = f(pair, start, end)
        if len(s):
            s = s.loc[pd.Timestamp(lo, tz="UTC") if lo else s.index.min(): pd.Timestamp(hi, tz="UTC") if hi else s.index.max()]
        parts.append(s)
    out = pd.concat(parts).sort_index()
    return out[~out.index.duplicated(keep="last")]


def _max_gap_days(s: pd.Series) -> int:
    if len(s) < 2:
        return 10 ** 6
    return int(s.dropna().index.to_series().diff().dt.days.max())


def fetch(coins: list[str], start: str, end: str, cache: Path = RAW / "prices.parquet") -> pd.DataFrame:
    """Wide frame: index = UTC day, columns = coin, values = close. Also returns the source per coin."""
    have = pd.read_parquet(cache) if cache.exists() else pd.DataFrame()
    src_file = cache.with_suffix(".sources.csv")
    old_src = pd.read_csv(src_file, index_col=0).iloc[:, 0].to_dict() if src_file.exists() else {}
    cols, src = {}, {}
    for c in sorted(set(coins) - STABLE):
        if c in have.columns and c not in PLANS and have[c].notna().sum() > 30:
            cols[c] = have[c].dropna(); src[c] = old_src.get(c, "cache"); continue
        need = (pd.Timestamp(end) - pd.Timestamp(start)).days
        if c in PLANS:
            s, src[c] = _plan(c, start, end), "plan:" + "+".join(f"{v}:{p}" for v, p, _, _ in PLANS[c])
        else:
            s, src[c] = _binance(c, start, end), "binance"
            # listed late, delisted, renamed, or a gap that may hide a ticker reuse: try KuCoin, then Gate
            for name, f in (("kucoin", _kucoin), ("gate", lambda x, a, b: _gate(f"{x}_USDT", a, b))):
                if s.notna().sum() >= 0.9 * need and _max_gap_days(s) <= 7:
                    break
                alt = f(c, start, end)
                if alt.notna().sum() > s.notna().sum():
                    s, src[c] = alt, name
        if s.notna().sum() >= 60:
            cols[c] = s
        else:
            src[c] = "none"
        print(f"  price {c:<8} {src[c]:<14} {s.notna().sum():>5} days, max gap {_max_gap_days(s)}", flush=True)
    wide = pd.DataFrame(cols).sort_index()
    wide.index = wide.index.normalize()
    wide = wide[~wide.index.duplicated(keep="last")]
    cache.parent.mkdir(parents=True, exist_ok=True)
    wide.to_parquet(cache)
    pd.Series(src, name="source").to_csv(cache.with_suffix(".sources.csv"))
    return wide
