"""Second and third headline sources from Hugging Face, normalised to the CryptoVision schema.

- maryamfakhari/crypto-news-coindesk-2020-2025: the CryptoCompare news feed (229k items, CC BY-NC 4.0).
  Its `categories` field lists coin tickers; it is used only when the title names no coin and lists one.
(SahandNZ/cryptonews-articles-with-price-momentum-labels was checked and rejected: it stores
preprocessed sentence fragments, not headlines.)
Only titles enter the model prompt; article bodies are never used.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import RAW
import re

from .coins import tag
from .http import get_bytes, get_json

SOURCES = {
    "coindesk_cc": "maryamfakhari/crypto-news-coindesk-2020-2025",
}


# CryptoCompare category labels that are topics, not assets; tickers renamed after the Terra collapse
NOT_A_COIN = {"BUSINESS", "MARKET", "TRADING", "TECHNOLOGY", "EXCHANGE", "REGULATION", "FIAT", "ALTCOIN", "BLOCKCHAIN",
              "MINING", "WALLET", "ICO", "SPONSORED", "COMMODITY", "ASIA", "OTHER", "STABLECOIN", "DEFI", "NFT", "METAVERSE",
              "AI", "CRYPTOCURRENCY", "RESEARCH", "SECURITY", "LEGAL", "GAMBLING", "PRICE", "GAMING", "FORKS", "AIRDROP",
              "ENTERTAINMENT", "SPORTS", "WEB3", "DAO", "MACROECONOMICS", "CBDC", "RWA", "STAKING", "LAYER2", "ZK", "ETF",
              "USDT", "USDC", "DAI", "BUSD", "TUSD"}
RENAMED = {"LUNC": "LUNA", "USTC": "UST"}


def download(name: str, dest: Path = RAW) -> list[Path]:
    repo = SOURCES[name]
    d = dest / name
    d.mkdir(parents=True, exist_ok=True)
    urls = get_json(f"https://huggingface.co/api/datasets/{repo}/parquet")
    flat = [u for cfg in urls.values() for split in cfg.values() for u in split] if isinstance(urls, dict) else urls
    paths = []
    for i, u in enumerate(flat):
        p = d / f"part{i:03d}.parquet"
        if not p.exists():
            p.write_bytes(get_bytes(u, timeout=300))
        paths.append(p)
    return paths


def _explode(df: pd.DataFrame) -> pd.DataFrame:
    df = df.explode("coins").dropna(subset=["coins"]).rename(columns={"coins": "coin"})
    return df[["ts", "coin", "title", "source", "url"]]


def load_coindesk(dest: Path = RAW) -> pd.DataFrame:
    df = pd.concat(pd.read_parquet(p) for p in sorted((dest / "coindesk_cc").glob("*.parquet")))
    ts = pd.to_numeric(df["published_on"], errors="coerce")
    df["ts"] = pd.to_datetime(ts, unit="s", utc=True) if ts.notna().mean() > 0.9 else pd.to_datetime(df["published_on"], utc=True, errors="coerce")
    def coins(row):
        cats = {RENAMED.get(c.strip().upper(), c.strip().upper()) for c in str(row["categories"]).split("|")}
        cats = {c for c in cats if re.fullmatch(r"[A-Z0-9]{2,10}", c) and c not in NOT_A_COIN}
        named = set(tag(row["title"]))
        # the feed's categories are broad (generic market news carries many tickers): a coin counts only if
        # the title names it; categories are used alone only when the title names no coin and lists one
        return sorted(named if named else (cats if len(cats) == 1 else set()))
    df["coins"] = df.apply(coins, axis=1)
    df["source"] = df["source"].astype(str)
    df["title"] = df["title"].astype(str).str.strip()
    return _explode(df)
