"""CryptoVision (Mendeley Data 3c3xtxtfb6, v2): headline-level crypto news with coin and timestamp.

The dataset ships titles and URLs only, no article bodies. This module downloads it, finds the
headline table, and normalises it to: ts (UTC), coin (upper-case symbol), title, source, url.
"""
from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

import pandas as pd

from ..config import RAW
from .coins import tag
from .http import get_bytes, get_json

DATASET_ID = "3c3xtxtfb6"
VERSION = 2
API = f"https://data.mendeley.com/public-api/datasets/{DATASET_ID}"

# Coin_Type labels seen in crypto news datasets mapped to tickers. Unknown labels are kept
# upper-cased; labels that are not a single asset are dropped.
ALIASES = {
    "BITCOIN": "BTC", "ETHEREUM": "ETH", "RIPPLE": "XRP", "CARDANO": "ADA", "SOLANA": "SOL",
    "DOGECOIN": "DOGE", "POLKADOT": "DOT", "LITECOIN": "LTC", "CHAINLINK": "LINK",
    "TERRA": "LUNA", "TERRALUNA": "LUNA", "TERRAUSD": "UST", "BINANCECOIN": "BNB",
    "BINANCE COIN": "BNB", "POLYGON": "MATIC", "AVALANCHE": "AVAX", "TRON": "TRX",
    "STELLAR": "XLM", "TETHER": "USDT", "USDCOIN": "USDC", "SHIBAINU": "SHIB",
    "SHIBA INU": "SHIB", "CELSIUS": "CEL", "FTX TOKEN": "FTT", "FTXTOKEN": "FTT",
    "MONERO": "XMR", "VECHAIN": "VET", "COSMOS": "ATOM", "UNISWAP": "UNI", "ALGORAND": "ALGO", "TEZOS": "XTZ",
    "EOS": "EOS", "NEAR PROTOCOL": "NEAR", "APTOS": "APT", "ARBITRUM": "ARB", "TONCOIN": "TON",
    "BITCOIN CASH": "BCH", "ETHEREUM CLASSIC": "ETC", "FILECOIN": "FIL", "HEDERA": "HBAR",
    "INTERNET COMPUTER": "ICP", "DECENTRALAND": "MANA", "THE SANDBOX": "SAND", "AXIE INFINITY": "AXS",
}
NOT_AN_ASSET = {"", "NAN", "NONE", "GENERAL", "CRYPTO", "CRYPTOCURRENCY", "MARKET", "OTHER", "NFT",
                "DEFI", "BLOCKCHAIN", "ALTCOIN", "ALTCOINS", "STABLECOIN", "STABLECOINS", "WEB3"}


def list_files() -> list[dict]:
    out = get_json(f"{API}/files?folder_id=root&version={VERSION}")
    return out if isinstance(out, list) else out.get("files", out)


def download(dest: Path = RAW / "cryptovision") -> list[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    meta = get_json(API)
    (dest / "metadata.json").write_text(json.dumps(meta, indent=2))
    paths = []
    for f in list_files():
        name = f.get("filename") or f.get("name")
        url = (f.get("content_details") or {}).get("download_url") or f.get("download_url")
        if not url:
            url = f"https://data.mendeley.com/public-files/datasets/{DATASET_ID}/files/{f['id']}/file_downloaded"
        p = dest / name
        if not p.exists():
            p.write_bytes(get_bytes(url))
        paths.append(p)
        if p.suffix.lower() == ".zip":
            with zipfile.ZipFile(p) as z:
                z.extractall(dest)
    return paths


def _find_table(dest: Path) -> Path:
    cands = [p for p in dest.rglob("*") if p.suffix.lower() in {".csv", ".xlsx", ".parquet"}]
    if not cands:
        raise FileNotFoundError(f"no table under {dest}")
    return max(cands, key=lambda p: p.stat().st_size)


def _col(df: pd.DataFrame, *names: str) -> str:
    low = {re.sub(r"[^a-z]", "", c.lower()): c for c in df.columns}
    for n in names:
        if n in low:
            return low[n]
    raise KeyError(f"none of {names} in {list(df.columns)}")


def normalise_coin(label: str) -> str | None:
    s = re.sub(r"\s+", " ", str(label).strip().upper())
    s = s.replace("$", "")
    if s in NOT_AN_ASSET:
        return None
    s = ALIASES.get(s, ALIASES.get(s.replace(" ", ""), s))
    if not re.fullmatch(r"[A-Z0-9]{2,10}", s):
        return None
    return s


def load(dest: Path = RAW / "cryptovision") -> pd.DataFrame:
    path = _find_table(dest)
    df = pd.read_parquet(path) if path.suffix == ".parquet" else (
        pd.read_excel(path) if path.suffix == ".xlsx" else pd.read_csv(path, low_memory=False))
    out = pd.DataFrame({
        "ts": pd.to_datetime(df[_col(df, "datetime", "date", "publishedat", "timestamp")], utc=True, errors="coerce"),
        "coin_label": df[_col(df, "cointype", "coin", "symbol", "currency")].astype(str),
        "title": df[_col(df, "title", "headline")].astype(str).str.strip(),
    })
    for extra, names in {"url": ("url", "link"), "source": ("source", "outlet", "domain")}.items():
        try:
            out[extra] = df[_col(df, *names)].astype(str)
        except KeyError:
            out[extra] = ""
    if (out["source"] == "").all():
        out["source"] = out["url"].str.extract(r"https?://(?:www\.)?([^/]+)")[0].fillna("")
    # coin = the dataset's own label, plus every coin the headline names (the label covers 16 coins only)
    lab = out["coin_label"].map(normalise_coin)
    out["coin"] = [sorted(({l} if isinstance(l, str) else set()) | set(tag(t))) for l, t in zip(lab, out["title"])]
    out = out.explode("coin")
    n0 = len(out)
    out = out.dropna(subset=["ts", "coin"])
    out = out[out["title"].str.len() > 5]
    out = out.drop_duplicates(subset=["coin", "title"])
    out.attrs["dropped"] = n0 - len(out)
    return out.sort_values("ts").reset_index(drop=True)
