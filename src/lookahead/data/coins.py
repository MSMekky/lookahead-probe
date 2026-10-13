"""Coin dictionary for tagging headlines by mention.

A headline is assigned to a coin when it names the coin (case-sensitive name match on word
boundaries) or writes its ticker. Tickers that are also common English words are matched only in
the $TICKER form. Stablecoins are excluded downstream (no market-neutral return to speak of).
"""
from __future__ import annotations

import re

NAMES = {
    "BTC": ["Bitcoin"], "ETH": ["Ethereum", "Ether"], "SOL": ["Solana"], "ADA": ["Cardano"], "XRP": ["Ripple", "XRP"],
    "DOGE": ["Dogecoin"], "DOT": ["Polkadot"], "BNB": ["Binance Coin", "BNB"], "MATIC": ["Polygon", "MATIC"],
    "LUNA": ["Terra", "LUNA", "Luna"], "UST": ["TerraUSD", "UST"], "AVAX": ["Avalanche", "AVAX"], "SHIB": ["Shiba Inu", "SHIB"],
    "UNI": ["Uniswap"], "LTC": ["Litecoin"], "LINK": ["Chainlink"], "BCH": ["Bitcoin Cash"], "CEL": ["Celsius"],
    "NEAR": ["NEAR Protocol", "Near Protocol"], "AAVE": ["Aave"], "ARB": ["Arbitrum"], "OP": ["Optimism network"],
    "ATOM": ["Cosmos", "ATOM"], "LDO": ["Lido"], "AXS": ["Axie Infinity", "Axie"], "CRV": ["Curve Finance", "Curve DAO"], "MKR": ["MakerDAO"],
    "FTM": ["Fantom"], "PEPE": ["PEPE", "Pepe"], "TRX": ["Tron", "TRON"], "APT": ["Aptos"], "ETC": ["Ethereum Classic"],
    "XLM": ["Stellar Lumens", "Stellar Development Foundation"], "ICP": ["Internet Computer"], "XTZ": ["Tezos"], "ALGO": ["Algorand"], "FIL": ["Filecoin"],
    "THETA": ["Theta Network"], "APE": ["ApeCoin"], "MANA": ["Decentraland"], "1INCH": ["1inch"], "STX": ["Stacks blockchain"],
    "FTT": ["FTT", "FTX Token"], "EOS": ["EOS"], "COMP": ["Compound Finance"], "XMR": ["Monero"], "SUSHI": ["SushiSwap", "Sushi"],
    "CELO": ["Celo"], "GRT": ["The Graph protocol"], "YFI": ["Yearn"], "SNX": ["Synthetix"], "HNT": ["Helium Network", "Helium"], "IMX": ["Immutable X"],
    "CAKE": ["PancakeSwap"], "SAND": ["The Sandbox"], "RUNE": ["THORChain"], "TON": ["Toncoin"], "SUI": ["Sui"],
    "VET": ["VeChain"], "BLUR": ["Blur"], "BSV": ["Bitcoin SV"], "FLOKI": ["Floki"], "HBAR": ["Hedera"], "WAVES": ["Waves blockchain"],
    "ZEC": ["Zcash"], "CHZ": ["Chiliz"], "LRC": ["Loopring"], "GMX": ["GMX"], "ENJ": ["Enjin"], "EGLD": ["Elrond", "MultiversX"],
    "KSM": ["Kusama"], "CRO": ["Cronos"], "DASH": ["Dash cryptocurrency"], "INJ": ["Injective"], "IOTA": ["IOTA"], "QNT": ["Quant Network"],
    "GALA": ["Gala Games"], "KAVA": ["Kava"], "DYDX": ["dYdX"], "MINA": ["Mina"], "ZIL": ["Zilliqa"], "RNDR": ["Render Network"],
    "OSMO": ["Osmosis"], "AR": ["Arweave"], "FLOW": ["Flow blockchain"], "ONE": ["Harmony One", "Harmony blockchain"], "NEO": ["NEO"],
    "FET": ["Fetch.ai"], "AGIX": ["SingularityNET"], "OCEAN": ["Ocean Protocol"], "BAT": ["Basic Attention Token"],
    "ANKR": ["Ankr"], "AUDIO": ["Audius"], "LPT": ["Livepeer"], "MASK": ["Mask Network"], "WOO": ["WOO Network"],
    "ROSE": ["Oasis Network"], "KLAY": ["Klaytn"], "CFX": ["Conflux"], "JASMY": ["Jasmy"], "GMT": ["STEPN"],
    "WLD": ["Worldcoin"], "SEI": ["Sei"], "TIA": ["Celestia"], "PYTH": ["Pyth"], "JUP": ["Jupiter"], "BONK": ["Bonk"],
}
# tickers safe to match bare (not ordinary words); others need a $ prefix
BARE_OK = {"BTC", "ETH", "SOL", "ADA", "XRP", "DOGE", "DOT", "BNB", "MATIC", "AVAX", "SHIB", "UNI", "LTC", "LINK", "BCH",
           "AAVE", "ARB", "ATOM", "LDO", "AXS", "CRV", "MKR", "FTM", "TRX", "APT", "ETC", "XLM", "ICP", "XTZ", "ALGO",
           "FIL", "STX", "FTT", "XMR", "CELO", "GRT", "YFI", "SNX", "HNT", "IMX", "TON", "VET", "BSV", "HBAR", "ZEC",
           "CHZ", "LRC", "EGLD", "KSM", "CRO", "INJ", "QNT", "DYDX", "RNDR", "OSMO", "AGIX", "ANKR", "LPT", "KLAY", "CFX",
           "JASMY", "WLD", "TIA", "PYTH", "BONK", "PEPE", "FLOKI", "APE", "GALA", "KAVA", "MINA", "ZIL", "EOS",
           "SUSHI", "RUNE", "WAVES", "CAKE", "UST", "LUNA", "NEAR", "MANA", "SAND", "GMT", "SUI", "SEI"}
STABLE = {"USDT", "USDC", "BUSD", "DAI", "TUSD", "USDP", "FDUSD", "UST"}


def _pattern(tk: str, names: list[str]) -> re.Pattern:
    alts = [re.escape(n) for n in names]
    alts.append(r"\$" + re.escape(tk))
    if tk in BARE_OK:
        alts.append(re.escape(tk))
    return re.compile(r"(?<![A-Za-z0-9])(" + "|".join(alts) + r")(?![A-Za-z0-9])")


PATTERNS = {tk: _pattern(tk, ns) for tk, ns in NAMES.items()}
# "Bitcoin Cash", "Bitcoin SV", "Ethereum Classic" also contain the parent name; parent is not tagged by those alone
_SUBSUMES = {"BTC": ["Bitcoin Cash", "Bitcoin SV"], "ETH": ["Ethereum Classic"], "LUNA": ["TerraUSD"],
             "XRP": ["Ripple Effect", "Ripple effect", "ripple effect"]}


def tag(title: str) -> list[str]:
    if not isinstance(title, str):      # missing titles (pandas 3 keeps NaN through astype(str))
        return []
    out = []
    for tk, p in PATTERNS.items():
        if p.search(title):
            if tk in _SUBSUMES:
                stripped = title
                for s in _SUBSUMES[tk]:
                    stripped = stripped.replace(s, "")
                if not p.search(stripped):
                    continue
            out.append(tk)
    return out
