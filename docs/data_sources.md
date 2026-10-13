# Data sources

Everything below was checked on 2026-10-12 against the live sources. Decisions taken after looking at the data are recorded here, with the reason.

## Headlines

| Source | What it is | Licence | Used for |
| --- | --- | --- | --- |
| CryptoVision v2, Mendeley Data `3c3xtxtfb6` (DOI 10.17632/3c3xtxtfb6.2) | 188k headlines, 2017 to 2025, six outlets (CoinDesk, Cointelegraph, Decrypt, Blockworks, CryptoNews, CryptoPanic). Titles and URLs only, no article bodies. | CC BY 4.0 (from the dataset metadata) | Headlines |
| `maryamfakhari/crypto-news-coindesk-2020-2025` on Hugging Face | The CryptoCompare news feed: 229k items, Oct 2019 to Feb 2025, about 40 outlets, with a `categories` field of tickers and topics | CC BY-NC 4.0 | Headlines |

Rejected:
- **DLT-Sentiment-News.** About 100 vote-selected headlines a week across all coins, no coin tag. Too thin.
- **`SahandNZ/cryptonews-articles-with-price-momentum-labels`.** It stores preprocessed sentence fragments (lowercased, stopwords removed), not headlines.

Because of the CC BY-NC source, the derived panel inherits a non-commercial restriction. The code is not affected.

### Coin assignment

CryptoVision's own `Coin Type` label covers only 16 coins. With it alone the panel had 14 priced coins, below the 50-coin floor from the power analysis. Coins are therefore assigned by what the headline names (`src/lookahead/data/coins.py`):

- **Matching rule.** A name match is case-sensitive and bounded by word edges. A bare ticker is accepted only for tickers that are not ordinary words; otherwise only the `$TICKER` form counts.
- **Names that are common words.** These are matched only in a qualified form: "Curve Finance", "Compound Finance", "Optimism network", "Stellar Lumens", and similar.
- **Parent coins.** "Bitcoin Cash", "Bitcoin SV" and "Ethereum Classic" do not tag the parent coin. "Ripple Effect" does not tag XRP.
- **CryptoCompare categories.** Used only when the title names no coin and the categories list exactly one. The feed attaches many tickers to generic market news, for example "Bitcoin and Altcoins Face Heat Amid Renewed COVID-19 Fear" carried XLM.
- **Roundups.** Headlines naming more than four coins ("Price analysis: BTC, ETH, XRP, ...") are dropped for every coin.
- **Duplicates.** Removed per coin on a normalised title, across sources.

## Prices

Daily closes in USDT (USD for Bitfinex), with this order of sources:
1. Binance spot.
2. KuCoin, if Binance covers less than 90% of the window or has a gap over 7 days. Such a gap can hide a ticker being reused for a different asset.
3. Gate, under the same rule.

No source needs an API key. CryptoCompare now requires one (HTTP 401), and so do CoinGecko's and CoinPaprika's historical endpoints.

The three 2022 failure coins use explicit plans, chosen after checking every source around May 2022:

| Coin | Plan | Why |
| --- | --- | --- |
| LUNA | Gate `LUNA_USDT` to 2022-05-26, then Gate `LUNC_USDT` from 2022-05-27 | Binance `LUNAUSDT` and Gate `LUNA_USDT` both switch to Terra 2.0 in late May 2022. Binance `LUNCUSDT` starts in September 2022. KuCoin `LUNC-USDT` starts after the crash. |
| UST | Bitfinex `tTERRAUST:USD` (to 2023-05-30) | Bitfinex's `UST` is Tether. Gate `USTC_USDT` starts on 2022-05-27. |
| CEL | Gate `CEL_USDT` | Not listed on Binance, KuCoin or OKX. |

A coin with no source is dropped and listed in `data/raw/prices.sources.csv`. Prices are never imputed.

## Models

PIT-1B (`Diamegs/PIT-1B-YYYY12`, Apache-2.0). Twelve annual snapshots are public, December of 2013 to 2024 (November for 2015). The paper trains monthly checkpoints, but they are not published. The vintages used are 2018 to 2024; 2018 is the clean reference for outcomes in 2021 to 2023.
