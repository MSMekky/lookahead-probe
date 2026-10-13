# lookahead-probe

White-box detection of lookahead bias in financial LLMs. The question is whether a language model's hidden states carry information about a market outcome that the model could only have learned by seeing the future.

The test uses a family of point-in-time models trained on one chronological data stream (PIT-1B, annual vintages 2018 to 2024). It asks whether the realised 30-day return of a coin can be decoded from that coin's pre-event headlines better by vintages that have seen the outcome year than by vintages that have not, beyond what capability and recency explain.

## Design in one paragraph

Every coin-week (headlines published that week) is encoded by every vintage. A ridge probe per vintage, cross-fitted with coin-held-out folds, gives each item a pointwise V-information score (PVI): the log-likelihood gain over a constant predictor. A stacked event study compares event coins with same-week non-event coins across the distance k = vintage minus outcome year, with these terms:
- item and vintage-by-week fixed effects;
- vintage-by-event-status fixed effects;
- vintage-specific slopes on the clean model's predictability and on coverage.

β0 at k = 0 against k = -1 is the leakage estimate. Events are surprises against the clean 2018 vintage, not raw returns. The power analysis (`docs/power_analysis.md`) shows that the raw-return definition rejects in 25% of leak-free panels. A residual LEACE eraser (`remedy.py`) removes only the part of the outcome the clean model cannot predict.

## Layout

```
src/lookahead/
  data/        CryptoVision + CryptoCompare-feed loaders, coin tagger, prices (Binance, KuCoin, Gate, Bitfinex)
  panel.py     coin-week items, market-neutral targets, cue flags, failure catalogue
  extract.py   hidden states per vintage (fp16, right padding, only the blocks that are needed)
  probes.py    ridge probes, GCV ridge path, cross-fitted PVI, control-task selectivity
  eventstudy.py surprise events, stacked event study, coin or two-way clustered SEs
  remedy.py    residual LEACE and its evaluation
  analysis.py  the full analysis and figure
  sim.py       the Monte Carlo used for the power analysis
docs/          power analysis, data sources and every data decision
tests/         unit and end-to-end tests (simulated features, tiny random GPT-2)
```

## Run

Windows with an NVIDIA GPU (6 GB is enough for PIT-1B):

```
scripts\setup_windows.bat
scripts\run_all.bat
```

Linux or macOS with CUDA: `pip install -e .[gpu,dev]` (install the CUDA build of torch first), then `scripts/run_all.sh`.

The steps can also run one at a time:

```
python -m lookahead data      # download headlines, fetch prices, build data/interim/items.parquet
python -m lookahead check     # hooks vs hidden_states, truncation, fp16 vs fp32 on a few items
python -m lookahead extract   # 7 vintages, about 6.5 GB of weights each, cached by Hugging Face
python -m lookahead analyze   # results/results.json and results/event_study.png
pytest
```

## Data and licences

Headlines come from CryptoVision (CC BY 4.0) and the CryptoCompare news feed as published on Hugging Face (CC BY-NC 4.0), so derived data is for non-commercial research. Models are PIT-1B (Apache-2.0). Prices come from public exchange APIs. Details and every judgement call are in `docs/data_sources.md`.

## Results (PIT-1B, 2018 to 2024, run on 2026-10-13)

Full numbers in `results/RESULTS.md`, the figure in `results/event_study.png`.

- **No leakage detected at first exposure.** β0 = -0.011 nats (SE 0.009, 85 coin clusters, one-sided p = 0.89). The one-sided 95% upper bound is +0.004 nats. That is below the smallest leak the design detects with 80% power in simulation (about 0.03 to 0.10 nats). The power analysis uses synthetic hidden states, though, and no positive control has yet been run on real ones. Until a planted leak is shown to be recoverable, this null cannot separate "no leakage" from "a readout that cannot see it".
- **The result is robust:**
  - two-way clustering by coin and week gives p = 0.89;
  - the cue-free subset (3,306 items) gives p = 0.95;
  - layer 13 gives p = 0.51;
  - layer 39 gives p = 0.62;
  - the v2 specification gives p = 0.93.
- **The legitimate signal is also tiny.** Probe correlation with the 30-day market-neutral return is 0.06 to 0.08 in every vintage, and mean PVI lies between -0.001 and +0.002 nats. Selectivity over a shuffled-label control is at most 0.0012 nats. Headlines of the week carry almost no linear, decodable information about the following month's relative return, before or after a vintage has seen it.
- **Remedy (H5).** With no leak to remove, residual erasure is a no-op on the leak gap (-0.006 to -0.001 nats). The retention ratio is not meaningful at this baseline.
- **Estimator check.** The event-study coefficients were reproduced exactly by a direct sparse OLS with every fixed effect as an explicit dummy.

What the null does and does not say: PIT-1B's mid-layer residual stream, pooled over the week's headlines, does not encode which coins went on to outperform in a way a linear probe can read. It says nothing about larger models, about generation-time recall under prompting, or about date-conditioned recall from a single name and date.

## Status

Known limits:
- **No positive control yet.** The next step is a planted leak: fine-tune a pre-exposure vintage on synthetic documents about an invented coin and its outcome at known doses, then check that the probe recovers it.
- **Monthly checkpoints:** not public, so familiarity with a period's text and knowledge of its outcomes cannot be separated (power analysis, finding 5).
- **Sample size:** the panel has about 5,300 coin-weeks over 85 coins, between the 50- and 100-coin cases in the power analysis.
