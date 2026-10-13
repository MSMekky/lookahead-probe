# Power analysis

Monte Carlo study of the panel design, run before any model was touched. It answers two questions: does the event study stay quiet when there is no leakage, and how large must a leak be for it to find it. Code: `src/lookahead/sim.py`, `scripts/power_grid.py`. Raw runs: `results/power_grid.jsonl`.

## Simulated world

- Coin-week items over 3 outcome years (2021 to 2023), 52 weeks each; a coin is present in a week with probability 0.6.
- Text carries a legitimate signal about the market-neutral outcome (R² = 0.05).
- Coverage per coin is log-normal and three times higher for events.
- Seven annual vintages (2018 to 2024) produce 48-dimensional hidden states with three planted effects:
  - capability gain of 10% per vintage;
  - recency boost of 30% at k = -1 and 0, scaled by coverage (the harder case);
  - leakage γ · √coverage · outcome along one direction, full strength at k = 0, then 0.8 and 0.6.
- Pipeline as on real data: ridge probes per vintage, coin-held-out folds, Gaussian PVI per item, stacked event study, coin-clustered SEs, one-sided test of β0 at 5%.

## Finding 1: the v2 specification over-rejects

With events defined on the raw return and only vintage-by-week and item FE, H1 is rejected in 25% of leak-free panels and the pre-trend test fails in 100%. Selecting on large outcomes selects items whose text predicts more, so capability and recency gains raise their PVI more.

## Finding 2: three corrections restore size

| Specification | H1 false positives | Pre-trend rejections |
| --- | --- | --- |
| v2: raw-return events, vintage-by-week and item FE | 25% | 100% |
| + vintage-by-event-status FE | 7.5% | 88% |
| + surprise-defined events | 6.7% | 35% |
| + vintage slopes on predictability and coverage (adopted) | 5% to 7% | 12% to 90%, rising with N |

## Finding 3: power (adopted specification, 60 panels per cell)

| Coins (items) | γ = 0 | 0.05 | 0.1 | 0.2 | 0.4 |
| --- | --- | --- | --- | --- | --- |
| 50 (about 4,700) | 5% | 10% | 27% | 80% | 100% |
| 100 (about 9,300) | 7% | 20% | 47% | 97% | 100% |
| 200 (about 18,700) | 7% | 20% | 72% | 100% | 100% |

Leak size in PVI units (extra PVI on events at k = 0 over k = -1, net of the γ = 0 baseline): γ = 0.05 is 0.006 nats, 0.1 is 0.026, 0.2 is 0.10, 0.4 is 0.37. The clean probe extracts 0.014 nats on an average item and 0.074 on an event item. At 80% power the design detects about 0.10 nats with 50 coins and about 0.03 to 0.05 nats with 200 coins.

## Finding 4: the pre-trend test cannot be a gate

Under coverage-scaled recency the joint test of β-2 and β-3 rejects in 12% of leak-free panels at 50 coins, 50% at 100 and 90% at 200. H1 therefore rests on β0 against k = -1; pre-trend coefficients are reported next to β0 as a diagnostic.

## Finding 5: what no passive design can separate

Familiarity with a period's text that arrives together with its outcomes is a k = 0 effect by construction. Annual vintages cannot separate it from leakage. Monthly checkpoints could (PIT trains them but publishes only annual snapshots); otherwise only a planted test with known outcomes can.

## Limits

Hidden states are synthetic and linear, the leak lies on one direction, clustering is by coin only.
