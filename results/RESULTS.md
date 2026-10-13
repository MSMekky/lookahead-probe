# Results: PIT-1B, layer 26

Panel: 5327 coin-weeks, 85 coins, 156 weeks, 383 events
(surprise-defined against the 2018 vintage, plus the failure catalogue). Vintages [2018, 2019, 2020, 2021, 2022, 2023, 2024].

## H1: leakage at first exposure

beta0 = -0.0109 nats (SE 0.0089, coin-clustered, 85 clusters), one-sided p = 0.886.
Null rejected at 5%: **False**. One-sided 95% upper bound on beta0: +0.0040 nats. Two-way clustered (coin and week): beta0 -0.0109, SE 0.0089, p = 0.886.
Descriptive leak gap (events, k = 0 minus k = -1): -0.0059 nats.
Pre-trend joint test (a diagnostic, not a gate; see docs/power_analysis.md): p = 0.233.

| k | beta | SE | t |
| --- | --- | --- | --- |
| -3 | +0.0276 | 0.0176 | +1.56 |
| -2 | +0.0140 | 0.0084 | +1.67 |
| 0 | -0.0109 | 0.0089 | -1.22 |
| 1 | -0.0211 | 0.0167 | -1.26 |
| 2 | -0.0350 | 0.0266 | -1.32 |

v2 specification for contrast (raw-return events, no vintage-by-event terms): beta0 -0.0065, p = 0.934.
Cue-free subset: beta0 -0.0192, p = 0.945, n = 3306.

## Other layers

| layer | beta0 | SE | p |
| --- | --- | --- | --- |
| 13 | -0.0004 | 0.0135 | 0.512 |
| 39 | -0.0031 | 0.0105 | 0.617 |

## Probe quality

Mean PVI by vintage: 2018: -0.0008 | 2019: 0.0012 | 2020: 0.0022 | 2021: 0.0000 | 2022: 0.0005 | 2023: 0.0008 | 2024: 0.0005

| vintage | PVI | control task | selectivity |
| --- | --- | --- | --- |
| 2018 | -0.0008 | -0.0006 | -0.0002 |
| 2024 | 0.0005 | -0.0006 | 0.0012 |

## H5: residual erasure

Leak gap -0.0059 before, -0.0007 after. beta0 after erasure +0.0030 (p = 0.063).
Mean PVI on pre-exposure cells: +0.0002 before, +0.0066 after.
Retention ratio: not meaningful (baseline below 0.005 nats).
