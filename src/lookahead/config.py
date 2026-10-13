"""Pre-registered settings. Changing any of these after looking at results is a deviation and must be logged."""
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
FEATURES = DATA / "features"
RESULTS = ROOT / "results"


@dataclass(frozen=True)
class PanelConfig:
    first_week: str = "2020-11-30"      # first Monday whose outcome window can end in 2021
    last_week: str = "2023-11-27"       # last Monday whose outcome window ends in 2023
    horizon_days: int = 30              # outcome window after the week closes
    benchmark: str = "BTC"              # market-neutral target: return in excess of BTC
    max_headlines: int = 12             # headlines per coin-week fed to the model
    max_coins_per_headline: int = 4     # roundup headlines naming more coins are dropped
    min_coins_per_week: int = 10        # weeks with fewer priced coins are dropped
    event_tail: float = 0.025           # each tail of the weekly surprise distribution
    outcome_years: tuple = (2021, 2022, 2023)


@dataclass(frozen=True)
class ModelConfig:
    family: str = "PIT-1B"
    repo_template: str = "Diamegs/PIT-1B-{v}12"
    vintages: tuple = (2018, 2019, 2020, 2021, 2022, 2023, 2024)
    reference_vintage: int = 2018       # clean reference: has seen no outcome year
    layers: tuple = (13, 26, 39)        # 1/4, 1/2, 3/4 of 52 blocks
    main_layer: int = 26                # pre-registered layer for H1
    max_tokens: int = 256
    batch_size: int = 16
    dtype: str = "float16"


@dataclass(frozen=True)
class ProbeConfig:
    n_folds: int = 5                    # coin-held-out folds
    alphas: tuple = tuple(10.0 ** k for k in range(-1, 6))
    seed: int = 0


@dataclass(frozen=True)
class StudyConfig:
    k_bins: tuple = (-3, -2, 0, 1, 2)   # k = -1 is the reference bin
    alpha_h1: float = 0.05              # one-sided
    panel: PanelConfig = field(default_factory=PanelConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    probe: ProbeConfig = field(default_factory=ProbeConfig)


CFG = StudyConfig()

# Failure catalogue (2022). Coin-weeks containing these dates are events regardless of the surprise rule.
CATALOGUE = [
    ("LUNA", "2022-05-07"), ("UST", "2022-05-07"), ("LUNA", "2022-05-12"),
    ("CEL", "2022-06-12"), ("FTT", "2022-11-08"), ("FTT", "2022-11-11"),
]

# Frozen cue lexicon for the cue-free robustness subset (headlines naming the outcome).
CUE_WORDS = [
    "depeg", "de-peg", "collapse", "crash", "plunge", "halt", "insolven", "bankrupt",
    "chapter 11", "liquidat", "soar", "surge", "skyrocket", "rally", "all-time high",
    "ath", "delist", "hack", "exploit", "rug", "fraud", "default", "freeze", "withdrawals",
]
