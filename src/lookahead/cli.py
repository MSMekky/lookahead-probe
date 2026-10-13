"""Command line: python -m lookahead <step>.

  data      download CryptoVision, fetch prices, build the coin-week panel
  extract   hidden states for every vintage (GPU)
  analyze   probes, PVI, event study, robustness, remedy
  simulate  one power-simulation cell (see scripts/power_grid.py for the full grid)
  check     validate the extraction path on a few items (truncation and dtype)
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from .config import CFG, INTERIM, RESULTS


def cmd_data(a):
    from .data import cryptovision, hfnews, prices
    from . import panel
    INTERIM.mkdir(parents=True, exist_ok=True)
    print("downloading CryptoVision and the Hugging Face feeds", flush=True)
    cryptovision.download()
    for name in hfnews.SOURCES:
        hfnews.download(name)
    parts = {"cryptovision": cryptovision.load(), "cryptocompare_feed": hfnews.load_coindesk()}
    for k, v in parts.items():
        v["dataset"] = k
        print(f"  {k}: {len(v)} coin-headline rows, {v['coin'].nunique()} coins", flush=True)
    news = pd.concat(parts.values(), ignore_index=True)
    news = news[~news["coin"].isin(["USDT", "USDC", "BUSD", "DAI", "TUSD"])]
    # roundups ("Price analysis: BTC, ETH, XRP, ...") are not about any one coin
    n_named = news.groupby(["title"])["coin"].transform("nunique")
    news = news[n_named <= CFG.panel.max_coins_per_headline]
    key = news["title"].str.lower().str.replace(r"[^a-z0-9 ]", "", regex=True).str.strip()
    news = news.assign(_k=key).drop_duplicates(subset=["coin", "_k"]).drop(columns="_k").sort_values("ts")
    print(f"headlines after cross-source dedup {len(news)}, coins {news['coin'].nunique()}", flush=True)
    news.to_parquet(INTERIM / "news.parquet")
    pc = CFG.panel
    ts = news["ts"].dt.tz_convert("UTC").dt.tz_localize(None)
    window = news[(ts >= pd.Timestamp(pc.first_week)) & (ts < pd.Timestamp(pc.last_week) + pd.Timedelta(days=7))]
    counts = window["coin"].value_counts()
    coins = counts[counts >= a.min_headlines].index.tolist()
    if pc.benchmark not in coins:
        coins.append(pc.benchmark)
    print(f"fetching prices for {len(coins)} coins with >= {a.min_headlines} headlines in the window", flush=True)
    end = (pd.Timestamp(pc.last_week) + pd.Timedelta(days=7 + pc.horizon_days + 5)).strftime("%Y-%m-%d")
    px = prices.fetch(coins, (pd.Timestamp(pc.first_week) - pd.Timedelta(days=5)).strftime("%Y-%m-%d"), end)
    items = panel.build(news, px)
    items.to_parquet(INTERIM / "items.parquet")
    s = panel.summary(items)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "panel_summary.json").write_text(json.dumps(s, indent=2, default=str))
    print(json.dumps(s, indent=2, default=str))


def cmd_extract(a):
    from . import extract
    items = pd.read_parquet(INTERIM / "items.parquet")
    texts = items["text"].tolist()[: a.limit] if a.limit else items["text"].tolist()
    vint = [int(v) for v in a.vintages.split(",")] if a.vintages else None
    extract.run(texts, vintages=vint, device=a.device, delete_weights=a.delete_weights)


def cmd_analyze(a):
    from . import analysis
    items = pd.read_parquet(INTERIM / "items.parquet")
    res = analysis.run(items, do_remedy=not a.no_remedy)
    h = res["h1"]
    print(json.dumps(dict(beta0=h["beta0"], se0=h["se0"], p=h["p0_one_sided"], reject=h["reject_h1"],
                          pretrend_p=h["pretrend_p"], leak_gap=res["leak_gap_nats"]), indent=2))


def cmd_simulate(a):
    from . import sim
    import numpy as np
    rs = [sim.one_run(s, a.coins, a.gamma, "coverage", True, True) for s in range(a.sims)]
    print(dict(power=float(np.mean([r["reject_h1"] for r in rs])), pretrend=float(np.mean([r["reject_pre"] for r in rs]))))


def cmd_check(a):
    """Validate extraction on a few items, all on the GPU:
    (1) hooks vs the model's own hidden_states, (2) truncated vs full model, (3) fp16 vs fp32."""
    import numpy as np
    import torch
    from . import extract
    items = pd.read_parquet(INTERIM / "items.parquet")
    texts = items["text"].tolist()[: a.n]
    mc = CFG.model
    repo = mc.repo_template.format(v=a.vintage)
    cos = lambda X, Y: float(min(np.dot(x, y) / np.linalg.norm(x) / np.linalg.norm(y)
                                 for x, y in zip(X.reshape(-1, X.shape[-1]), Y.reshape(-1, Y.shape[-1]))))
    out = dict(repo=repo, n=len(texts))
    tok, full = extract.load(repo, "float16", a.device)
    H_full = extract.encode(texts, tok, full, mc.layers, mc.max_tokens, 4, a.device)
    with torch.no_grad():
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=mc.max_tokens,
                  return_token_type_ids=False).to(a.device)
        hs = full(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], output_hidden_states=True).hidden_states
        m = enc["attention_mask"].unsqueeze(-1).float()
        native = np.stack([((hs[L].float() * m).sum(1) / m.sum(1)).cpu().numpy() for L in mc.layers], 1)
    out["hooks_vs_hidden_states_min_cos"] = cos(H_full, native)
    del full, hs; torch.cuda.empty_cache()
    tok, cut = extract.load(repo, "float16", a.device, keep_blocks=max(mc.layers))
    H_cut = extract.encode(texts, tok, cut, mc.layers, mc.max_tokens, 4, a.device)
    out["truncated_vs_full_min_cos"] = cos(H_cut, H_full)
    out["fp16_finite"] = bool(np.isfinite(H_cut).all())
    del cut; torch.cuda.empty_cache()
    # fp32 reference only for the first block group: the full fp32 model would exceed 6 GB of VRAM, and on
    # Windows that spills into shared system memory instead of failing, which can freeze the machine
    L0 = min(mc.layers)
    tok, m32 = extract.load(repo, "float32", a.device, keep_blocks=L0)
    H32 = extract.encode(texts, tok, m32, (L0,), mc.max_tokens, 2, a.device)
    out[f"fp16_vs_fp32_min_cos_layer{L0}"] = cos(H_cut[:, [0]], H32)
    del m32; torch.cuda.empty_cache()
    print(json.dumps(out, indent=2))
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "extraction_check.json").write_text(json.dumps(out, indent=2))


def main(argv=None):
    p = argparse.ArgumentParser(prog="lookahead")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("data"); s.add_argument("--min-headlines", type=int, default=20); s.set_defaults(f=cmd_data)
    s = sub.add_parser("extract"); s.add_argument("--vintages"); s.add_argument("--limit", type=int)
    s.add_argument("--device", default="cuda"); s.add_argument("--delete-weights", action="store_true"); s.set_defaults(f=cmd_extract)
    s = sub.add_parser("analyze"); s.add_argument("--no-remedy", action="store_true"); s.set_defaults(f=cmd_analyze)
    s = sub.add_parser("simulate"); s.add_argument("--coins", type=int, default=100)
    s.add_argument("--gamma", type=float, default=0.1); s.add_argument("--sims", type=int, default=20); s.set_defaults(f=cmd_simulate)
    s = sub.add_parser("check"); s.add_argument("--vintage", type=int, default=2018); s.add_argument("--n", type=int, default=8)
    s.add_argument("--device", default="cuda"); s.set_defaults(f=cmd_check)
    a = p.parse_args(argv)
    a.f(a)


if __name__ == "__main__":
    main()
