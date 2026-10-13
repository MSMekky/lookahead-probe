"""Hidden-state extraction: mean-pooled residual stream at fixed layers, one file per vintage.

Right padding is used on purpose: the PIT modelling code derives positions from the sequence
index, not the attention mask, so left padding would shift positions. With right padding and a
causal mask the real tokens never attend to pads, and pooling uses the mask.
Only the blocks up to the deepest requested layer are run, which also lets larger models fit.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from .config import CFG, FEATURES, ModelConfig


def truncate(model, n_blocks: int) -> None:
    """Drop blocks beyond n_blocks in place; their outputs are never used and they cost VRAM."""
    blocks = blocks_of(model)
    del blocks[n_blocks:]
    for attr in ("n_layer", "num_hidden_layers", "n_layers"):
        if hasattr(model.config, attr):
            setattr(model.config, attr, n_blocks)


def load(repo: str, dtype: str = "float16", device: str = "cuda", keep_blocks: int | None = None):
    """Load a model. With keep_blocks, the config is cut to that many blocks before loading, so weights of
    later blocks are never read: peak VRAM and host RAM scale with the blocks actually used."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    cfg = AutoConfig.from_pretrained(repo, trust_remote_code=True)
    if keep_blocks:
        for attr in ("n_layer", "num_hidden_layers", "n_layers"):
            if hasattr(cfg, attr):
                setattr(cfg, attr, keep_blocks)
    kw = dict(config=cfg, trust_remote_code=True, dtype=getattr(torch, dtype), low_cpu_mem_usage=True)
    if device == "cuda":
        kw["device_map"] = {"": 0}     # stream weights straight to the GPU; keeps host RAM use low
    model = AutoModelForCausalLM.from_pretrained(repo, **kw)
    if keep_blocks:
        assert len(blocks_of(model)) == keep_blocks, "block count does not match the truncated config"
    return tok, model.to(device).eval()


def blocks_of(model) -> torch.nn.ModuleList:
    for path in ("transformer.h", "model.layers", "gpt_neox.layers", "transformer.blocks", "layers"):
        obj = model
        try:
            for a in path.split("."):
                obj = getattr(obj, a) if not isinstance(obj, dict) else obj[a]
            if isinstance(obj, torch.nn.ModuleList):
                return obj
        except (AttributeError, KeyError, TypeError):
            continue
    raise AttributeError("cannot find the list of transformer blocks")


class _Stop(Exception):
    pass


@torch.no_grad()
def encode(texts: list[str], tok, model, layers: tuple[int, ...], max_tokens: int, batch_size: int,
           device: str = "cuda", token_budget: int = 1536) -> np.ndarray:
    """Return array (n, len(layers), d) of mean-pooled outputs of blocks `layers` (1-indexed: layer L
    is the output of block L-1, i.e. hidden_states[L] in the transformers convention)."""
    blocks = blocks_of(model)
    want = {L - 1: i for i, L in enumerate(layers)}
    deepest = max(want)
    cache: dict[int, torch.Tensor] = {}
    hooks = []
    for b, i in want.items():
        def hook(_m, _inp, out, i=i, b=b):
            cache[i] = out[0] if isinstance(out, tuple) else out
            if b == deepest:
                raise _Stop
        hooks.append(blocks[b].register_forward_hook(hook))
    # Batches are sized by tokens, not items: long texts at the end otherwise push VRAM past its limit,
    # and on Windows that spills into shared system memory and slows the GPU by two orders of magnitude.
    lens = np.array([len(ids) for ids in tok(texts, truncation=True, max_length=max_tokens)["input_ids"]])
    order = np.argsort(lens, kind="stable")
    batches, cur = [], []
    for j in order:
        if cur and (len(cur) + 1) * lens[j] > token_budget or len(cur) == batch_size:
            batches.append(cur); cur = []
        cur.append(j)
    if cur:
        batches.append(cur)
    out = None
    t0 = last = time.time()
    done = 0
    try:
        for bi, idx in enumerate(batches):
            idx = np.array(idx)
            enc = tok([texts[j] for j in idx], return_tensors="pt", padding=True, truncation=True,
                      max_length=max_tokens, return_token_type_ids=False).to(device)
            cache.clear()
            try:
                model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], use_cache=False)
            except _Stop:
                pass
            m = enc["attention_mask"].unsqueeze(-1).float()
            pooled = torch.stack([(cache[i].float() * m).sum(1) / m.sum(1) for i in range(len(layers))], 1)
            if out is None:
                out = np.zeros((len(texts), len(layers), pooled.shape[-1]), dtype=np.float32)
            out[idx] = pooled.cpu().numpy()
            done += len(idx)
            del enc, m, pooled
            cache.clear()
            if device == "cuda" and bi % 25 == 0:
                torch.cuda.empty_cache()
            if time.time() - last > 60 or bi == len(batches) - 1:
                last = time.time()
                rate = done / max(last - t0, 1e-6)
                mem = f"  peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB" if device == "cuda" else ""
                print(f"    {done}/{len(texts)}  {rate:.1f} items/s  eta {(len(texts)-done)/max(rate,1e-6)/60:.1f} min{mem}",
                      flush=True)
    finally:
        for h in hooks:
            h.remove()
    if not np.isfinite(out).all():
        raise FloatingPointError("non-finite hidden states; try dtype=float32 or bfloat16")
    return out


def run(texts: list[str], mc: ModelConfig = CFG.model, out_dir: Path = FEATURES, device: str = "cuda",
        vintages=None, delete_weights: bool = False) -> None:
    import threading
    from huggingface_hub import snapshot_download
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [v for v in (vintages or mc.vintages) if not (out_dir / f"{mc.family}_{v}.npy").exists()]
    prefetch = None
    for i, v in enumerate(todo):
        if prefetch is not None:
            prefetch.join()
        if i + 1 < len(todo):       # download the next vintage while this one runs on the GPU
            nxt = mc.repo_template.format(v=todo[i + 1])
            prefetch = threading.Thread(target=snapshot_download, args=(nxt,), daemon=True)
            prefetch.start()
        f = out_dir / f"{mc.family}_{v}.npy"
        if f.exists():
            print(f"  {f.name} exists, skipping", flush=True); continue
        repo = mc.repo_template.format(v=v)
        print(f"  loading {repo}", flush=True)
        tok, model = load(repo, mc.dtype, device, keep_blocks=max(mc.layers))
        H = encode(texts, tok, model, mc.layers, mc.max_tokens, mc.batch_size, device)
        np.save(f, H.astype(np.float16))
        print(f"  saved {f.name} {H.shape}", flush=True)
        del model
        torch.cuda.empty_cache() if device == "cuda" else None
        if delete_weights:
            from huggingface_hub import scan_cache_dir
            info = scan_cache_dir()
            hashes = [r.commit_hash for repo_ in info.repos if repo_.repo_id == repo for r in repo_.revisions]
            if hashes:
                info.delete_revisions(*hashes).execute()
