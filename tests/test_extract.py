"""Extraction on a tiny randomly initialised GPT-2 (no download needed)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")

from lookahead import extract  # noqa: E402


@pytest.fixture(scope="module")
def tiny():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
    words = "crypto news headlines about btc eth published in the week of january 2022 - price up down falls rises".split()
    vocab = {w: i for i, w in enumerate(["[UNK]", "[PAD]"] + words)}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]")); tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="[UNK]", pad_token="[PAD]")
    tok.padding_side = "right"
    torch.manual_seed(0)
    model = GPT2LMHeadModel(GPT2Config(vocab_size=len(vocab), n_embd=32, n_layer=6, n_head=4, n_positions=64)).eval()
    return tok, model


TEXTS = ["crypto news about btc price up", "eth falls", "headlines about eth published in the week of january 2022 - price down",
         "btc rises"]


def test_hook_pooling_matches_hidden_states(tiny):
    tok, model = tiny
    H = extract.encode(TEXTS, tok, model, (2, 4), 64, 2, "cpu")
    with torch.no_grad():
        enc = tok(TEXTS, return_tensors="pt", padding=True, return_token_type_ids=False)
        hs = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], output_hidden_states=True).hidden_states
    m = enc["attention_mask"].unsqueeze(-1).float()
    for j, L in enumerate((2, 4)):
        ref = ((hs[L] * m).sum(1) / m.sum(1)).numpy()
        assert np.allclose(H[:, j], ref, atol=1e-5)


def test_padding_and_batching_do_not_change_features(tiny):
    tok, model = tiny
    a = extract.encode(TEXTS, tok, model, (3,), 64, 1, "cpu")
    b = extract.encode(TEXTS, tok, model, (3,), 64, 4, "cpu")
    assert np.allclose(a, b, atol=1e-5)


def test_truncation_keeps_features(tiny):
    import copy
    tok, model = tiny
    full = extract.encode(TEXTS, tok, model, (2, 3), 64, 2, "cpu")
    m2 = copy.deepcopy(model)
    extract.truncate(m2, 3)
    assert len(extract.blocks_of(m2)) == 3
    cut = extract.encode(TEXTS, tok, m2, (2, 3), 64, 2, "cpu")
    assert np.allclose(full, cut, atol=1e-6)


def test_loading_with_fewer_blocks_matches_full_model(tiny, tmp_path):
    tok, model = tiny
    model.save_pretrained(tmp_path); tok.save_pretrained(tmp_path)
    tok2, cut = extract.load(str(tmp_path), "float32", "cpu", keep_blocks=3)
    assert len(extract.blocks_of(cut)) == 3
    full = extract.encode(TEXTS, tok, model, (2, 3), 64, 2, "cpu")
    part = extract.encode(TEXTS, tok2, cut, (2, 3), 64, 2, "cpu")
    assert np.allclose(full, part, atol=1e-5)
