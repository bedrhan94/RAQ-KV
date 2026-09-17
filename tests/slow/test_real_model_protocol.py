"""Protocol checks on the real pinned model (marked slow; needs the model in the HF cache).

Run:  python -m pytest -m slow tests/slow            (CPU, float32, ~8 GB RAM)
      RAQ_TEST_DEVICE=cuda python -m pytest -m slow tests/slow   (only on an idle GPU; bfloat16)
"""

from __future__ import annotations

import os

import pytest
import torch

from raq_kv.budget.bit_accounting import full_precision_kv_budget
from raq_kv.config.loader import load_yaml
from raq_kv.config.schema import ModelConfig
from raq_kv.evaluation.protocol import prefill, score_through_cache
from raq_kv.models.loader import load_model
from raq_kv.utils.paths import configs_dir

pytestmark = pytest.mark.slow

TEXT = (
    "The KV cache stores the keys and values of every previous token so that autoregressive decoding does not "
    "recompute them. Its size grows linearly with the context length, the number of layers and the number of "
    "key-value heads. Invoice INV-48213 was issued on 2026-03-14 for 12.5 EUR to Acme Corporation. "
) * 8


@pytest.fixture(scope="module")
def lm():
    device = os.environ.get("RAQ_TEST_DEVICE", "cpu")
    base = load_yaml(configs_dir() / "models" / "qwen3_1p7b.yaml")["model"]
    dtype = "float32" if device == "cpu" else "bfloat16"
    return load_model(ModelConfig.model_validate({**base, "device": device, "dtype": dtype}))


@pytest.fixture(scope="module")
def ids(lm):
    return lm.tokenizer(TEXT, return_tensors="pt").input_ids[:, :300].to(lm.device)


def test_geometry_and_kv_bytes(lm, ids):
    g = lm.geometry
    assert (g.num_layers, g.num_q_heads, g.num_kv_heads, g.head_dim) == (28, 16, 8, 128)
    snap = prefill(lm.model, ids[:, :256])
    assert snap.nbytes == full_precision_kv_budget(g, 256).total_bytes


def test_two_phase_matches_single_pass(lm, ids):
    prefix, cont = 256, 32
    tol = 1e-4 if lm.dtype == torch.float32 else 5e-2
    with torch.no_grad():
        single = torch.log_softmax(lm.model(ids[:, : prefix + cont]).logits[0].float(), -1)[prefix:]
    res = score_through_cache(
        lm.model, prefill(lm.model, ids[:, :prefix]), ids[:, prefix : prefix + cont],
        ids[:, prefix + 1 : prefix + cont + 1], return_logprobs=True,
    )
    kl = (single.exp() * (single - res.logprobs)).sum(-1)
    assert torch.equal(single.argmax(-1), res.logprobs.argmax(-1))
    assert float(kl.max()) < tol, f"max KL {float(kl.max()):.3e}"


def test_middle_pruning_equals_masking(lm, ids):
    prefix, n = 256, 8
    pruned = [1, 17, 18, 100, 101, 102, 200, 255]
    keep = torch.tensor([i for i in range(prefix) if i not in pruned])
    full = prefill(lm.model, ids[:, :prefix])
    chunk = ids[:, prefix : prefix + n]
    pos = torch.arange(prefix, prefix + n, device=lm.device).unsqueeze(0)
    mask = torch.ones(1, prefix + n, dtype=torch.long, device=lm.device)
    mask[0, pruned] = 0
    with torch.no_grad():
        ref = lm.model(chunk, past_key_values=full.to_cache(), position_ids=pos, attention_mask=mask).logits.float()
        got = lm.model(chunk, past_key_values=full.select_tokens(keep).to_cache(), position_ids=pos).logits.float()
    tol = 1e-3 if lm.dtype == torch.float32 else 5e-2
    assert float((ref - got).abs().max()) < tol
