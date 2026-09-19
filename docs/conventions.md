# Conventions

Binding conventions shared by every module (and, later, by the RTL golden vectors). Changing any of these
requires bumping `SCHEMA_VERSION` and recording the change in `decisions.md`.

## 1. KV tensor layout

- Per layer `l`: `keys[l]` and `values[l]` have shape `[B, H_kv, T, d_head]` (transformers 5.17 `DynamicLayer`).
- Keys are cached **after** RoPE (and, for Qwen3, after QK-Norm).
- The allocation head axis `h` is the **KV head** (`0 <= h < H_kv`), not the query head. With GQA, query heads
  `h * G .. h * G + G - 1` read KV head `h` (`G = H_q / H_kv`). Attention-derived importance must aggregate over
  those `G` query heads; the aggregation (mean or max) is stated in the importance module and in results.
- `B = 1` on every snapshot/mutation path. Batching is only allowed across variants of the *same* sample with the
  *same* retained length (e.g. several fault trials on one snapshot), which needs no padding.

## 2. Positions

- Token index `t` always means the **original** sequence position, also after pruning.
- `KVSnapshot.positions[i]` is the original position of physical cache slot `i` (strictly increasing).
- `KVSnapshot.next_position` is the original position of the next token to feed.
- **RoPE:** every forward on a snapshot-derived cache passes `position_ids` explicitly, starting at `next_position`.
- **Mask / query offset:** derived by transformers from the physical cache length (`get_mask_sizes`,
  `get_query_offset`). transformers 5.17 has no `cache_position` argument. Verified: middle pruning with explicit
  `position_ids` equals a full cache with a 2D attention mask hiding the pruned slots (tests/test_protocol_equivalence.py).
- **Never** call `model.generate()` on a snapshot-derived or modified cache: it re-derives positions from the
  compacted cache length and silently shifts RoPE. Use `evaluation.protocol.greedy_decode`.

## 3. Two-phase protocol

1. `prefill(context)` -> `KVSnapshot` (context logits are discarded).
2. Transform the snapshot (M0: identity; later quantize / encode / inject / decode / prune).
3. Read through the snapshot: `score_through_cache` (teacher forcing) or `greedy_decode`.

Golden and perturbed conditions start from the same phase-1 tensors. For cache-conditioned perplexity a chunk is
`prefix_len` cached tokens + `continuation_len` read inputs + 1, so every scored logit depends on the cache.

**Phase split for chat prompts (RAQ-Probe, GSM8K).** The full prompt is rendered with the chat template and tokenized
once; the split is a character offset moved left to a token boundary (a straddling token belongs to phase 2).
- Phase 1 (cache): chat-template prefix (incl. tool schema if any) + intro + document / problem statement.
- Phase 2 (read): question or instruction + assistant header; then generated tokens.
- **Known confounds:** template tokens always occupy the first cache positions (attention-sink region), and
  `tool_arg` prompts have a ~150-token longer tool-schema prefix, which shifts absolute positions of their documents.
  Analyses by token role must control for absolute and relative position (`meta.depth_actual`, span token indices).
- Generated tokens' own KV is not transformed in M0-M2 (only the phase-1 cache is).

**Spans.** `[token_start, token_end)` covers every token overlapping the span's characters, so edge tokens may include
neighbouring characters (e.g. `.ĊĊ`). Span-to-token mapping is verified at build time by decoding.

## 4. What is quantized (scope)

- **Only the KV cache.** Weights, activations and the attention computation stay in the model's dtype (bf16 on
  GPU). A configuration labelled "4-bit" here means 4-bit cached keys/values with bf16 weights — it is *not* a
  4-bit model, and its numbers are not comparable with weight-quantization results.
- **Keys are quantized after RoPE** (and after Qwen3's QK-Norm), because that is what the cache stores. Papers
  that quantize keys *before* RoPE see different key statistics; comparisons must say which side of RoPE they are on.
- **Offline vs online schemes.** `per_channel` and `per_tensor` compute scales over every token in the context, so
  they are prefill-time (offline) schemes: a streaming decoder cannot form those statistics. `per_token` and
  `group_token` are causal — a scale depends only on tokens that have already arrived. The distinction is enforced
  by a test (`test_group_token_is_causal_and_per_channel_is_not`) and must be stated whenever an operating point
  is quoted.
- **Fake quantization.** Codes are reconstructed into the model dtype before attention, so the tensors in VRAM do
  not shrink; memory is reported from the bit budget (D6).

## 5. Bits and bytes

- Bit indexing inside an integer code: **LSB = bit 0**, MSB = bit `b-1` (fixed now; used from M1/M2).
- Memory numbers come from `budget/bit_accounting.py`. Bits are integers; bytes = `ceil(bits / 8)`.
- Units: KiB = 1024 B, MiB = 1024 KiB. Never mix decimal MB into tables.

## 6. Identity and records

- `exp_id = <slug>-<sha256(identity)[:8]>`; the identity excludes `output` and `notes`.
- `unit_id = <exp_id>::<unit_key>`; `unit_seed = derive_seed(seed, unit_key)`.
- `outputs/raw/<exp_id>/results.jsonl` is append-only; summaries are regenerated from it.
