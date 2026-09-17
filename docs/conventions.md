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

## 4. Bits and bytes

- Bit indexing inside an integer code: **LSB = bit 0**, MSB = bit `b-1` (fixed now; used from M1/M2).
- Memory numbers come from `budget/bit_accounting.py`. Bits are integers; bytes = `ceil(bits / 8)`.
- Units: KiB = 1024 B, MiB = 1024 KiB. Never mix decimal MB into tables.

## 5. Identity and records

- `exp_id = <slug>-<sha256(identity)[:8]>`; the identity excludes `output` and `notes`.
- `unit_id = <exp_id>::<unit_key>`; `unit_seed = derive_seed(seed, unit_key)`.
- `outputs/raw/<exp_id>/results.jsonl` is append-only; summaries are regenerated from it.
