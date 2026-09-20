# Literature review (2026-09-20) — closes hakem condition 4, and changes M4/M5

Searched 2026-09-20 for the three axes separately and for their intersection. What follows is what the retrieved
work actually claims, what it does *not* do, and the consequences for this project. Two of those consequences
retract things written in this repo.

## 1. KV cache quantization

**KIVI (ICML 2024)** — tuning-free asymmetric 2-bit: **keys per-channel (group 32), values per-token**, and
**the most recent 128 tokens kept in full precision** (a residual window). ~1 point loss on CoQA/GSM8K for
Llama-2-7B and Mistral-7B, 2.6x peak memory reduction, 2.35-3.47x throughput.

> **Consequence (baseline fairness).** Our configs set `residual_window: 0`. A KIVI comparison without its residual
> window is **not faithful** and would be a weakened baseline — precisely the failure mode the hakem protocol calls
> a red line. The `residual_window` field already exists in `QuantizationConfig` and is unused; the M5 baseline
> table must use `residual_window: 128` with keys per-channel group 32, i.e. KIVI's own optimum, not our defaults.
> Note our D15 finding (causal `group_token` beats offline `per_channel` at equal bits) was measured *without* a
> residual window, so it does not transfer automatically to the faithful configuration and must be re-measured
> there before being used to argue against KIVI's axis choice.

Related: RotateKV (outlier-aware rotations, 2-bit), Kitty (dynamic channel-wise precision boost), KVarN
(variance-normalised, error accumulation in reasoning), MixKVQ (query-aware mixed precision), KV-AdaQuant /
"Quantize What Counts" (more bits for keys than values — consistent with our D13 keys-11x-values result),
Ada-KV (adaptive eviction budget per head), "Beyond Token Eviction" (mixed-dimension budget allocation).

## 2. Token retention versus precision — already answered; retract our claim of novelty

**"More Tokens, Lower Precision" (arXiv 2412.12706)** compares, at equal memory, 1x tokens @16-bit vs 2x @8-bit vs
**4x @4-bit** on LongBench, Needle-in-a-Haystack and RULER (Llama-3-8B-Instruct, Mistral-7B-Instruct-v0.2,
Llama-3.2). **Finding: more tokens at lower precision wins**, with the largest gains on retrieval-heavy tasks and
smaller gains on reasoning-heavy ones. They state they did not explore head/layer dimensions, and they make **no
mention of ECC, fault tolerance, or quantization metadata**.

> **Consequence (retraction).** `m4_m5_plan_draft.md` claims the pairing "all tokens at 4 bits versus half the
> tokens at 8 bits" is untested and that nobody answers it directly. **That is wrong and must be deleted.** M4 must
> not re-derive their result. Instead it should **adopt their optimum as the operating point** and ask the question
> they did not: their recommendation pushes systems toward low precision with many tokens, which is exactly the
> regime where quantization metadata grows to 6-13% of stored bits — and, by our D17/D21/D23 measurements, becomes
> the concentrated fault surface. Their gains are largest on retrieval tasks, which is also where our D16 shows
> perplexity understates damage, so their frontier needs a reliability axis before it can be trusted in deployment.

## 3. KV cache faults, soft errors and protection

**"Bit-Flip Vulnerability of Shared KV-Cache Blocks in LLM Serving Systems" (arXiv 2604.17249)** — Rowhammer
adversary against vLLM prefix-cache blocks, which exist as a single unprotected physical copy. Fault model is
**single-bit flips in BF16**; 13 of 16 BF16 bit positions yield coherent but altered outputs. Three properties:
silent divergence, selective propagation, persistent accumulation. Countermeasure: **a checksum that detects any
single-bit corruption at scheduling time**, bounding damage to one batch.

**GPU soft-error / dependability studies** — an instruction-level fault injection study (arXiv 2601.19912) with
24,000 trials per model, reporting vulnerability as MVF = SDC rate + DUE rate, which is the same split we use.
Note that it **explicitly assumes faults do not occur in the KV cache**, so it says nothing about our object.

**A threshold we could not verify, and therefore do not cite.** A search summary attributed to this literature
the statement that "for KV cache error rates below 1e-3 the increase in perplexity remains minimal". The
instruction-level study grouped with it cannot be the source (see above), and the dependability chapter that
might be is paywalled; a targeted follow-up search found no accessible primary source stating it. **Any
comparison of the form "the literature says 1e-3 is safe, we measure damage at 1e-6" is struck** until the
rate's definition is read in the original, because the denominator is exactly what would make or break it.

**What is verified, and is the better engagement point anyway.** The readable literature supports two
qualitative claims: the KV cache is the error-tolerant part of the model, because attention softmax masks small
perturbations; and within quantization error **the largest perturbations dominate** — "the largest errors (e.g.
top 5%) in the KV-cache cause most of the end-to-end degradation, while smaller errors (even if they are many)
are less important."

The second is the same mechanism our D25/D27/D29 measurements found, reached from the other direction. So the
positioning is an extension rather than a contradiction:

> Prior work establishes that the KV cache tolerates small perturbations because softmax masks them, and that
> within quantization error the largest perturbations dominate. We show that **compressing the cache manufactures
> a small population of bits that can produce exactly those largest perturbations** — the quantization scales and
> zero points, three of whose sixteen bits rescale a whole group by orders of magnitude — and that this
> population, which does not exist in the unquantized caches those studies examine, carries 98% of the
> fault-induced task damage at 6% of the flips. The cache's celebrated error tolerance is a property of its
> *data*; it does not extend to the metadata that compression adds, and 0.075% of extra storage restores it.

That rests only on our own measurements plus the two verified qualitative claims, and needs no unverified number.

## 4. What is left for us — the novelty statement, corrected

Nothing retrieved does any of the following, and together they are the contribution:

1. **Pricing protection against precision at a fixed bit budget.** The quantization literature never counts ECC; the
   reliability literature never counts the bits its protection costs. D22's ~8:1 exchange rate (SECDED = 12.5% of
   data bits, a bit-width step = 100%) exists in neither.
2. **Quantization metadata as the failure surface that compression creates.** Prior fault work targets BF16 KV,
   where no scales or zero-points exist; the vulnerability we measure is *produced by* compressing the cache and is
   invisible in their setting. A checksum over BF16 blocks cannot express "protect the scales, not the codes".
3. **Exposing the protection's own bits to the fault model** (D23). Prior protection proposals are not charged for
   their check bits, nor are those bits corruptible in their evaluations.
4. **Correction with an accounted budget, not detection alone.** The serving-systems countermeasure detects; we
   compare detect-only against correct-and-erase at matched cost, and report SDC and DUE separately.

> **Consequence (new arm).** The literature's proposed defense is a *checksum*, i.e. detection without correction.
> Our `ParityCodec` is its analogue at 1/64 = 1.6% overhead versus SECDED's 12.5%. An arm F (parity on data,
> metadata SECDED) belongs in the comparison so the cheap prior-art defense is represented — and it must be run
> under `due_policy: pass_through` for the damage measurement, because `erasure_mid` on a detected single-bit error
> replaces one wrong code with a whole wrong word and would make detection look *worse* than doing nothing. The
> honest framing is that detection buys awareness (recompute, evict) rather than correctness, and awareness is what
> the serving paper's "bound damage to one batch" actually delivers.

## 5. Threat-model separation, to state explicitly in the paper

Prior KV bit-flip work is **adversarial** (Rowhammer, targeted bit, worst case). Ours is **uniform random soft
error** at a stated rate over declared storage. Neither subsumes the other: an adversary picks the metadata bit our
measurements say is worst, so our uniform-rate numbers are a *lower bound* on adversarial damage and must never be
presented as a security result. Conversely, a targeted study cannot produce a rate-based allocation rule.

## Sources

- KIVI: https://proceedings.mlr.press/v235/liu24bz.html and https://github.com/jy-yuan/KIVI
- More Tokens, Lower Precision: https://arxiv.org/html/2412.12706v1
- Bit-Flip Vulnerability of Shared KV-Cache Blocks: https://arxiv.org/abs/2604.17249
- GPU soft-error instruction-level study: https://arxiv.org/html/2601.19912
- Dependability of LLM parameters/variables to memory soft errors: https://link.springer.com/chapter/10.1007/978-3-032-17948-7_4
- RotateKV: https://arxiv.org/pdf/2501.16383 ; Kitty: https://arxiv.org/pdf/2511.18643
- Quantize What Counts / KV-AdaQuant: https://arxiv.org/pdf/2502.15075
- MixKVQ: https://arxiv.org/pdf/2512.19206 ; Beyond Token Eviction: https://arxiv.org/pdf/2603.20616
