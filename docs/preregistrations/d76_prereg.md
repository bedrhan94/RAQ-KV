# D76 pre-registration — does the codec move δ? Written before the run

D62 tested four candidate mechanisms for the cross-family fragility gap — dynamic range, attention spread,
attention-rank-conditioned severity, per-flip severity — and none was a codec-side lever. That was the useful negative
for the allocator, and it left the allocator with nothing to choose *between* on reliability grounds.

**D75 produced a codec-side quantity where there was none.** Inverting the softmax-curvature identity gives **δ, the
effective logit perturbation** from one metadata flip — how far the fault moves the pre-softmax scores, with the
softmax's own amplification divided out. Unlike a KL it is comparable across configurations. It is a **model** property
across tasks (Qwen 0.2514 probe / 0.1901 LongBench, vehicle ratios not meeting D31) and differs **2–3×** across
families. What is entirely unmeasured is whether it differs across **codec settings**, which is the only version of the
question an allocator can act on.

## The quantity under test, and why the key group size is the right dial

Keys are quantized `group_token`, so a key scale is indexed by (head, token-group, channel) and governs exactly
**`group_size` values** — the g tokens of that group, at one channel of one head. So `group_size` *is* the number of
values one corrupted scale reaches, and it is a config field with a 16× usable range (32 to 512) on this vehicle.

## AMENDED BEFORE THE RUN, on geometry alone and with no outcome data seen

The original primary regressed ln(median δ) on ln(`group_size`) over five rungs. Inspecting the **span geometry** —
which is arithmetic on cache lengths, not an outcome — showed that the nominal group size is not what a corrupted
scale reaches, because the trailing group of a sequence is short and at coarse granularity it is not rare. At the
probe's median cache of 2052 tokens and g=512 there are 5 groups and the last holds **4** values; at its shortest
cache of 514 there are 2 groups and one holds **2**. So a uniform draw over key scales yields spans of 2 to 512, and
the *mean* realised span at g=512 is **261 on a 514-token cache and 413 on a 2052-token one**, against a nominal 512.

Two consequences, both acted on before the run:

1. `governed_values` is now recorded per trial, so the realised span of every flip is known rather than assumed.
2. **The primary becomes a regression on that per-flip span**, pooled across rungs, which spans 2–512 continuously and
   carries far more leverage than five points. The short groups become data instead of contamination.

Stated plainly because the timing matters: **no δ, KL, or break outcome had been computed from this arm when this
amendment was written** — the arm had not been run. The amendment rests on cache lengths and grid shapes alone. The
original per-rung statistic is retained below as a pre-registered secondary so the two readings can be compared, and
it remains meaningful because the **median** realised span equals the nominal group size exactly at every rung.

## PRIMARY — the exponent from the per-flip realised span

**α = slope of ln δ against ln(`governed_values`)**, over all trials pooled, with a bootstrap CI resampling *samples*
rather than trials so the interval respects the clustering. Report which of the three candidate values it excludes:

* **α ≈ 1 — coherent.** The g values a corrupted scale governs contribute additively; damage is proportional to span.
* **α ≈ 0.5 — incoherent.** Their errors add in quadrature, as independent perturbations do.
* **α ≈ 0 — span-independent.** Damage is set by something else: one dominant token, or the attention softmax
  saturating so that further corruption of the same group adds nothing.

## The allocator consequence, written down now so it is not constructed afterwards

Key metadata bits scale as **1/g** (fewer, larger groups need fewer scales). So at a fixed bit error rate the *number*
of key metadata flips also scales as 1/g, while damage per flip scales as g^α. **Total expected damage ∝ g^(α−1).**

* **α = 1 → group size is not a reliability lever at all.** The two effects cancel exactly, and the allocator should
  choose g on storage grounds alone.
* **α < 1 → larger groups are strictly better for reliability**, and they are also cheaper in bits. No trade-off: the
  allocator should push g up until quantization quality binds.
* **α > 1 → smaller groups are better for reliability** and cost more bits. A real trade-off, and the exponent sets its
  exchange rate.

All three are actionable and the first two are the ones that would change current practice, since the project has used
`group_token 128` everywhere since M1 without this having been measured.

## SECONDARY, all pre-registered

1. **The original per-rung statistic**: ln(median δ) against ln(`group_size`) over the five rungs. Retained so the
   amended and original readings are both on the record. It is not merely a weaker version — if the two disagree, the
   disagreement localises the effect to the short-span flips, which is itself informative.
2. **Break rate per rung** — the task-level outcome, which curvature cannot fake. At 40 × 16 and rates of 2–6% this is
   expected to be **underpowered** and will be reported as a bound, not as a null.
3. **Key SQNR per rung, as the confound.** Group size moves quantization fidelity as well as span. Existing arms
   already bound this on this exact model, axis and bit width: **29.67 dB at g=32 to 25.19 dB at g=512**, so a 16×
   change in span buys only 4.5 dB (≈1.7× in RMS error). It cannot mimic an α of 0.5 or 1, and it is reported beside
   the primary rather than argued away.
4. **Non-finite collapse per rung, split scale against zero-point.** D68 found the zero-point is the most dangerous
   word in the cache on three models; this is a fourth axis to check it on.
5. **δ computed two ways on the same trials.** This is the first arm to record `top1_prob_clean` and
   `nontop_mass_clean`, so δ can be recovered without the two-state reading D75 had to assume. Reporting the assumed
   and observed values together **measures how wrong D75's assumption was**, on real data, in the regime that matters.
6. **Actual bits per value per rung** (`avg_kv_bits` is recorded), so the result is a cost/benefit curve rather than an
   exponent in isolation.

## Design, and the guard that had to be added first

Qwen3-1.7B eager on RAQ probe v0 — the vehicle where the two-state form is **validated** (`d ln(1−p₁)/dm` = −0.994
[−1.029, −0.958] against a two-state −1.000) so the assumed δ path is sound. 40 samples × 16 trials × 1 control, one
flip per trial, no protection, no pruning — D62's arm with only the group size moving.

**The g=128 anchor, stated correctly.** D75's δ = 0.2514 came from `d62_qwen_meta`, whose draw was storage-wide and so
≈94% **value** scales on this geometry — it is not a keys-only number, and it cannot be filtered into one because the
part was never recorded per trial (`plan_composition` now records `by_part`, so the next arm has no such gap). It is
still a real check, through an arithmetic coincidence worth using: `head_dim` is 128 here, so a `per_token` value scale
governs 128 values and a `group_token` key scale at `group_size` 128 governs 128 values too. **If δ depends on the span
a corrupted scale reaches and not on which part it sits in, the g=128 rung must land near 0.2514.** If it does not,
that is a part effect — a finding, not a failed check.

**`metadata_part: keys` is required and is new.** A storage-wide metadata draw would make this sweep uninterpretable:
key scale count moves with 1/g while value scale count does not, so the *mixture* of parts a uniform draw lands on
shifts at every rung, and a value scale governs a different number of values than a key scale. Measured in
`tests/test_storage_faults.py` on a small fixture: the key share of a bit-uniform metadata draw is 0.500 at g=8 and
0.059 at g=128. Damage per flip would have moved for a reason unrelated to span. The field defaults to None so no
existing arm's population or identity changes.

**Exposure is held at exactly one key-metadata flip per trial, not at a fixed BER.** The 1/g scaling of flip *count* is
then applied analytically in the allocator consequence above. Running at fixed BER would move span and count together
and the exponent would not be recoverable from the result.

## Scope, stated in advance

One model, one vehicle, one bit width, keys only. The primary regresses on per-flip span, but it is fitted on the
**median** δ within each span bin, so it describes the typical flip and not the tail — and the tail is where D68 found scale-dependent behaviour the mean missed. Values are
`per_token` throughout, so nothing here speaks to value granularity. 40 samples is D62's size, chosen for comparability
rather than for power on the secondary. ETA about **35–40 minutes** at the measured 0.48 s/forward on this vehicle
(40 samples × 17 forwards × 5 rungs, plus per-rung load and quantization).
