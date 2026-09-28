# D77 pre-registration — the two hooks D76 left, written before either runs

D76 closed with two named, cheap follow-ups. Both are run here, and the first one's **power was computed before the
run and it changes what the run can claim.**

## Part A — how tight can the g=256 quality bound get, and can its sign be resolved at all?

H2f's recommendation is conditional on one thing: the perplexity cost of moving the key group from 128 to 256. At 40
WikiText chunks that cost is **+0.00133 nats [−0.00193, +0.00452], p = 0.405** — a bound of ≤0.45% perplexity, not a
zero. D76 said resolving it was "one cheap measurement". **The power calculation says otherwise.**

From the existing 40-chunk arms, the paired difference has **sd = 0.010638**. So:

| goal | chunks needed |
|---|---|
| 95% half-width ≤ the observed effect (0.00133) | **245** |
| half-width ≤ 0.75× the effect | 436 |
| half-width ≤ 0.50× the effect | 980 |

**WikiText-2 raw test provides 145 non-overlapping chunks** (299,078 tokens at 2,049 per chunk, measured). At n = 145
the half-width is **0.00173**, still larger than the effect, so the interval would run about [−0.0004, +0.0031] and
**fail D31**. Therefore:

* **PRIMARY (A): this run cannot resolve the sign, and is not run to.** It is run to **tighten the bound** from
  ≤0.45% perplexity to ≈≤0.31%, on the same split, same chunk geometry, same pinned revision, so every number stays
  comparable to every other M1 perplexity figure in the project. The allocator decision is then made against a bound,
  which is the correct form for a "is the cost small enough" question and is what D30/D44 requires anyway.
* **Stated in advance:** if the interval at 145 chunks happens to exclude zero, that is a **surprise** relative to this
  calculation and would mean the 40-chunk sd under-estimated the effect; it will be reported as such rather than as a
  clean confirmation.
* **SECONDARY (A2), pre-registered with its cost:** the same four group sizes on the WikiText-2 **train** split, which
  provides enough chunks to resolve a 0.00133 effect. This **breaks comparability** with the project's other
  perplexity numbers — a different corpus partition, plausibly different difficulty — so it can establish the *sign* of
  the group-size effect while its absolute perplexities are not interchangeable with the test-split figures. Reported
  separately for that reason, never merged into the same table.

Four rungs (32, 128, 256, 512) rather than five, because those are the group sizes the existing arms cover and the
comparison is paired against them.

## Part B — does `group_channel` give α > 0, as D76's mechanism hypothesis predicts?

D76 found δ **span-independent** on the token axis: α = +0.024 [−0.110, +0.318] from the margin and +0.035 [−0.082,
+0.341] from the recorded `p₁`, both excluding quadrature (0.5) and coherent (1.0). Its closing hypothesis, stated as a
hypothesis:

> a `group_token` scale spans **tokens** at one channel of one head, and attention re-normalizes over tokens, so
> corrupting more positions may move the softmax's scale without moving its direction. That predicts `group_channel` —
> spanning channels **within** a token, which attention does **not** re-normalize over — should show α > 0.

**PRIMARY (B): α on the channel axis, same statistic and same cluster bootstrap as D76.** `group_channel` group sizes
**8, 16, 32, 64, 128** — a 16× range matching D76's, and `head_dim` is 128 so **every one divides it exactly**: unlike
the token axis there is no short trailing group, so the realised span equals the nominal one at every rung and the
regressor is clean by construction.

**Predicted: α > 0, and the interval should exclude 0.** The three named outcomes:

* **α > 0 with an interval excluding it** — the hypothesis is supported, the token/channel asymmetry is real, and the
  allocator gains a second, *different* lever: on the channel axis coarsening would cost reliability rather than being
  neutral.
* **α ≈ 0 again, interval containing zero** — the hypothesis is **refuted**, and span-independence is a property of
  metadata corruption generally rather than of the token axis. That is the more useful negative for the allocator,
  because it means granularity is never a reliability lever on either axis.
* **α < 0** — unmodelled; would need its own explanation and would not be reported as support for anything.

Everything else is D76's arm unchanged: Qwen3-1.7B eager on RAQ probe v0, 40 samples × 16 trials × 1 control, one
key-metadata flip per trial, `metadata_part: keys`, values `per_token` throughout, no protection, no pruning.

**One asymmetry to report rather than hide:** `group_channel` at group size g gives `L·H·T·(D/g)` key scales against
`group_token`'s `L·H·(T/g)·D`, so at small g the channel axis carries far more metadata and far more bits per value.
The rungs are therefore not iso-storage with D76's, and the two sweeps are compared on **α**, which is a slope within
each axis, not on their absolute δ or bit totals.

## Costs, from measured rates rather than guesses

`cache_ppl` on this model runs at **0.50 s/unit** and `protection_arm` (eager) at **~9 s/unit**. So Part A is
145 × 4 ≈ **5 minutes**, A2 about the same, and Part B is 40 × 17 × 5 ≈ **25 minutes**. Total under 40 minutes.

## Scope

One model throughout. Part A is a quality measurement on WikiText and Part B a reliability measurement on the probe —
the project's standard split, and the two are not combined into one table. Part B inherits D76's limits: 4 bits, keys
only, `median` δ so the tail is not described, and an α interval whose width is set by how noisy the small-span bins
are.
