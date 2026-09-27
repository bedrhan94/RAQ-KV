# D75 pre-registration — the graded KL as the outcome, written before looking

D71 and D74 both end with the same named escape. D71's Scope: *"a design measuring fragility per sample more
finely -- more trials, or the graded KL as the outcome instead of the rate -- could resolve a smaller effect. The KL
route is the obvious one and is not run here, because D62 already established that Qwen's KL is saturation-limited on
this probe and would be measuring the floor."* D74's Scope: *"a finer per-sample fragility measure -- the graded KL as
the outcome instead of the rate -- could resolve a smaller effect, and on this vehicle Qwen is no longer
saturation-limited so that route is now open. It is not run."*

It needs no GPU. `d74_margin_curve_llama_lb` has `record_perturbation: true`, so every one of its 80 x 16 trials
already carries `kl_first_token`. This is a re-analysis of existing records with a different outcome variable.

## Why the outcome change is the whole point

A per-sample break rate over 16 trials takes 17 possible values and in practice takes **six** (D74: 51 samples at
exactly 0, then 0.0625, 0.125, 0.1875, 0.25, 0.5625). 75 of 80 samples are tied. A rank correlation against an
outcome that is 64% a single tied value is a blunt instrument, and that bluntness is exactly what D71 and D74 both
named as the limit of their bounds. The per-sample **median KL over its trials** is continuous, has no ties, and uses
every trial rather than only the ones that crossed the answer boundary.

## PRIMARY, and the direction is predicted rather than read off

**Spearman(`margin_clean`, per-sample median `kl_first_token`) on `d74_margin_curve_llama_lb`, condition `stable`,
n = 80.** D31 applies: p < 0.05 **and** a bootstrap CI excluding zero.

**Predicted sign: NEGATIVE.** This is D62's saturation account stated at the sample level. A saturated output
distribution has little mass to move, so the same exposure should shift it less; a thin margin means a less peaked
distribution and more mass in play, so the same flip should produce a **larger** KL. If saturation is the mechanism
and it operates per sample, thinner margin must mean larger KL.

The vehicle is chosen on the same measurement D74 used: Llama-3.2-3B on LongBench `2wikimqa` spans 0.250-6.500 nats
with 75% below 3, so it places one model across the region where the margin is contestable. And it is desaturated
there, which is what closes D71's stated objection to running this test on the probe.

## What each outcome means, both written down now

* **A resolved negative rho** — the margin is not irrelevant after all. It governs *how far a flip moves the
  distribution* while carrying no information about *which samples break* (D71, D74). That is a real and reportable
  dissociation, and it would mean D71's and D74's nulls are about the coarseness of the rate, not about the margin.
* **A null with a tight CI** — the margin's sample-level irrelevance is established against the most sensitive
  measure this project has, not only against a coarse rate. The escape D71 and D74 both left open is then closed on
  its own terms, and it is closed on the vehicle those records named.

## SECONDARY, pre-registered so it is not chosen afterwards

1. **The same statistic with the mean and the max KL in place of the median**, reported whatever the primary does. The
   median is the primary because it describes the typical trial; the max is what a tail-risk reading would want. If
   they disagree, the disagreement is the finding and neither is promoted after the fact.
2. **The reverse direction, already measured before this file was written and reported as such:** per-sample median KL
   against per-sample break rate. d74 rho = +0.088 [-0.128, +0.299] p = 0.44; d71 rho = +0.026 [-0.118, +0.172]
   p = 0.72. Both null. This asks whether a sample's typical perturbation response predicts its fragility, and is a
   different question from the primary.
3. **`d73_lb_fault_qwen` and `d62_*` are reported for consistency and carry no weight.** d73 has one sample with a
   nonzero rate out of 20; its margin CI [-0.552, -0.180] excludes zero at p = 0.27, which is a degenerate-outcome
   artefact and a live demonstration of why D31 needs both prongs.

## The confound to state rather than discover

The margin and the KL are both computed from the same clean distribution, so a correlation between them is partly a
statement about that distribution's shape. It is not definitional — a one-hot clean distribution can still produce a
large KL, since the KL is dominated by `p_clean(top) * (logp_clean(top) - logp_faulty(top))` and the faulty term is
free to move — but the two are not independent measurements either. So a positive primary result supports "the clean
distribution's peakedness governs perturbation magnitude" and **not** "this answer's distance from its rival governs
it". The rate-based tests in D71 and D74 are the ones with a task-level outcome, and they are null.
