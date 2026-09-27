# M6 exit checklist — agent benchmark and generalisation

**Status: M6 CLOSED on 2026-09-27.** Every deliverable in `docs/00_research_plan.md` §M6 is built and measured, and
the claim it supports — **C4** — is closed. Four of the six deliverables were unbuilt as of 2026-09-26 and were found
by auditing the plan against the repository rather than by any control noticing: **M0 through M5 all had exit
documents and M6 did not.** That is the same gap M2 had, and M2's gap hid an unmet criterion for four days
(`docs/m2_exit.md`). Writing this is what closed the others.

Headline: **the agent axis behaves unlike the retrieval axis in one specific way that matters for the allocator.**
On the probe, protection decisions descend from *what gets corrupted*; on the tool-call vehicle, every measured
failure descends from *what gets kept*. Across the three failing retention arms, **0 of 271 failures are argument
errors** — when compression drops the gold schema the model names a different tool, and it never names the right tool
and then mis-transcribes it. The same 4-bit quantization runs under all of them, so fidelity is not the lever there
and retention is.

**What M6 closing does NOT mean.** The agent work is **one model** on the task side. The BFCL arms are Qwen3-1.7B
only, the state probe is Qwen3-1.7B only, and the second family's M6 contribution is the perplexity half of C4 plus
the fault arms it already had. A third *family* remains unrun; D68 adds a third *model* on a scale axis inside one
family, which is a different and weaker kind of generalisation, and D68 itself found that the QK-Norm account those
family differences rested on does not survive a scale change cleanly.

Criteria, from `docs/00_research_plan.md` §M6. M6 closes when every row is ✅ with evidence.

| # | Deliverable | Status | Evidence |
|---|---|---|---|
| 1 | Tool-call evaluation: JSON parsing, function-name accuracy, type-aware argument matching, task success | ✅ | `evaluation/scoring.py::score_bfcl_ast` (D46): family-agnostic parsing that accepts `arguments`/`parameters`/`args`, tagged or bare JSON, BFCL's acceptable-value lists, and `""` as optional, reporting `named_gold_function` separately from `correct`. `score_tool_arg` is deliberately **frozen** (D46) so the older probe arms stay comparable. **D64 then used the split the field was built for:** across the three failing arms, **0 of 271** failures are argument errors, Wilson upper bounds 3.5–4.5%, while the uncompressed golden splits its own failures roughly evenly (8 of 15 with the right function named) |
| 2 | BFCL AST subsets | ✅ | `data/bfcl.py`. Measured **before** building: BFCL's own AST categories cache 157–333 tokens at the median against this project's 512–4096, so they cannot host a retention experiment. The vehicle is the scenario the benchmark leaves out — a tool **catalogue** of 10–25 real schemas drawn from BFCL's 769 distinct ones, which is what a real agent carries and what makes the cache long enough to compress. `cache_len` sits at the user-question boundary, so the catalogue is a shared prefix cache |
| 3 | Deterministic multi-turn state-carrying tasks | ✅ **D69** | BFCL cannot serve: `multi_turn_long_context` caches **157 tokens** at the median and its "long_context" comes from tool responses accumulated during execution, which needs the stateful harness row 8 records as declined. `data/state_probe.py` instead: document → correction → question, with the **superseded value registered as a scored distractor** so `mentions_distractor` names a reversion rather than flagging a generic error. Both assistant turns are fixed strings, which is what "deterministik" requires |
| 4 | Pass@1 (greedy) | ✅ | Every arm's primary outcome throughout, greedy decoding on a modified cache with `model.generate()` banned (D4) |
| 5 | **Pass^k** — success across all k independent fault realisations | ✅ **D65** | Absent from the repository until it was looked for, and computable with no GPU: every protection arm already draws `n_trials` independent plans per sample. Estimated as `C(s, k) / C(n, k)` averaged over samples, so all n trials are used rather than the first k. **Over eight faults an unprotected 4-bit cache returns a correct answer 13.6% of the time and the metadata-guarded arm 100%**, same budget to within +0.075% of storage. And failures **cluster**: the curve sits above the geometric `Pass^1`ᵏ everywhere (excess +0.0790 unprotected), so 13.6% is closer to "13.6% of inputs are immune" than to "every deployment survives 13.6% of the time" |
| 6 | Metric-mismatch analysis: rank correlation between token-level damage and task failure | ✅ **D66** | This is C4, and closing it needed both metrics on **one vehicle and one axis**. `cache_ppl` now accepts the BFCL catalogue — 769 schemas as JSON in deterministic order, tokenized as raw text with no chat template, chunked by WikiText's own `make_chunks` at WikiText's geometry so the two perplexities are comparable. At 2 bits perplexity moves **+13.41%** and tool-call accuracy **−23.81%** (80/120 against 105/120, Fisher **p = 1.9e-04**); at 4 bits both agree nothing happened (−0.03%, −0.95%, p = 1). The bf16 variant **reproduces `m6_bfcl_golden` exactly** at 105/120 |
| 7 | Generalisation: Qwen3-4B and Llama-3.2 | ✅ **D68** + the second-family work throughout | Llama-3.2-3B carries D47, D51, D52, D54, D55, D57, D58, D59, D61, D62 and C4's perplexity half. Qwen3-4B adds a **scale** axis inside one family: Q48's frozen *separate signals* branch now holds on **three** models (partial |ρ| = 0.282, 0.156, 0.047) and metadata/code damage is **2.845× / 3.534× / 4.446×**, three-for-three. Geometry differs and is honoured — 36 layers, 73,728 values per token against 57,344 — so every quantity compared is per-value or a within-model ratio, and layer-indexed protection maps do not transfer |
| 8 | Decision point: tau-bench-style user simulator | ✅ **declined, with the reason** | Not built. A simulated user makes the conversation **non-deterministic**, and every provenance rule in this project assumes two runs of one config build the same cache — `data_fingerprint` refuses an experiment whose data changed (`runner.py::_check_existing`). D69 gets multi-turn state with fixed assistant turns instead, which is what the plan's own word "deterministik" asks for. The cost is honest: no measurement here involves a user who reacts to the model |

## What M6 found that the plan did not anticipate

**The agent axis and the retrieval axis fail through different channels (D64).** Compression damage on the tool-call
vehicle is **entirely** tool selection — `m6_bfcl_joint` answers 102 of 105 against prefix-cache's 20, and the
difference is whether the gold schema survives, not how precisely it is stored. `m6_bfcl_magnitude` fails **105 of
105**, never selecting the gold schema on any sample, which is the strongest form yet of D47's correction of D42:
key magnitude is not a weak signal for retention, it is an anti-signal.

**Pass^k restates the project's central protection result far more sharply than any rate did (D65).** "The guard
removes 100% of measured task failure at BER 1e-6" is true and sounds like a small-count artefact. **0.1356 against
1.0000 over eight faults** is the same fact as a survival probability, which is the quantity a reliability budget is
about — and it is why the plan asked for this metric rather than another rate.

**C4 holds and the number that made it striking was inflated by the comparison it came from (D66).** D16's pairing
implied ≈2.4× (+29% perplexity against −71% accuracy) across two corpora and two vehicles; within one vehicle it is
**1.8×**. Part of D16's gap was the domain difference the objection to it supposed, and the effect survives removing
that part. The mechanism for the inflation is measured: the agent corpus is about **twice as robust** to precision
loss as general text (+13% against +27.56% at 2 bits) because JSON schemas are far more predictable (perplexity
2.1–2.4 against 9.5).

**The multi-turn vehicle's baseline is a finding on its own (D69).** With a perfect bf16 cache the model answers from
the **superseded** turn **43.06%** of the time, almost perfectly binary — 1 of 360 samples said anything other than
the new or the old value — and the rate ranges from **0%** on tool arguments to **95%** on invented entity names.
Under one metadata flip on the low-floor types, regression is **2 of 416** against 2.16% of answers breaking at all,
so **roughly a fifth of fault-induced failures are reversions and four fifths are not** — the opposite of the failure
anyone would predict of a corrupted conversation cache.

**The zero-point is the most dangerous word in a quantized KV cache (D68, corroborating D62).** Not one of **12,000**
code flips across three models drove the NLL non-finite; about **1%** of metadata flips did, and zero-points are the
worse half on all three (1.21% / 1.06% / 1.69% against scale's 0.52% / 0.47% / 0.00%). Every metadata/code ratio in
this project is therefore a **lower bound**, since those sites are excluded from the means and are the worst outcomes
there are.

## What M6 got wrong, and where it was fixed

1. **The state probe's question turn carried chat-template scaffolding (D69).** Its first baseline reported 44.4%
   regression with a plausible fact-type breakdown and was a **malformed prompt**: the conversation ended with two
   assistant headers. It surfaced only because `tool_arg` scored 0% correct *and* 0% regressed — a third failure mode
   no model explanation fits. **A bug that had degraded all six fact types uniformly would have passed.**
2. **The state probe's fault arm measured the single-turn probe (D69).** `ProtectionArm` and `FaultQA` called
   `build_raq_probe` directly rather than through the dispatcher, and `StateProbeConfig` is *deliberately* a superset
   of `RAQProbeConfig`'s field names — so the wrong builder **succeeded silently** and the arm reported zero state
   regressions measured on a vehicle with no correction turn. `build_raq_probe` now refuses a foreign config by type,
   and a test asserts that all three task classes use the dispatcher.
3. **C4's headline table printed a hundredfold error (D66).** `{value:+.2f}%` on a *fraction* rendered a 13.41%
   perplexity rise as "+0.13%", in the one table a reader of C4 would look at, while the lines below it were correct.
4. **`kv_magnitude.json` was one fixed path for model-specific data (D68).** Running `kv_stats.py` for a second model
   overwrote the first, after which `report_m2`'s partial correlation for the original campaign would merge the
   **wrong** magnitudes onto layers 0–27 and report a number with nothing announcing it. Exactly one direction was
   dangerous, which is the kind of asymmetry that goes unnoticed. Now per model and resolved by matching the file's
   own recorded `model`; **D18 reproduces exactly through the new path** (partial −0.2817 against the recorded −0.282).
5. **A `--set` override under a swept key was silently discarded (D67).** `--set task.n_samples=2` on a config whose
   sweep lists whole `task` dicts does nothing: `apply_overrides` runs first and `expand_sweep` rewrites the key. It
   cost 15 minutes believing a smoke run was a smoke run, and the output gives no sign because the run that happens
   is a perfectly valid one nobody asked for. Now refused by name.
6. **A config built by search-and-replace kept a false header (D68).** `d68_quant_sensitivity_4b` read "ON THE SECOND
   FAMILY" and "Identical KV geometry (verified: both 57344 values per token)" — both false for a 36-layer model with
   73,728, and neither touched by the replacement. **Copying a config copies its claims.**

## What M6 changed for the project's conclusions

**The allocator gains no new lever and loses two candidates.** D61 closes off attention-weighted metadata protection
— the highest-attention token group is the **least** damaging cell to corrupt on both families, so that refinement
would be worse than the uniform guard. D62 then located the cross-family fragility difference **outside the codec**
entirely: Qwen holds these answers at top-1 probability 1.0000 against Llama's 0.9880, and at matched value-space
exposure Qwen's median KL is flat across a **million-fold** perturbation range, which is what a saturated
distribution does and what no codec explanation produces.

**And one caution the project must carry into any cross-model claim.** "Did the answer change under fault" — the
outcome every arm here uses — plausibly **rewards overconfidence**. On the probe the more confident model is also the
more accurate one, so its lower hazard reads as competence; on a LongBench pilot Qwen is *less* accurate and *still*
far more confident, so it would win the reliability metric on the task it is worse at. That is a direction at n = 8,
not a result, and it is recorded as such — but no cross-model hazard number in this project should be reported
without calibration beside it.

## Carried forward from M6, still open

1. **A third model **family**.** D68 is a scale axis inside Qwen, and it weakened rather than strengthened the QK-Norm
   account: ρ attenuates −0.377 → −0.229 between two models that both apply QK-Norm, putting the larger one roughly
   halfway to Llama. If the architecture feature were the whole story the two Qwens would agree.
2. ~~**The agent axis on a second family.**~~ **Closed for the BFCL arms: D70.** H3's span finding replicates
   decisively on the agent vehicle (+0.7810 and +0.7571, overlapping intervals, both meeting the rule) and `key_norm`
   fails 69 of 69 on Llama as the config predicted. D64's *absolute* phrasing is narrowed: `prefixcache` on Llama
   gives 9 of 57 argument errors with an interval excluding zero, because Llama is argument-fragile at baseline (92.2%
   of its uncompressed failures have the right function and wrong arguments, against Qwen's 53.3%). **Still
   single-model: the state probe (D69) and C4's task half (D66).**
3. **Fault hazard on a hard task.** The saturation account (D62) is measured on a probe where Qwen sits at p = 1.0.
   A LongBench pilot desaturates it completely (top-1 0.9284, margin 2.875 against 16.5–19.5) but measures
   **confidence only** — whether the hazard *ordering* follows is unmeasured and is what would close D59's line. The
   vehicle is costly and constrained: eager attention OOMs at ~7k tokens on a 16 GB card, so the attention-scorer
   path is unavailable there at any length.
4. **The within-family margin dose-response.** D62's own stated limitation: thin-margin samples break about 2× more
   often in both families (4.41% vs 2.23%; 7.03% vs 3.75%) and neither correlation is resolvable. It needs many more
   samples at fixed model, no faults beyond what an arm already draws.
5. **Placement at an exposure where random data corruption hurts (D67).** No such exposure was found below 100
   bursts, and the linear DUE scaling suggests orders of magnitude more would be needed — at which point the arm
   stops modelling a rate anyone deploys against. A `layer_token`-scope comparison would test D21's prediction
   properly but targets whole rows, which is Q49's vehicle rather than H2b's.
6. **Multi-update and contested-update conversations (D69).** The correction always supersedes exactly one value.
   Longer histories, several updates and conflicting updates are all unbuilt.
