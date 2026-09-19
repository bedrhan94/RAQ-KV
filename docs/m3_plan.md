# M3 plan — error protection, designed around one comparison

M2 produced three findings that all point the same way: damage follows **how much data a bit governs**, not how
semantically important it is (D17–D21). The tempting M3 is "implement parity, SECDED and BCH, protect the
metadata, declare success". That would demonstrate engineering, not the project's claim.

> **Revision 2026-09-20.** The first version of this plan named the decisive pair as *4-bit + uniform SECDED*
> versus *2-bit + metadata-only SECDED* "at ±1% equal budget". Costing it showed the pair differs by **107%**
> (D22): SECDED costs 12.5% of the data bits while one step of the bit-width ladder costs 100%, so the exchange
> rate is ~8:1 and no protection policy can pay for a bit-width step. The arms below are rebuilt on the axis that
> *is* exchangeable with ECC — **quantization group size** — and the fault model is rebuilt over the arm's whole
> declared storage, check bits included (D23).

## The experiment M3 exists to run

At a **fixed effective bit budget** (4.86 bits per cached value, ±0.18% at every probe length) and a **fixed bit
error rate over each arm's own storage**, compare four ways of spending the same bits:

| Arm | Key groups | Data ECC | Metadata ECC | bits/value @2066 | Why it is here |
|---|---|---|---|---|---|
| **B** | 128 | SECDED, **all 28 layers** | SECDED | 4.8529 | the textbook baseline: protect everything, equally |
| **C** | **32** | **none** | SECDED | 4.8640 | spends every one of B's data check bits on precision instead |
| **D** | 64 | SECDED, **layers 9–27** | SECDED | 4.8626 | half to precision, half to protection where D21 found damage |
| **E** | 64 | SECDED, **layers 0–18** | SECDED | 4.8626 | **bit-identical to D**, protection placed where D21 found none |
| A | 128 | none | none | 4.2604 (−12.3%) | cost reference, *not* iso-budget: the M1/M2 operating point |
| G | 128 | none | SECDED | 4.3581 (−10.3%) | cost reference, added after A–E: how little does the 98% cost? |
| H | 128 | none | **range check, no code** | 4.2605 (**+0.075%** over A) | cost reference, added after D27: is the fix a code at all? |

`scripts/check_iso_budget.py` costs the configs before any of them runs and exits non-zero if the set is not
within tolerance; its arithmetic is pinned bit-for-bit against the real quantizer and codec in
`tests/test_iso_budget.py`. Arm A is labelled a cost reference everywhere, never an equal-budget competitor.

**The three questions, and what each answer means**

1. **B vs C — does protection beat precision at equal bits?** C has *no data ECC at all*. If C matches or beats B,
   uniform data ECC is dominated by simply quantizing better, and the reliability contribution reduces to
   "protect the metadata, it is nearly free". If B beats C, protection earns its bits and the classical answer is
   right — a negative result for the adaptive thesis, and a clean one.
2. **D vs E — does the fault-sensitivity map have predictive value?** Same storage, same amount of ECC, different
   layers. This is the sharpest falsification test in the project: if D ≈ E, the map that M2 spent its campaign
   building does not predict where protection should go, and H2's stated mechanism is refuted even if D happens
   to beat B.
3. **A vs the rest — what does 12.3% of the budget buy?** The cost-benefit framing, reported separately from
   every equal-budget claim.

## The fault model (D23)

A trial is a **bit error rate applied to everything the arm declares it stores**: codes, quantization metadata,
its ECC check bits, and — for the range-guarded arm — the guard's own stored bounds. Check-bit flips are not a
curiosity: they make the decoder compute a syndrome for an error that did not happen and "repair" intact data.
Bound flips are not either: a corrupted bound narrows the range and the guard clamps hundreds of valid scales
(D29). Sparing either would hand the protected arms an immunity no physical memory grants, and the equal-budget
claim rests on exactly that being honest.

Two physical regimes matter, and both are already reachable without new code (measured occupancy, D23):

| regime | generator | words with >1 flip | SECDED |
|---|---|---|---|
| uniform SEU | `protection_arm` task, rate over storage | ~0% up to 2048 flips | corrects everything |
| word/row failure | `fault_qa` with `burst_scope`, 64–2048 b | 36% → 100% | partly → not at all |

So ECC's value is a statement about **spatial correlation**, not about bit counts, and the crossover rate — where
multi-bit-per-word coincidences start defeating the code — is the quantity the M5 allocator actually needs.

**Rate calibration comes first.** `configs/experiments/m3_rate_calibration.yaml` sweeps four decades on arm A
alone (320 decodes). The decade where damage reaches ~50% sets the campaign's rates, because at D21's 4% base
rate detecting a halving needs ~1113 trials per arm while at 35–50% it needs 58–99. The rate is a reported
condition, never a claim; what is compared is the ordering of arms at a fixed rate, and the rate at which that
ordering changes.

## Measurement rules

- **Replay is scoped.** A plan's (layer, part, index, bit) addresses denote the same physical bits only within one
  layout, so plans are replayed for *protected vs unprotected at the same quantization* and **not** across arms
  with different group sizes or protection maps (D23 supersedes the unqualified M2 rule). Across arms the match is
  on rate, with the trial count set by power.
- **SDC, DUE and masked are counted separately.** Each trial records `answer_changed` and `detected`
  independently: a changed answer with no detection is silent corruption; a changed answer with a DUE is a failure
  the system could have caught and recomputed (at a latency cost this project does not measure). Conflating them
  is how reliability papers overstate their case.
- **The golden answer is generated through the arm's own read path**, so protected and unprotected arms differ in
  faults and nothing else. A fault-free protected read that reports any correction raises rather than being
  logged (`ProtectedKV.read_or_raise`).
- **No-fault controls in every arm**, exactly zero by construction.
- **Clean quality is part of the comparison, not a footnote.** C quantizes keys 4× finer than B, so its fault-free
  accuracy should be *better*; the arms are compared on the joint (clean quality, damage under faults) outcome,
  because an arm that is more robust only because it was already worse has won nothing.
- Effect sizes below the bf16 noise floor (D10) are not attributable.

## Status of the build

Done: codecs (`none`/`parity`/`SECDED`, exhaustively tested), `ProtectedKV` with layer-selective and
metadata-independent policies and an explicit DUE erasure policy, ECC bits in `BitBudget.ecc_bits`, the
storage-wide fault model, the `protection_arm` task, the five arm configs, the rate sweep, and the iso-budget
checker.

Run and reported (see `docs/m3_exit.md`): the rate calibration, the six arms at BER 1e-6, arms D/E and G/H at
1e-5. Open: the long-context placement test, and ECC golden vectors for later RTL (plan §39).
