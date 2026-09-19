# M3 plan — error protection, designed around one comparison

M2 produced three findings that all point the same way: damage follows **how much data a bit governs**, not how
semantically important it is (D17-D21). The tempting M3 is "implement parity, SECDED and BCH, protect the
metadata, declare success". That would demonstrate engineering, not the project's claim.

## The experiment M3 exists to run

At a **fixed total bit budget** and a **fixed, replayed fault plan**, compare:

| Arm | Data precision | Protection | Why it is here |
|---|---|---|---|
| A | 4-bit block-128 | none | the M1/M2 reference point |
| B | 4-bit block-128 | uniform SECDED over everything | the textbook baseline |
| C | **2-bit block-32** | **SECDED on quantization metadata only** | spends the freed data bits differently |
| D | 4-bit block-128 | SECDED on metadata only | the cheap-protection arm |
| E | 4-bit block-128 | SECDED on early-layer data (D18's fault-sensitive region) | tests the depth signal |

Arms B and C must land within ±1% of the same total bits (`budget.bit_accounting` already counts data,
quantization metadata and ECC as separate terms, so the comparison is enforced by construction, not by hand).

**What each outcome means**
- C beats B → the thesis holds: reallocating protection bits by what they govern beats uniform protection.
- B beats C → uniform protection at higher precision wins; the adaptive story needs rethinking, and that is a
  publishable negative result given how cheaply metadata protection appeared to work in M2.
- D ≈ B at far lower cost → protection is nearly free if aimed correctly, which is the practical headline.
- E tells us whether the fault-sensitivity map earns its keep beyond the metadata/data split.

## What has to be built (in this order)

1. `ecc/` — `none`, `parity`, `hamming_secded` over configurable word sizes, with exhaustive unit tests on small
   words (every 1-bit error corrected, every 2-bit error detected) and golden vectors exported for later RTL.
2. Protection policy: a mapping from (target class, layer, part) to a codec, expressed in config so an arm is a
   config, not code.
3. Decode-time behaviour: a detected-uncorrectable error must do something explicit. Default: treat the word as
   an erasure and fall back to the group's scale (documented), never silently pass corrupted bits.
4. Accounting: ECC bits enter `BitBudget.ecc_bits`; the runner refuses arms whose budgets differ by more than 1%.

## Measurement rules inherited from M0-M2

- The fault plan is generated once per unit and replayed across arms, so arms see identical faults (M2 already
  saves plans under `outputs/fault_plans/`).
- Report SDC (wrong answer, no detection), DUE (detected, uncorrectable) and masked separately — a detected
  error is not a silent corruption and must not be counted as one.
- No-fault controls in every arm; they must be exactly zero.
- Task-level outcomes on the probe, with the operating point chosen so the model is not already broken: 4-bit
  block-128 (147/150 correct) rather than 2-bit (109/120).
- Effect sizes below the bf16 noise floor (D10) are not attributable.
