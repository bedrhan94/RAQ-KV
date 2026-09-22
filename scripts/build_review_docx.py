"""Builds the RAQ-KV technical review document (.docx) for an external reviewer.

Regenerate with:  python scripts/build_review_docx.py
Content is assembled from docs/decisions.md (D1-D43), docs/claims.md and the measured campaign outputs; the
numbers here are transcribed from those records, and the record is the source of truth if they ever disagree.
"""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor

ACCENT = RGBColor(0x1F, 0x4E, 0x79)
ACCENT2 = RGBColor(0x2E, 0x6C, 0xA4)
GREY = RGBColor(0x66, 0x66, 0x66)
MONO = "Consolas"

# Section 10's table. Module-level so the prose above it can count the rows rather than restate a
# number that then drifts -- which it did: the intro said eight while the table held eleven.
WITHDRAWN_ROWS = [
    ["The multiplicative Joint Reliability Score (the brief's central mechanism)",
     "Axes not commensurate (H2c); the two signals it multiplies are anti-correlated (Q48, rho = -0.377); "
     "protection is a lookup; decisions are over spans, not (layer, token) cells",
     "A staged rule with one measured switching point"],
    ["'2-bit is degraded but functional' (D12)",
     "Rested on perplexity alone; probe accuracy collapses at the same setting (D16)",
     "Task failure, not perplexity, is the metric for every later claim"],
    ["Metadata is 315x more damaging per bit than codes (D25)",
     "The code-flip denominator was not a measured rate",
     "A bound: at least 1543x"],
    ["Metadata is 762x more damaging per bit than codes",
     "The per-code-flip rate was three events in one non-shared sample - noise (D28)",
     "The same bound: at most 6.5e-6 per code flip against about 1.0e-2 per metadata bit"],
    ["ECC placement matters: late layers beat early by 1.73x",
     "A clustering artefact. Arm D's 12 events came from 3 samples, arm E's 7 from 2; pooled-trial Fisher "
     "treated them as hundreds of independent observations (D26)",
     "Sample-level pairing everywhere; the direction is reported below significance (D31)"],
    ["Protection fails through its own erasure policy (OR 4.79, p = 0.012)",
     "Simpson's paradox. Long samples have 4x the flips and all of the damage; inside the only band with "
     "events the OR falls to 1.66 (D28)",
     "Context length is a reported stratum in every fault analysis"],
    ["Arms B and C are equal; arms D and E are equal",
     "Both pairs recorded zero failures, so the paired test had zero discordant samples",
     "Upper bounds (about 0.6% on 472 trials) instead of equalities"],
    ["The range check matches metadata SECDED",
     "True at BER 1e-6 only; at 1e-5 it loses by +0.0593, p = 1.1e-5 (D29/D30)",
     "A rate-conditioned lookup with the crossover located"],
    ["oracle_keep_sentence is a ceiling (D34/D35)",
     "It spends a fifth of its budget by recency regardless of relevance; a real method beats it by 13 "
     "points (D36)",
     "It is 'the best span policy tried', and every use of the word ceiling is withdrawn"],
    ["Token-level importance scoring is actively harmful",
     "Holds for query-agnostic scoring only; query-aware token scoring is not harmful (D36)",
     "The claim is scoped to query-agnostic scoring"],
    ["Four protection pilot runs",
     "Source was edited while the sweep ran, so three manifests recorded three different diff hashes (D24)",
     "The runner now refuses an experiment whose code changed mid-process; the four runs were retired, not "
     "deleted"],
    ["'H3 is supported' (D41, stated one day before this revision)",
     "A pre-registration audit, not new data. The plan requires two model families and names a magnitude-only "
     "baseline; the campaign had one family and had not run that arm (D42)",
     "Supported on one model family against the baselines actually run, with the unmet conditions named; the "
     "general claim is not made"],
]

# H3, transcribed from `scripts/report_m4.py --arms m5h3_*` over the eight arms that ran. Kept as data next to
# the prose so a reviewer can diff it against the report output rather than trusting the paragraph around it.
# The retention-only rows are the calibrated re-runs (D42); the originals spent 2.8% and 1.45% less.
H3_ROWS = [
    ["joint @ 1.05 (4-bit, query-aware spans)", "24.67%", "1.0401", "1.0000", "0.0342"],
    ["retention-only @ 1.05 (bf16, query-aware spans)", "6.76%", "1.0409 (+0.07%)", "0.9487", "0.1624"],
    ["magnitude-only @ 1.05 (4-bit, key-norm tokens)", "24.67%", "1.0539 (+1.33%)", "0.2128", "0.9573"],
    ["recency-retention @ 1.05 (4-bit, recency tokens)", "24.67%", "1.0539 (+1.33%)", "0.3162", "0.6923"],
    ["joint @ 2.00 (4-bit, query-aware spans)", "46.98%", "1.9849", "1.0000", "0.0000"],
    ["retention-only @ 2.00 (bf16, query-aware spans)", "12.71%", "1.9882 (+0.17%)", "1.0000", "0.0513"],
    ["magnitude-only @ 2.00 (4-bit, key-norm tokens)", "46.98%", "2.0011 (+0.82%)", "0.4984", "0.8376"],
    ["recency-retention @ 2.00 (4-bit, recency tokens)", "46.98%", "2.0011 (+0.82%)", "0.3712", "0.6838"],
    ["precision-only, either budget", "-", "floor 2.63: infeasible", "-", "not emitted"],
]
H3_PAIRS = [
    "                                    difference                  p         Holm     discordant",
    "1.05  vs retention-only      -0.1282 [-0.1966, -0.0684]   2.75e-04   5.5e-04      17 of 117",
    "1.05  vs magnitude-only      -0.9231 [-0.9658, -0.8718]   2.69e-25   1.6e-24     108 of 117",
    "1.05  vs recency-retention   -0.6581 [-0.7436, -0.5726]   1.71e-18   5.1e-18      77 of 117",
    "2.00  vs retention-only      -0.0513 [-0.0940, -0.0171]   1.43e-02   1.4e-02       6 of 117",
    "2.00  vs magnitude-only      -0.8376 [-0.8974, -0.7692]   4.18e-23   2.1e-22      98 of 117",
    "2.00  vs recency-retention   -0.6838 [-0.7607, -0.5983]   3.74e-19   1.5e-18      80 of 117",
]


def setup(doc: Document) -> None:
    n = doc.styles["Normal"]
    n.font.name = "Calibri"
    n.font.size = Pt(10.5)
    n.paragraph_format.space_after = Pt(6)
    for name, size, colour in (("Heading 1", 15, ACCENT), ("Heading 2", 12, ACCENT2)):
        st = doc.styles[name]
        st.font.name = "Calibri"
        st.font.size = Pt(size)
        st.font.bold = True
        st.font.color.rgb = colour
        st.paragraph_format.space_before = Pt(14 if name == "Heading 1" else 11)
        st.paragraph_format.space_after = Pt(6)
    for s in doc.sections:
        s.left_margin = s.right_margin = Pt(54)
        s.top_margin = s.bottom_margin = Pt(54)


def para(doc, text="", *, bold=False, italic=False, size=10.5, colour=None, space_after=6):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(space_after)
    r = p.add_run(text)
    r.bold, r.italic = bold, italic
    r.font.size = Pt(size)
    if colour is not None:
        r.font.color.rgb = colour
    return p


def rich(doc, parts, *, bullet=False, space_after=6):
    """parts: list of (text, style) where style is '' | 'b' | 'i' | 'm'.

    A bare string is accepted as unstyled text. Without that, a long implicitly-concatenated string in
    parentheses reads as a string rather than a 1-tuple and unpacks into characters, which is a silent trap in
    a file this long.
    """
    p = doc.add_paragraph(style="List Bullet" if bullet else None)
    p.paragraph_format.space_after = Pt(space_after)
    for part in parts:
        text, style = (part, "") if isinstance(part, str) else part
        r = p.add_run(text)
        r.bold = style == "b"
        r.italic = style == "i"
        if style == "m":
            r.font.name = MONO
            r.font.size = Pt(9)
        else:
            r.font.size = Pt(10.5)
    return p


def bullet(doc, text, *, bold=False):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run(text)
    r.bold = bold
    r.font.size = Pt(10.5)
    return p


def code(doc, lines):
    # Consolas 8.5pt in 486pt of usable width (Letter minus margins minus the 18pt indent) fits ~103 chars.
    # A wrapped monospace block reads as corrupted, and no structural check can see wrapping, so the width is
    # asserted here rather than eyeballed in a renderer this machine does not have.
    for i, line in enumerate(lines):
        assert len(line) <= 98, f"code line wraps ({len(line)} chars): {line[:60]}"
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(7 if i == len(lines) - 1 else 0)
        p.paragraph_format.space_before = Pt(3 if i == 0 else 0)
        p.paragraph_format.left_indent = Pt(18)
        r = p.add_run(line if line else " ")
        r.font.name = MONO
        r.font.size = Pt(8.5)


def _shade(cell, fill):
    el = OxmlElement("w:shd")
    el.set(qn("w:val"), "clear")
    el.set(qn("w:fill"), fill)
    cell._tc.get_or_add_tcPr().append(el)


def table(doc, headers, rows):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.LEFT
    for i, htxt in enumerate(headers):
        c = t.rows[0].cells[i]
        c.text = ""
        r = c.paragraphs[0].add_run(str(htxt))
        r.bold = True
        r.font.size = Pt(9)
        _shade(c, "E8EEF4")
    for row in rows:
        cells = t.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = ""
            p = cells[i].paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            r = p.add_run(str(val))
            numeric = i > 0 and all(ch in "0123456789.,+-−×%/ []e←nan" for ch in str(val))
            r.font.name = MONO if numeric else "Calibri"
            r.font.size = Pt(8.5 if numeric else 9)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def rule(doc):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(8)
    pr = p._p.get_or_add_pPr()
    b = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "6")
    bottom.set(qn("w:color"), "BBBBBB")
    b.append(bottom)
    pr.append(b)


def build(doc: Document) -> None:
    # ---------------------------------------------------------------- title
    t = doc.add_paragraph()
    t.paragraph_format.space_after = Pt(2)
    r = t.add_run("RAQ-KV")
    r.bold = True
    r.font.size = Pt(26)
    r.font.color.rgb = ACCENT
    para(doc, "Reliability-Aware Joint Token Compression, Mixed-Precision KV Quantization and "
              "Unequal Error Protection for LLM Agents", size=13, colour=RGBColor(0x44, 0x44, 0x44),
         space_after=10)
    para(doc, "Technical review dossier - complete record of methods, formulas, measurements and corrections",
         italic=True, colour=GREY, space_after=2)
    para(doc, "Model: Qwen3-1.7B @ 70d244cc | Hardware: RTX 5060 Ti | "
              "333 automated tests passing (7 GPU-gated, deselected) | decision records D1-D43",
         size=9, colour=GREY)
    rule(doc)

    doc.add_heading("How to read this document", level=1)
    para(doc, "This dossier is written for an external reviewer who has not seen the codebase. It records what "
              "was built, what was measured, the exact formulas behind every number, and - in its own section - "
              "every finding that was withdrawn or re-scoped after a control. Section 10 is the one worth "
              f"reading first: {len(WITHDRAWN_ROWS)} results in this project were stated and then withdrawn or "
              "narrowed by the project's own checks, none by an external reviewer, and the most useful thing a "
              "reviewer can do is find the next one.")
    rich(doc, [("Convention: ", "b"),
               ("every quantitative claim below is either a measurement with a paired confidence interval, or "
                "an explicit bound. Where a comparison could not discriminate - typically because both arms "
                "recorded zero failures - that is stated as a bound rather than as equality. Where a number was "
                "withdrawn, the withdrawal is recorded beside it rather than deleted.")])

    # ---------------------------------------------------------------- 1
    doc.add_heading("1. The problem and the original hypothesis", level=1)
    para(doc, "A transformer's KV cache dominates memory at long context. Three compression axes are usually "
              "studied separately:")
    bullet(doc, "Token / context retention - which cached tokens to keep at all.")
    bullet(doc, "Numerical precision - how many bits per cached value (KV quantization).")
    bullet(doc, "Error protection - ECC over the stored representation, against memory faults.")
    para(doc, "The brief proposed optimising all three jointly under one budget, driven by a multiplicative "
              "Joint Reliability Score:")
    code(doc, [
        "score(region) = semantic_importance x quantization_sensitivity x fault_vulnerability",
        "",
        "per-region decisions:  r in {KEEP, COMPRESS, PRUNE}   b in {2, 4, 8, 16}",
        "                       e in {NONE, PARITY, SECDED, BCH}",
    ])
    rich(doc, [("The measurements rejected that form.", "b"),
               (" Four findings make a single multiplicative score over per-(layer, token) cells the wrong "
                "object: the axes are not commensurate (6.1), protection is a lookup rather than a search "
                "dimension (7), two of the three signals are anti-correlated (6.4), and the retention decision "
                "is over spans that are all-or-nothing across layers (8). What replaced it is a staged rule "
                "with one measured switching point (9).")])

    # ---------------------------------------------------------------- 2
    doc.add_heading("2. Measurement protocol", level=1)
    doc.add_heading("2.1 Two-phase protocol", level=2)
    para(doc, "In a single forward pass K and V are produced and consumed together, so a cache transform is "
              "invisible. Every measurement therefore uses two phases:")
    code(doc, [
        "phase 1   prefill(context)        -> KVSnapshot       (detached copy of the cache)",
        "phase 2   transform(snapshot)     -> quantize / prune / inject faults / decode through ECC",
        "phase 3   score_through_cache(.) or greedy_decode(.)  -> the metric",
    ])
    para(doc, "Consequence: WikiText numbers are cache-conditioned perplexity and are not number-for-number "
              "comparable with published sliding-window perplexities; only trends are compared.")

    doc.add_heading("2.2 Position and masking convention", level=2)
    para(doc, "transformers 5.17 derives the attention-mask size and the query offset from the physical cache "
              "length; there is no cache_position argument. Only position_ids (RoPE) is controlled and it must "
              "be passed explicitly. After pruning, retained tokens keep their ORIGINAL absolute positions - "
              "pruning never renumbers.")
    rich(doc, [("Verified empirically: ", "b"),
               ("middle-pruning plus a 5-token chunk plus 3 greedy steps equals the masked full cache to "
                "max |delta| = 4.2e-7, under both sdpa and eager attention. Omitting position_ids changes "
                "logits by 0.30. model.generate() is banned on modified caches; a hand-written greedy loop is "
                "used everywhere.")])

    doc.add_heading("2.3 Noise floor", level=2)
    para(doc, "In bfloat16, cache-geometry changes alone move logits by up to about 0.3 (KL about 1e-4) without "
              "changing the argmax, and roughly 0.4% of greedy steps sit on exact bf16 ties, which makes those "
              "steps tie-break dependent. Effect sizes below that floor are not attributable.")

    # ---------------------------------------------------------------- 3
    doc.add_heading("3. Bit accounting - the formulas every budget uses", level=1)
    para(doc, "All memory numbers are analytic rather than read from the allocator, because the implementation "
              "uses fake quantization: codes are reconstructed to bf16, so the stored tensors do not shrink. At "
              "full precision the analytic number was validated against the CUDA allocator delta - ratio "
              "exactly 1.0000.")
    code(doc, [
        "kv_values_per_token = 2 * L * H_kv * d_head           (K and V, all layers, all KV heads)",
        "",
        "total_bits     = kv_data_bits + quant_metadata_bits + ecc_bits + policy_metadata_bits",
        "bits_per_value = total_bits / (T * kv_values_per_token)",
        "",
        "Qwen3-1.7B:  L = 28,  H_kv = 8,  d_head = 128,  dtype = bf16",
        "             kv_values_per_token = 2*28*8*128 = 57344       (112 KiB per token at bf16)",
    ])
    doc.add_heading("3.1 Quantization metadata", level=2)
    para(doc, "Scales and zero points are fp16 and are counted. For one layer-part of shape [1, H_kv, T, d]:")
    code(doc, [
        "n_groups(per_tensor)      = 1",
        "n_groups(per_head)        = H_kv",
        "n_groups(per_channel)     = H_kv * d",
        "n_groups(per_token)       = H_kv * T",
        "n_groups(group_channel,g) = H_kv * T * ceil(d / g)",
        "n_groups(group_token,g)   = H_kv * d * ceil(T / g)",
        "",
        "quant_metadata_bits = n_groups * 16 * (1 if symmetric else 2)",
        "kv_data_bits        = n_values * b",
    ])
    doc.add_heading("3.2 Residual window (the streaming tail KIVI uses)", level=2)
    para(doc, "A residual window R keeps the most recent R tokens at the model dtype. Both terms split:")
    code(doc, [
        "kv_data_bits = (T-R) * V_part * b  +  R * V_part * dtype_bits     V_part = H_kv * d per layer-part",
        "groups are computed over (T-R) tokens only",
        "",
        "=> the window's cost per value scales as R/T: expensive at short context, cheap at long.",
    ])
    doc.add_heading("3.3 Error-correcting codes", level=2)
    code(doc, [
        "extended Hamming (SECDED) over w data bits: check_bits = r + 1, smallest r with 2^r >= w + r + 1",
        "     w = 8 -> 5      w = 16 -> 6      w = 32 -> 7      w = 64 -> 8",
        "overhead(SECDED over 64) = 8/64 = 12.5% of the data bits",
        "",
        "ecc_bits(region) = ceil(region_bits / w) * check_bits     (the tail word is padded and paid for)",
        "range guard      = 2 fp16 bounds per metadata region      (about 3584 bits, 0.0006% of storage)",
    ])
    doc.add_heading("3.4 When SECDED fails - the coincidence argument", level=2)
    para(doc, "SECDED corrects one error per word and detects two, so it fails only when two flips land in the "
              "same word. With n flips uniform over W words, by the birthday argument:")
    code(doc, [
        "E[words with >= 2 flips] ~= n^2 / (2W),     W = code_bits / word_bits",
        "",
        "Qwen3-1.7B at T = 2066, 4-bit:  code_bits = 474 M,  W = 7.40 M words of 64 bits",
        "   BER 1e-6  -> n ~= 575    E ~= 0.03   (measured DUE per trial 0.0148)",
        "   BER 1e-5  -> n ~= 5749   E ~= 1.5    (metadata words: 0.051 predicted, 0.058 measured)",
    ])

    # ---------------------------------------------------------------- 4
    doc.add_heading("4. Statistical method", level=1)
    rich(doc, [("The unit of analysis is the sample, not the trial.", "b"),
               (" Trials within one sample are strongly dependent: in one campaign arm 12 damaging events came "
                "from 3 samples and another arm's 7 came from 2. A test on pooled trials treats those as "
                "hundreds of independent observations and reports a difference that is really one arm drawing "
                "one more fragile sample.")])
    code(doc, [
        "per-sample rate    r_i = failures over that sample's draws / draws",
        "paired comparison  over the samples EVERY arm answers correctly without perturbation",
        "significance       Wilcoxon signed-rank on the paired per-sample rates",
        "interval           bootstrap over samples, 10 000 resamples, 2.5% / 97.5% quantiles",
        "multiplicity       Holm correction across the pre-registered family",
        "zero-count arms    Wilson score interval (finite and asymmetric at k = 0)",
        "zero events        rule of three: 95% upper bound on the rate = 3/n",
    ])
    rich(doc, [("Two arms both at zero are a bound, not an equality.", "b"),
               (" A paired test with zero discordant samples is uninformative about small differences; such "
                "rows report an upper bound (about 0.6% on 472 trials) rather than claiming the arms match.")])
    rich(doc, [("Context length is a stratum, not a nuisance.", "b"),
               (" The probe spans 512-4096 tokens, an eightfold range, and one pooled finding turned out to be "
                "Simpson's paradox: long samples have both four times the flips and all of the damage, so a "
                "pooled odds ratio of 4.79 (p = 0.012) fell to 1.66 and non-significance inside the only band "
                "that had events.")])

    doc.add_page_break()

    # ---------------------------------------------------------------- 5
    doc.add_heading("5. Infrastructure built", level=1)
    table(doc, ["Component", "What it does", "The guard that makes it trustworthy"], [
        ["kv/snapshot", "Detached cache copies; token selection", "prune-equals-mask pinned to 4.2e-7"],
        ["quantization/", "Uniform affine, six axes, exact bit accounting",
         "grid values round-trip exactly; causality test"],
        ["budget/iso", "Closed-form budget predictor",
         "pinned bit-for-bit to the real encoders, every axis and residual windows"],
        ["faults/", "Bit-flip plans over the DECLARED representation",
         "every site addressable and exactly reversible"],
        ["ecc/", "none / parity / SECDED, explicit DUE erasure policy",
         "exhaustive 1-bit and 2-bit tests; CPU and GPU bit-identical"],
        ["importance/", "Token scorers, each declaring what it may see",
         "oracles labelled; an information leak is visible in the interface"],
        ["compression/", "Span segmentation and retention",
         "spans kept whole or skipped, never truncated"],
        ["allocation/", "The staged joint allocator",
         "every rule cites its decision record; ladder boundaries enumerated"],
        ["experiments/", "Config-hashed runs, resume, provenance",
         "refuses an experiment whose code changed mid-process"],
    ])
    rich(doc, [("Reproducibility machinery. ", "b"),
               ("Every run is keyed by exp_id = <human slug>-<first 8 hex of sha256(canonical identity JSON)>. "
                "The manifest stores the full config, the dataset fingerprint, the git commit and a hash of "
                "git diff HEAD, and a run whose code state differs from previous runs of the same id is "
                "refused. Raw JSONL is never edited or deleted: superseded runs are marked with superseded_by "
                "and renamed, and a crash-truncated line is kept with a marker record appended.")])

    # ---------------------------------------------------------------- 6
    doc.add_heading("6. Quantization and fault sensitivity (M1, M2)", level=1)
    doc.add_heading("6.1 Precision and protection are not exchangeable (H2c)", level=2)
    para(doc, "The brief assumed a bit-width step could be traded against ECC. The arithmetic forbids it: SECDED "
              "costs 12.5% of the data bits, a bit-width step costs 100%. The originally planned equal-budget "
              "pair measures 4.844 against 2.344 bits per value - not a comparison. The tradeable fine axis is "
              "group size, which is why the allocator has a coarse stage and a fine stage rather than one search.")
    doc.add_heading("6.2 Where quantization hurts (H1, keys vs values)", level=2)
    para(doc, "Sensitivity is heterogeneous but not concentrated: the top 10% of cells carry 40.2% of the damage "
              "- above the 20% refutation line, below the 50% support line, so H1 is recorded as intermediate on "
              "the quantization side. The ranking is stable across seeds, and keys are about 11x more sensitive "
              "than values, which is why every configuration in this project quantizes keys along group_token "
              "and values per_token.")
    doc.add_heading("6.3 Where faults hurt - the finding that redirected the project (H1b, H1c, H1d)", level=2)
    para(doc, "The pilot expected layer and bit position to be the axes. They are not. The dominant axis is the "
              "target class:")
    table(doc, ["Signal", "Measured", "What it implies"], [
        ["metadata vs codes (mean damage)", "2.4x", "metadata is the target, not the data"],
        ["worst case", "23.6 nats vs 0.81", "the entire damage tail is metadata"],
        ["metadata share of stored bits", "about 6%", "the high-value target is also the cheap one"],
        ["LSB -> MSB stored-value change", "128x", "the obvious protection signal..."],
        ["LSB -> MSB damage", "1.30x", "...buys almost nothing: MSB-only ECC is poor value"],
        ["one scale bit", "7/218 = 3.21% of answers", "per-bit, metadata is catastrophic"],
        ["2048 code bits in one token row", "14/654 = 2.14%", "per-bit, codes are almost inert"],
    ])
    doc.add_heading("6.4 The two sensitivity signals are anti-correlated (Q48)", level=2)
    rich(doc, [("rho = -0.377 [-0.588, -0.128], partial rho on KV magnitude -0.282, replicated at 8 bits. ", "b"),
               ("Faults hurt most in early layers; quantization hurts most in late-layer keys. This landed in the "
                "pre-registered 'separate signals' branch (the threshold was frozen before measurement) and it is "
                "the single most damaging result for the brief's design: multiplying two anti-correlated signals "
                "produces a ranking that is worse than either.")])
    doc.add_heading("6.5 Token roles matter only at the scale where a whole row dies (Q49)", level=2)
    para(doc, "Pre-registered before measurement, and the answer is scale-dependent: a single bit flip or a whole "
              "16-bit word gives an odds ratio of 1.00 - no effect. Destroying the token's entire row in a layer "
              "gives OR 7.13 against filler and 4.75 against a content-matched distractor, and only in late "
              "layers. The content-matched distractor is the control that makes this a claim about role rather "
              "than about position.")
    doc.add_heading("6.6 Perplexity understates task damage (C4, first evidence)", level=2)
    para(doc, "+29% cache-conditioned perplexity coexists with -71% probe accuracy; +7.5% with -13%. This is why "
              "M3-M5 report task failure, not perplexity, and why D12's earlier '2-bit is degraded but "
              "functional' was withdrawn - it rested on perplexity alone.")

    doc.add_page_break()

    # ---------------------------------------------------------------- 7
    doc.add_heading("7. Error protection under equal budget (M3)", level=1)
    para(doc, "Eight arms, all at the same declared storage +/-1%, all exposed to the same bit error rate over "
              "every bit they claim to store - codes, quantization metadata and their own ECC check bits. "
              "Sparing check bits would grant protected arms an immunity no physical memory grants them.")
    table(doc, ["Arm", "Policy", "Storage vs unprotected", "Task failure at BER 1e-6"], [
        ["A", "unprotected 4-bit", "0.0% (4.2605 b/v)", "0.3211 [0.2629, 0.3815]"],
        ["B", "uniform SECDED over all data", "+14.0%", "0 of 472"],
        ["C", "finer groups, no data ECC", "about +0%", "0 of 472"],
        ["D", "data ECC on late layers", "+9.5%", "0 of 472"],
        ["E", "data ECC on early layers", "+9.5%", "0 of 472"],
        ["G", "metadata SECDED only", "+2.3% (4.3581 b/v)", "0.0000 (p = 6e-10)"],
        ["H", "range check on metadata only", "+0.075%", "0 of 472"],
    ])
    rich(doc, [("The headline (H2d). ", "b"),
               ("Arm G removes 100% of the measured task failure - 0.3211 [0.2629, 0.3815] to 0.0000, "
                "p = 6e-10, 50 of 58 samples discordant, a powered comparison - for 2.3% more storage. Arm H "
                "does the same at BER 1e-6 for 0.075% more storage and no code at all. Metadata carries 100% of "
                "the task-failure damage and 95.3% of the answer-changed proxy while being about 6% of the bits.")])
    rich(doc, [("The honest limit (H2a, H2b). ", "b"),
               ("Arms B, C, D and E all recorded zero task failures on all 58 shared samples. A paired test with "
                "zero discordant samples cannot discriminate, so these rows are reported as bounds: uniform data "
                "ECC buys at most about 0.6 points of task failure (rule of three on 472 trials) for its 0.5 "
                "bits per value, and ECC placement cannot matter while data ECC itself buys nothing. On the "
                "answer-changed proxy B beats C by -0.0108 [-0.0237, -0.0022], p = 0.059 on 4 discordant "
                "samples - below the pre-registered bar.")])
    rich(doc, [("Where the cheap guard breaks (H2e, D30). ", "b"),
               ("At BER 1e-5 the range check loses decisively to metadata SECDED: +0.0593 [+0.0381, +0.0847], "
                "p = 1.1e-5, 22 of 59 samples discordant, because the roughly 13 of 16 float16 bits it cannot "
                "see accumulate. It catches 20.1% of metadata flips against the 3/16 = 18.75% the mechanism "
                "predicts. Its own failure mode is named rather than hidden: its bounds are themselves stored "
                "and exposed, and a corrupted bound narrows the range so the guard flags 625% of the real flips "
                "- it clamps valid scales. The code self-validates the bounds and counts every such event.")])
    rich(doc, [("H2b, resolved as far as it can be (D31). ", "b"),
               ("On 177 paired samples of the 4096-token condition at 1e-5, task failure is 0 of 1416 trials in "
                "both placement arms - protecting 19 of 28 layers is sufficient regardless of which 19. The "
                "proxy difference is +0.0113 [+0.0007, +0.0247], p = 0.074, 7 discordant samples, which does not "
                "meet the pre-registered rule (p < 0.05 AND a CI excluding zero). The direction has replicated "
                "4 of 4 times at 1.7x-3.5x and in 6 of 7 discordant samples, and is reported as a consistent "
                "direction below significance - not as a result.")])

    # ---------------------------------------------------------------- 8
    doc.add_heading("8. Token retention (M4)", level=1)
    doc.add_heading("8.1 The unit of retention is the span, not the token (H4)", level=2)
    para(doc, "At 25% retention, paired over 117 samples:")
    table(doc, ["Arm", "Task failure", "Paired difference vs the arm above"], [
        ["recency (the trivial baseline)", "0.6923", "-"],
        ["perfect TOKEN oracle (keeps every answer token)", "0.5641", "worse than a same-sized span"],
        ["contiguous window elsewhere (locality control)", "0.5812", "+0.4274 [+0.3333, +0.5214], p = 1.5e-12"],
        ["sentence oracle (span unit)", "0.1538", "+0.4103 [+0.3248, +0.5043], p = 4.3e-12"],
        ["query-aware attention at span granularity", "0.0256", "+0.6667 [+0.5812, +0.7521], p = 1e-18"],
    ])
    rich(doc, [("A perfect token oracle keeping every answer token fails 56% of the time.", "b"),
               (" That is the result: the answer tokens alone are not sufficient, their neighbourhood is. The "
                "locality control settles what kind of neighbourhood - a same-sized contiguous window somewhere "
                "else scores 0.5812, indistinguishable from keeping no context at all, so the gain is the fact's "
                "own span and not contiguity. The unit effect is worth +18 to +26 points on every real scorer "
                "tried and compounds with query-awareness.")])
    doc.add_heading("8.2 The attention sink, caught by a control arm (D32)", level=2)
    para(doc, "Both M4 gate arms failed 100% of samples. A diagnostic isolated the cause: dropping token 0 alone "
              "fails 12 of 12. Position 0 is an attention sink (StreamingLLM), so every retention policy in the "
              "project now pins the first 4 tokens. This was found by the control arm, not by inspection - "
              "without a 100%-failure arm in the design it would have been read as a weak scorer.")
    doc.add_heading("8.3 Query-awareness dominates, and leaves an open problem (D36)", level=2)
    rich(doc, [("Query-aware scoring is worth 57 points over the query-agnostic scorer (0.026 against 0.624). ", "b"),
               ("Every scorer therefore declares in its interface whether it may see the query and whether it is "
                "an oracle, so an information leak is visible in the type rather than buried in a config. The "
                "gap is also the project's largest unsolved problem: a shared prefix cache is compressed before "
                "any query arrives, and there the best available arm sits at 0.624.")])

    doc.add_page_break()

    # ---------------------------------------------------------------- 9
    doc.add_heading("9. The joint allocator (M5)", level=1)
    doc.add_heading("9.1 Two regimes, and an interior optimum (H5, D37, D38, D40)", level=2)
    para(doc, "Iso-bit families were run at about 4 and about 1 bit per original token, each family holding total "
              "storage fixed while moving the split between how many tokens are kept and how many bits each gets.")
    table(doc, ["Regime", "Result", "Statistics"], [
        ["about 4 bits / original token", "no trade: all three splits within 3 failures of 117",
         "nothing significant"],
        ["about 1 bit / original token", "more tokens at lower precision wins at every step",
         "+0.1880, +0.1282, +0.0598; p = 2.7e-06, 1.1e-04, 0.0082"],
        ["below 4 bits (D40)", "the trade REVERSES: 2-bit at 40.5% loses to 4-bit at 25%",
         "+0.1282, p = 0.00027, 100% of answer tokens kept in both"],
        ["3-bit vs 4-bit", "statistically indistinguishable - the optimum is a plateau", "p = 0.32"],
    ])
    rich(doc, [("This is the one rule in the allocator that could not be derived and had to be measured. ", "b"),
               ("Because both arms keep 100% of the answer tokens, the damage below 4 bits is per-token "
                "quantization error rather than lost content, which is what reconciles D38 (spend a tight budget "
                "on tokens) with D16 (2-bit destroys the task). The floor is set at 4 bits, the conservative end "
                "of the plateau.")])
    doc.add_heading("9.2 Against a faithful published baseline (H6, D39)", level=2)
    para(doc, "Review required a real baseline rather than a reimplementation at our convenience, so KIVI was "
              "built at its own published configuration - keys group_token 32, values group_channel 32, residual "
              "window 128 - and its window is paid for in the budget:")
    table(doc, ["Configuration", "Bits / original token", "Task failure"], [
        ["KIVI faithful 2-bit, residual 128", "4.331", "5.13% (6 of 117)"],
        ["KIVI 4-bit (lossless on this probe)", "6.13", "0%"],
        ["RAQ-KV joint point: 25% of tokens at 4-bit", "1.054 (15.2x compression)", "0.85% (1 of 117)"],
    ])
    rich(doc, [("4.1x cheaper and 6x fewer failures. ", "b"),
               ("One finding runs against us and is reported: KIVI's fixed residual window costs R/T per value, "
                "so below about 1500 tokens KIVI is dominated by plain 4-bit with no window at all - a result "
                "about the window, not about KIVI's authors' claim, which is made at long context.")])
    doc.add_heading("9.3 What the allocator actually does", level=2)
    para(doc, "Not a search over a multiplicative score. A staged rule, each stage citing the measurement that "
              "set it:")
    code(doc, [
        "allocate(budget_bits_per_original_token, ber, query_known) -> Allocation",
        "",
        "PROTECTION   lookup on ber, not a search dimension            (D30)",
        "             ber <= 1e-6  -> range check on metadata, +0.075%",
        "             ber >  1e-6  -> metadata SECDED, +2.3%",
        "             uniform data ECC is never chosen                 (D25)",
        "",
        "COARSE       highest precision whose FULL-retention cost fits (D38/D40)",
        "             floor = 4 bits; trade tokens ONLY below the floor",
        "",
        "FINE         spend the remainder on smaller groups            (D22)",
        "",
        "UNIT         spans, not tokens             +18 to +26 points  (D34)",
        "SINK         first 4 tokens always pinned                     (D32)",
        "SCORER       query_attention if the query is known            (D36)",
    ])
    rich(doc, [("Budget convention: per ORIGINAL token, never per retained token. ", "b"),
               ("Bits per retained token improves for free under pruning and would flatter every retention "
                "policy; the budget type keeps the retained count as a separate field to enforce this.")])
    rich(doc, [("Infeasibility is returned, not hidden. ", "b"),
               ("A precision-only policy bottoms out at 2-bit, about 2.63 bits per original token with its "
                "scales, so below that it returns None rather than overspending and calling the result a "
                "comparison. That is a result about the axis.")])
    rich(doc, [("A bug this design caught. ", "b"),
               ("An earlier ladder walked cheapest-first and at a budget of 4.30 chose 8-bit at 52% of tokens "
                "over 4-bit at 100% - exactly the choice D38 and D40 had measured as worse. The rung boundaries "
                "are now enumerated in tests. H3's configs are also generated by the allocator rather than "
                "written by hand, so a gap between what the policy decides and what the experiment runs cannot "
                "hide.")])
    doc.add_heading("9.4 H3: the joint policy against each single-signal policy at equal budget", level=2)
    para(doc, "Six arms at two budgets, every config emitted by the allocator rather than written by hand. "
              "Precision-only emits nothing at either budget: quantization alone bottoms out at about 2.63 bits "
              "per original token, so the policy the quantization literature uses does not reach these budgets "
              "at all. That is H3's first result and it is an absence, not a number.")
    table(doc, ["Arm (117 paired samples)", "Retention", "Measured bits / orig. token", "Kept target",
                "Task failure"], H3_ROWS)
    rich(doc, [("Paired against the joint arm, task failure (Wilcoxon, bootstrap CI over samples, Holm across "
                "the family of four):", "b")])
    code(doc, H3_PAIRS)
    rich(doc, [("All six contrasts survive Holm correction - and H3 still does not meet its own "
                "pre-registration. ", "b"),
               ("The plan requires the joint policy to beat every single-signal policy on AT LEAST TWO model "
                "families. This is one. So the honest statement is: supported on Qwen3-1.7B against the "
                "baselines actually run, with the general claim not made. That gap was found by re-reading the "
                "pre-registration after the result was written up, not by a control arm, and it is the twelfth "
                "row of section 10.")])
    rich(doc, [("The baseline the plan called 'the strongest single-signal policy' is the weakest thing "
                "measured. ", "b"),
               ("Magnitude-only retention fails 95.7% at 1.05 bits - worse than recency, which was supposed to "
                "be the trivial baseline. It was also missing from the first campaign entirely. Key magnitude "
                "selects answer-bearing tokens at chance: against the share a uniformly random selection of "
                "the same size would keep, it lands at -0.0339 (p = 0.077) at 1.05 and +0.0286 (p = 0.17) at "
                "2.00 - indistinguishable at both, with the two trends pointing opposite ways, which supports "
                "'no signal' over 'anti-signal'. This was tested rather than eyeballed; the 0.2128-against-"
                "0.2467 gap looks like a below-chance result and is not one. It matters beyond H3, because "
                "magnitude is computable at prefill and is therefore the only baseline here a shared prefix "
                "cache could run.")])
    rich(doc, [("The iso-budget rule was broken on first measurement, and the fix was to re-run rather than to "
                "argue. ", "b"),
               ("The first campaign had the joint arm spending 2.8% and 1.45% MORE than retention-only, outside "
                "the +/-1% tolerance and in the winning arm's favour. The mechanism is span retention: whole "
                "spans are kept, so the realised ratio undershoots the requested one, and it undershoots more "
                "at a small ratio because spans are coarse relative to a tiny budget. The baselines were "
                "re-emitted at the ratio whose REALISED budget matches the joint arm and re-run; every "
                "comparison in the table above is now within +0.17%, and no baseline is given less storage "
                "than the arm it is compared against. The correction cost the joint arm 0.86 points at 1.05 "
                "and 1.71 at 2.00 - both smaller than the margins, and both measured rather than estimated. "
                "The prior estimate from D40's frontier was about 1.4 points, which overstated the 1.05 case "
                "and understated the 2.00 one.")])
    rich(doc, [("The correction is applied in one direction only. ", "b"),
               ("An arm already spending at or above the joint arm is left alone: the recency arms sit at "
                "+1.33% and +0.82%, outside a symmetric tolerance, and pulling them down to comply would be "
                "shaving a baseline's budget to make it lose by more. That asymmetry is a tested property of "
                "scripts/calibrate_iso_budget.py, not a convention.")])
    rich(doc, [("What this comparison does NOT establish. ", "b"),
               ("Read the kept_target column before the task-failure column. Recency keeps 31.6% of the answer "
                "tokens and the joint arm keeps 100%, and the depth breakdown shows recency succeeding only on "
                "deep targets, which its window keeps for free (0.0270 failure deep, 1.0000 shallow). So the "
                "margin against recency is mostly the query-awareness effect D36 already measured, re-measured "
                "at a fixed budget - it is not independent evidence that joint allocation is what helps. The "
                "comparison that isolates the precision axis is joint against retention-only, where both arms "
                "use query-aware span retention and differ only in whether the budget is spent on bits or on "
                "more tokens - and that margin is 12.8 points at 1.05 and 5.1 at 2.00, not 66.")])
    rich(doc, [("Why section 9.2 says 0.85% and this table says 3.42% for what looks like the same row. ", "b"),
               ("They are two runs, not one. The M5 joint point keeps 25.00% of tokens and the H3 joint arm "
                "keeps 24.67%, at the same 4-bit quantization, so the H3 arm has a 1.3% smaller realised budget "
                "(1.0401 against 1.0543). Paired over the same 117 samples the difference is -0.0256 [-0.0598, "
                "+0.0000], p = 0.083 on 3 discordant samples - it does not reach significance and the interval "
                "touches zero. That is measured, not argued, and it doubles as a small replication: the joint "
                "point reproduces at a tighter budget with a difference below the noise the design can resolve.")])
    rich(doc, [("A coupling the emitter found the hard way. ", "b"),
               ("Both retention-only arms were initially refused by the config schema: the allocator attached "
                "its +0.075% metadata guard to a policy that does not quantize, and an unquantized cache has no "
                "quantization metadata to guard. The allocator was wrong and the schema was right. The fix is a "
                "result in its own right - every protection mechanism M3 chose acts on quantization metadata, so "
                "a retention-only policy is unprotectable by the cheapest mechanism this project has, and its "
                "only remaining option is uniform data ECC at +14%, which D25 measured as never worth its bits. "
                "This is one place where the axes are genuinely coupled rather than separable, and it "
                "is now asserted by a test rather than left for the schema to catch.")])

    doc.add_heading("9.5 Crossing two axes for the first time: does pruning change what a fault costs?", level=2)
    rich(doc, [("Every result above this line prices one axis at a time. ", "b"),
               ("Protection was measured on unpruned caches (section 7) and retention and precision with no "
                "faults injected (sections 8 and 9) - all 30 retention runs were fault-free. The allocator's "
                "staged design assumes the reliability tax is flat in retention: its protection stage is a "
                "lookup on the error rate with no retention term at all. Nothing had tested that.")])
    para(doc, "Five arms, 4-bit, BER 1e-6, 60 samples x 8 trials plus a no-fault control per sample (every "
              "control at exactly zero damage). Paired over the 56 of 60 samples every arm answers correctly "
              "unfaulted AND that the full cache also answers, so a sample pruning alone breaks is never "
              "charged to the fault model.")
    table(doc, ["Arm", "Stored", "Flips / trial", "Task failure"], [
        ["unpruned, unprotected", "540.5 Mbit", "540", "0.3125"],
        ["pruned 25%, unprotected - matched RATE", "132.6 Mbit", "132", "0.1875"],
        ["pruned 25%, unprotected - matched EXPOSURE", "132.6 Mbit", "540", "0.4018"],
        ["unpruned, +0.075% range guard", "540.5 Mbit", "540", "0.0000"],
        ["pruned 25%, +0.075% range guard", "132.6 Mbit", "132", "0.0268"],
    ])
    code(doc, [
        "                                       difference                  p    discordant   rule",
        "1  matched RATE      pruning helps  -0.1250 [-0.1942, -0.0558]  6.1e-04    45 of 56    met",
        "2  matched EXPOSURE  pruning hurts  +0.0893 [+0.0223, +0.1562]  1.6e-02    42 of 56    met",
        "3  132 vs 540 flips, same cache     +0.2143 [+0.1652, +0.2656]  5.1e-09    44 of 56    met",
        "4  guard, pruned vs unpruned        +0.0268 [+0.0022, +0.0625]  6.6e-02     4 of 56    NOT met",
    ])
    rich(doc, [("Contrast 1 is not a result, and the config header said so before the run. ", "b"),
               ("A cache pruned to a quarter stores a quarter of the bits, so at a matched RATE it takes a "
                "quarter of the flips - 132 against 540. 'Pruning improved reliability' is then arithmetic "
                "about exposure, not a statement about importance.")])
    rich(doc, [("Contrast 2 is the result: at equal exposure, pruning makes faults MORE damaging. ", "b"),
               ("Give the pruned cache the same absolute number of flips the unpruned one takes on that same "
                "sample and it fails 0.4018 against 0.3125. The mechanism follows from section 8: query-aware "
                "span retention keeps exactly the answer-bearing spans (kept_target = 1.000), so every scale "
                "that survives pruning governs a group the answer depends on, where most groups in a full cache "
                "govern filler. Protection is NOT separable from retention - which is the one thing the "
                "multiplicative form this project rejected got right.")])
    rich(doc, [("The compositional alternative, checked rather than argued. ", "b"),
               ("Contrast 2's arms differ in more than pruning: the pruned cache holds about 4x fewer "
                "quantization groups, so its flips might land in metadata more often - and metadata is at least "
                "1543x more damaging per bit (section 7), which would explain part of +0.0893 without any "
                "appeal to importance. Measured: metadata takes 6.1192% of the unpruned arm's flips and 6.1468% "
                "of the pruned arm's, a paired difference of +0.028 percentage points at p = 0.57. The reason is "
                "structural - with keys on group_token at group size 128, the group count scales with the token "
                "count exactly as the code count does, so pruning leaves the composition invariant. The effect "
                "is therefore not compositional. This control would need rerunning for a per_channel or "
                "per_tensor arm, where group count does not scale with tokens.")])
    rich(doc, [("Contrast 3 explains why contrast 1 looks so good. ", "b"),
               ("Damage is strongly sublinear in flip count: 4.1x the flips buys only 2.1x the failure, because "
                "an answer already broken cannot break further. The exposure drop therefore pays off more than "
                "proportionally, and that is what lets the concentration effect hide inside a matched-rate "
                "comparison.")])
    rich(doc, [("What was deliberately NOT changed. ", "b"),
               ("The allocator's protection constants stand. Contrast 4 - the guard reaching exactly 0.0000 "
                "unpruned and 0.0268 pruned - has 4 discordant samples and p = 0.066, so it fails the "
                "pre-registered rule of section 4 and no retention-dependent threshold was derived from it. "
                "Inventing one from four events is the error that produced three of the withdrawals in section "
                "10. What the code carries instead is a stated caveat, pinned by a test so a refactor cannot "
                "tidy it away: this lookup is unconservative for aggressively pruned caches. The experiment "
                "that would set the constant is contrast 4 at 1e-5, where the guard's breakdown was located.")])
    rich(doc, [("The practical reading, which runs the other way. ", "b"),
               ("At the matched rate a real deployment faces, pruning still cuts absolute failure from 0.3125 "
                "to 0.1875, and the guard still removes most of what is left. Compression and robustness do "
                "pull the same way in practice - just not for the reason the staged design assumed.")])

    doc.add_page_break()

    # ---------------------------------------------------------------- 10
    doc.add_heading("10. Findings withdrawn or re-scoped after a control", level=1)
    rich(doc, [("This is the section to read first. ", "b"),
               ("Every row below was stated as a result and then withdrawn or narrowed by this project's own "
                "checks - none by an external reviewer. The decision log numbers the D36 retraction as the "
                "seventh such event, and there have been more since. Nothing was deleted: superseded runs keep "
                "their raw JSONL, marked with superseded_by and renamed.")])
    table(doc, ["Withdrawn claim", "What killed it", "What stands instead"], WITHDRAWN_ROWS)
    rich(doc, [("Two of these were caught by a control arm that was in the design for a different reason. ", "b"),
               ("The attention sink surfaced because an arm was expected to fail and failed 100% instead of "
                "partially; the clustering problem surfaced because per-sample counts were printed beside the "
                "pooled table. Both would have been invisible in a design that only ran the arms it expected to "
                "win.")])

    # ---------------------------------------------------------------- 11
    doc.add_heading("11. Limitations - stated, not discovered by the reviewer", level=1)
    bullet(doc, "One model. Qwen3-1.7B only. Every claim in this document is single-model. Llama-3.2-3B has "
                "identical KV geometry and is the planned replication; it has not been run.", bold=False)
    bullet(doc, "One synthetic probe for the task metric. RAQ-Probe's facts are single sentences by "
                "construction, which is exactly the structure the span finding (H4) exploits. The span result "
                "must be treated as probe-dependent until it replicates on natural long-context tasks.")
    bullet(doc, "No agent benchmark yet. C4 - that token-level metrics understate agent failures - has evidence "
                "from retrieval only. The agent-level claim needs M6 (BFCL / tool-call) and is not made here.")
    bullet(doc, "Fake quantization: no speed or throughput claim. Codes are reconstructed to bf16, so nothing "
                "actually shrinks in memory and no kernel exists. Every memory number is analytic (validated "
                "against the CUDA allocator at full precision, ratio 1.0000). This project makes NO latency or "
                "bandwidth claim, and a reviewer should treat any such claim as absent rather than implied.")
    bullet(doc, "Perplexity numbers are cache-conditioned and are not comparable to published sliding-window "
                "perplexities. Only trends are compared.")
    bullet(doc, "Fault model is uniform random bit flips at a stated rate over declared storage. Real DRAM "
                "faults are bursty and spatially correlated; the burst regime is where the live version of the "
                "ECC-placement question (H2b) belongs and it has not been run.")
    bullet(doc, "The shared-prefix-cache gap is open. Query-agnostic scoring sits at 0.624 against query-aware "
                "0.026, and a prefix cache is compressed before the query exists.")
    bullet(doc, "ECC golden vectors for an RTL cross-check have not been emitted, and the M2 and M4 exit "
                "checklists have not been written (M0, M1 and M3 have).")
    bullet(doc, "H3 does not meet its own pre-registration. The plan requires the joint policy to beat every "
                "single-signal policy on AT LEAST TWO model families; this is one. Until the second family is "
                "run, H3 is supported on Qwen3-1.7B and the general claim is not made.")
    bullet(doc, "The allocator's protection lookup is unconservative for aggressively pruned caches, and is "
                "shipped that way on purpose. Section 9.5 measures the effect; the constant that would fix it "
                "rests on 4 discordant samples, so it was not invented. A deployment pruning to 25% should not "
                "assume the +0.075% guard buys the clean zero M3 measured on a full cache.")
    bullet(doc, "Only ONE pair of axes has ever been varied together. Section 9.5 crosses retention with "
                "faults; precision has never been crossed with faults at a fixed retention, and the three-way "
                "interaction is unmeasured. Every joint claim outside 9.5 is a claim about two axes priced "
                "independently.")
    bullet(doc, "Single-seed for most retention arms. The retention task is deterministic greedy decoding, so "
                "the seed affects sample selection rather than the measurement, but multi-seed replication of "
                "the M4 and M5 headline arms has not been done.")

    # ---------------------------------------------------------------- 12
    doc.add_heading("12. Questions for the reviewer", level=1)
    para(doc, "Ranked by how much the answer would change. A reviewer who only has time for three should take "
              "1, 2 and 4.")
    bullet(doc, "1. Is the span finding (H4) an artefact of RAQ-Probe's construction? Its facts are single "
                "sentences, so 'the span is the unit' may be a restatement of the probe design. What would "
                "falsify it on a natural task?")
    bullet(doc, "2. Does the metadata-protection result (H2d) survive a real quantization kernel? The claim is "
                "that 0.075% of storage removes all measured fault-induced failure. In a real implementation "
                "scales may live in registers or SRAM with different exposure than the codes. Does the finding "
                "then reduce to 'protect your SRAM', which nobody disputes?")
    bullet(doc, "3. Is the 4-bit floor (D40) a property of this model's outlier structure rather than a general "
                "frontier? 3-bit is indistinguishable from 4-bit (p = 0.32), so the plateau's left edge is not "
                "actually located.")
    bullet(doc, "4. Is the KIVI comparison fair? It is run at its published configuration including its "
                "residual window, at 512-4096 tokens, while KIVI's own claims are at longer context where the "
                "window is cheaper. Is 'below 1500 tokens the window does not pay' a fair thing to report, or "
                "does the comparison need to move to KIVI's own regime?")
    bullet(doc, "5. Is sample-level pairing over-conservative? It reduced several comparisons to bounds. The "
                "alternative is a mixed-effects model with sample as a random effect, which would recover some "
                "power. Is the bound the right call, or a loss of information?")
    bullet(doc, "6. Does the staged allocator deserve to be called joint? It is a lookup plus a two-stage "
                "ladder, not a joint optimisation. The defence is that the measurements refused a joint search "
                "space. Is that defensible in a paper whose title says 'joint'?")
    bullet(doc, "7. Section 9.5 shows protection is not separable from retention, but leaves the allocator's "
                "constants alone because the contrast that would move them has 4 discordant samples. Is "
                "shipping a stated caveat the right call, or should the allocator refuse to promise the "
                "+0.075% guard at all below some retention ratio - accepting a threshold that is not yet "
                "measured?")
    bullet(doc, f"8. Are any of the {len(WITHDRAWN_ROWS)} withdrawals in section 10 wrong in the other "
                "direction - something withdrawn that should have stood?")
    rule(doc)
    para(doc, "Generated by scripts/build_review_docx.py. Every number transcribed from docs/decisions.md "
              "(D1-D43), docs/claims.md and the run manifests in outputs/; those records are the source of "
              "truth if they disagree with this document.", size=9, italic=True, colour=GREY)


def main() -> None:
    # A placeholder that reaches the saved file is a fabricated table, so it is a build failure, not a typo.
    missing = [c for row in H3_ROWS for c in row if "PENDING" in str(c)]
    if missing:
        raise SystemExit(f"refusing to build: {len(missing)} H3 cell(s) still unfilled")
    doc = Document()
    setup(doc)
    build(doc)
    out = Path(__file__).resolve().parents[1] / "outputs" / "reports" / "RAQ-KV_technical_review.docx"
    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
