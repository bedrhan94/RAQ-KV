"""Experiment configuration schema.

All models forbid unknown keys: a misspelled field must fail loudly instead of silently falling back to a
default and corrupting a whole sweep. ``SCHEMA_VERSION`` is part of the experiment hash; bump it whenever a
change to this file alters what an existing config means.

M0 scope: model + {WikiText-2, RAQ-Probe, GSM8K} + {cache-conditioned perplexity, latency, QA generation}.
Quantization, ECC, faults and pruning sections are added in their milestones (never as unused placeholders).
Adding a new union member does not change the serialized identity of existing configs.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = 1


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelConfig(StrictModel):
    hf_id: str
    revision: str = Field(min_length=7, description="Pinned commit SHA of the model repository")
    short_name: str
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    attn_implementation: Literal["sdpa", "eager"] = "sdpa"
    device: str = "cuda"
    chat_template_kwargs: dict[str, Any] = Field(default_factory=dict)


class WikiText2Config(StrictModel):
    """Non-overlapping chunks of the WikiText-2 test split, each ``prefix_len + continuation_len + 1`` tokens."""

    name: Literal["wikitext2"] = "wikitext2"
    hf_id: str = "Salesforce/wikitext"
    subset: str = "wikitext-2-raw-v1"
    split: str = "test"
    revision: str = Field(min_length=7, description="Pinned commit SHA of the dataset repository")
    n_chunks: int = Field(gt=0)
    prefix_len: int = Field(gt=0, description="Tokens stored in the KV cache")
    continuation_len: int = Field(gt=0, description="Tokens scored through the cache")


FactType = Literal["number", "id", "date", "entity", "constraint", "tool_arg"]
ALL_FACT_TYPES: tuple[str, ...] = ("number", "id", "date", "entity", "constraint", "tool_arg")


class RAQProbeConfig(StrictModel):
    """Synthetic typed-fact retrieval probe with known token roles (see data/raq_probe.py)."""

    name: Literal["raq_probe"] = "raq_probe"
    version: Literal["v0"] = "v0"
    generator_seed: int = 0
    fact_types: tuple[FactType, ...] = ALL_FACT_TYPES  # type: ignore[assignment]
    context_lengths: tuple[int, ...] = Field(default=(512, 1024, 2048), description="Nominal cached tokens")
    depths: tuple[float, ...] = Field(default=(0.1, 0.5, 0.9), description="Nominal relative needle depth")
    groups_per_cell: int = Field(default=10, gt=0, description="Filler groups per (type, length); each has one "
                                                              "sample per depth")
    n_distractors: int = Field(default=2, ge=0)
    filler_hf_id: str = "Salesforce/wikitext"
    filler_subset: str = "wikitext-2-raw-v1"
    filler_split: str = "train"
    filler_revision: str = Field(min_length=7)
    max_new_tokens: int = Field(default=64, gt=0)

    @model_validator(mode="after")
    def _check(self) -> RAQProbeConfig:
        if any(not 0.0 <= d <= 1.0 for d in self.depths) or len(set(self.depths)) != len(self.depths):
            raise ValueError("depths must be distinct values in [0, 1]")
        if len(set(self.fact_types)) != len(self.fact_types):
            raise ValueError("fact_types must be distinct")
        return self


class GSM8KConfig(StrictModel):
    name: Literal["gsm8k"] = "gsm8k"
    hf_id: str = "openai/gsm8k"
    subset: str = "main"
    split: str = "test"
    revision: str = Field(min_length=7)
    n_samples: int = Field(default=200, gt=0)
    subset_seed: int = 0
    max_new_tokens: int = Field(default=512, gt=0)


class BFCLConfig(StrictModel):
    """BFCL tool calls inside a synthetic tool catalogue, so the cache is long enough to compress.

    Measured before building: BFCL's own AST categories cache 157-333 tokens at the median, an order of magnitude
    short of the 512-4096 this project works at, so they cannot host a retention experiment (see data/bfcl.py).
    `catalogue_sizes` is the knob that fixes it -- a real agent carries dozens of schemas, and BFCL supplies 769
    distinct ones to draw distractors from.
    """

    name: Literal["bfcl"] = "bfcl"
    version: Literal["v3"] = "v3"
    generator_seed: int = 0
    categories: tuple[Literal["simple", "multiple", "parallel", "parallel_multiple"], ...] = ("simple",)
    questions_per_category: int = Field(default=40, gt=0)
    catalogue_sizes: tuple[int, ...] = Field(default=(10, 25), description="Distractor schemas per prompt; sets "
                                                                          "the cache length")
    schema_span_chars: int = Field(default=400, gt=0, description="Fallback span width for the last schema")
    max_new_tokens: int = Field(default=96, gt=0)

    @model_validator(mode="after")
    def _catalogue_fits_the_pool(self) -> BFCLConfig:
        if max(self.catalogue_sizes) > 700:
            raise ValueError("only 769 distinct BFCL schemas exist; a catalogue near that size cannot be sampled "
                             "without near-total overlap between prompts")
        return self


Dataset = Annotated[WikiText2Config | RAQProbeConfig | GSM8KConfig | BFCLConfig, Field(discriminator="name")]


class CachePPLTask(StrictModel):
    """Cache-conditioned perplexity: every scored logit attends to the (possibly transformed) prefix cache."""

    kind: Literal["cache_ppl"] = "cache_ppl"
    store_token_nll: bool = True


class LatencyTask(StrictModel):
    """Greedy decoding with a fixed number of new tokens (EOS ignored) for TTFT/TPOT measurement."""

    kind: Literal["latency"] = "latency"
    prompt_lens: tuple[int, ...] = (512, 1024, 2048)
    gen_len: int = Field(default=64, gt=1)
    warmup: int = Field(default=2, ge=0)
    repeats: int = Field(default=5, gt=0)


class QATask(StrictModel):
    """Greedy answer generation through the cache, scored by the dataset's scorer; plus teacher-forced gold-answer
    NLL through the same cache (the primary continuous metric; EM is secondary)."""

    kind: Literal["qa_generate"] = "qa_generate"
    score_answer_nll: bool = True


class QuantSensitivityTask(StrictModel):
    """Per-region quantization sensitivity: quantize exactly one region and leave the rest in full precision,
    measuring the paired per-token NLL increase. Regions are enumerated per chunk, so the golden pass and every
    region share one prefill.

    Keys and values use their own stress specification: applying one granularity to both would confound "this
    region matters" with "this granularity is wrong for keys" (the M1 sweep shows per-token keys are catastrophic).
    """

    kind: Literal["quant_sensitivity"] = "quant_sensitivity"
    key_spec: QuantSpec = Field(default_factory=lambda: QuantSpec(bits=2, axis="per_channel"))
    value_spec: QuantSpec = Field(default_factory=lambda: QuantSpec(bits=2, axis="per_token"))
    granularity: Literal["layer_part", "head_part"] = "layer_part"
    n_chunks: int | None = Field(default=None, description="Chunks to profile; defaults to the dataset's n_chunks")
    # Null condition: the same regions quantized at a precision the sweep showed to be lossless. Their deltas are
    # the measurement's own noise band; a stress delta that does not clear it means nothing.
    # Both default to None (feature off) so that adding them did not change the identity of runs made before it
    # existed - optional features must be opt-in, see identity_dict's exclude_none.
    null_bits: int | None = Field(default=None, description="Bit width of the null condition; None disables it")
    null_regions: int | None = Field(default=None, ge=0, description="Regions sampled per chunk for the null run")


class FaultInjectionTask(StrictModel):
    """Single-bit faults in the stored KV representation, replayed from a per-chunk plan.

    Strata: ``uniform`` (any bit of any code), ``msb`` (the most significant code bit), ``metadata`` (a float16
    scale or zero point) and ``lsb_midrange`` (the LSB of a mid-range code — the *smallest possible* storage
    error, which at 4 bits is still half the tensor's mean magnitude, see D17). ``n_control`` no-fault scorings
    per chunk establish the measurement floor, which is exactly zero because the pipeline is deterministic.
    Requires a quantization section, because faults land in codes, not in bf16 tensors.
    """

    kind: Literal["fault_injection"] = "fault_injection"
    n_chunks: int | None = None
    n_uniform: int = Field(default=60, ge=0)
    n_msb: int = Field(default=20, ge=0)
    n_metadata: int = Field(default=20, ge=0)
    n_lsb: int = Field(default=20, ge=0)
    n_control: int = Field(default=2, ge=0, description="No-fault scorings; must produce exactly zero deltas")


class FaultQATask(StrictModel):
    """Faults aimed at token roles on the probe: does corrupting the KV of the answer-bearing token matter more
    than corrupting filler? Outcomes are task-level (the answer changes), not token-level.

    Strata are the probe's own span labels — ``target_value`` (the tokens holding the answer), ``distractor_value``
    (same-type, task-irrelevant), ``filler`` (everything else in the document) — plus ``metadata`` and no-fault
    controls. This is the Q49 vehicle.
    """

    kind: Literal["fault_qa"] = "fault_qa"
    n_samples: int | None = Field(default=120, description="Probe samples to use; None means all")
    # None means single-bit faults (the original behaviour), so adding this field left existing runs' ids intact.
    burst_bits: int | None = Field(default=None, ge=1, description="Bits flipped together inside one token")
    burst_scope: Literal["head", "layer_token"] | None = Field(
        default=None, description="'head' = one memory word; 'layer_token' = the token's whole row in one layer")
    n_target: int = Field(default=8, ge=0)
    n_distractor: int = Field(default=6, ge=0)
    n_filler: int = Field(default=8, ge=0)
    n_metadata: int = Field(default=4, ge=0)
    n_control: int = Field(default=1, ge=0)


class RetentionTask(StrictModel):
    """Token retention on the probe: keep a fraction of the cached tokens, chosen by an importance scorer.

    The golden answer is generated from the *unpruned* cache in the same run, so every arm carries its own
    reference and nothing is compared across experiments (the eager attention that ``window_attention`` needs
    changes ``attn_implementation``, and therefore the experiment identity, so an M0-M3 golden would not apply).
    """

    kind: Literal["retention"] = "retention"
    n_samples: int | None = Field(default=60, description="Probe samples to use; None means all")


class ProtectionArmTask(StrictModel):
    """One M3 arm at one bit error rate: does this arm's way of spending its bits survive faults better?

    The fault model is a rate over everything the arm stores -- codes, quantization metadata and its own ECC
    check bits -- so two arms holding the same number of bits are exposed on the same number of bits (D23). The
    rate is a stated condition, not a claim; what is compared is the ordering of arms at a fixed rate, and the
    rate at which that ordering changes.
    """

    kind: Literal["protection_arm"] = "protection_arm"
    n_samples: int | None = Field(default=60, description="Probe samples to use; None means all")
    ber: float | None = Field(default=None, gt=0.0, lt=1.0,
                              description="Bit error rate over the arm's declared storage")
    n_flips: int | None = Field(default=None, ge=1,
                                description="Absolute flips per trial instead of a rate (D43 matched-count arms)")
    # D49: matched exposure ACROSS PRECISIONS. `ber_basis: unpruned` handles pruning, because the unpruned cache
    # is a thing the same arm could have stored. It cannot express "the flips a 4-bit arm would take", which is
    # what a precision comparison needs: a 2-bit cache stores half the bits, so at a matched rate it takes half
    # the flips and "lower precision is safer" becomes exposure arithmetic again. A rate per ORIGINAL token is
    # the general form -- it is independent of both precision and context length, where a flat `n_flips` gives a
    # 512-token and a 4096-token sample the same exposure (the bug D43 nearly shipped).
    n_flips_per_token: float | None = Field(
        default=None, gt=0.0,
        description="Flips per ORIGINAL token, so exposure matches across precisions and context lengths (D49)")
    # Defaults to None, not to "stored", and that is load-bearing rather than stylistic: `identity_dict` uses
    # `exclude_none`, so a field with a non-None default serializes into every config that never mentions it and
    # shifts its hash. Adding this with `default="stored"` moved all 14 protection-arm identities at once, which
    # `tests/test_fault_retention_arm.py` caught. None means "stored".
    ber_basis: Literal["stored", "unpruned"] | None = Field(
        default=None,
        description="What the rate is applied to: this arm's own storage (default), or what it would store "
                    "unpruned (D43, the matched-exposure attribution arm)")
    # D61: which group a metadata flip lands in, instead of leaving it to the uniform draw. D60 ruled out two
    # mechanisms for why one family starts at the metadata hazard ceiling, and what separates the two survivors --
    # "more groups are consequential" against "the groups that matter matter more" -- is P(break | attention rank
    # of the corrupted group), which no uniform plan can report. All three default to None so `exclude_none`
    # leaves all 14 existing protection-arm identities untouched (the `ber_basis` lesson).
    fault_target: Literal["group_rank"] | None = Field(
        default=None, description="Confine each trial's flips to the key scales of one attention-ranked token "
                                  "group; None draws uniformly over all stored bits as before")
    target_group_rank: int | None = Field(
        default=None, ge=0, description="Which group to corrupt, 0 = the most-attended. Ranks past the end wrap "
                                        "to the least-attended group and the realised rank is recorded")
    rank_scorer: str | None = Field(
        default=None, description="Importance scorer supplying the ranking; needs eager attention (D5)")
    n_trials: int = Field(default=8, ge=1, description="Independent storage-wide fault draws per sample")
    n_control: int = Field(default=1, ge=0, description="No-fault scorings; must be exactly zero damage")

    @model_validator(mode="after")
    def _exactly_one_exposure(self) -> ProtectionArmTask:
        """A rate or a count, never both and never neither.

        A pruned arm stores fewer bits, so at a matched *rate* it receives proportionally fewer flips. That is
        simultaneously the confound and the finding (D43): a matched rate alone cannot separate "pruning is a
        reliability win" from "pruning concentrates risk and the exposure drop happens to cover it", and only a
        matched absolute count attributes the outcome. Both knobs therefore exist -- but an arm setting both
        would have an exposure nobody could state, so exactly one is required.
        """
        exposures = [self.ber, self.n_flips, self.n_flips_per_token]
        if sum(e is not None for e in exposures) != 1:
            raise ValueError("a protection arm needs exactly one of `ber` (a rate over its own storage), "
                             "`n_flips` (an absolute count) or `n_flips_per_token` (a count per original token); "
                             "more than one leaves the arm's exposure ambiguous, none leaves it undefined")
        if self.ber_basis == "unpruned" and self.ber is None:
            raise ValueError("`ber_basis` describes what a rate is applied to, so it needs `ber`, not `n_flips`")
        targeting = (self.fault_target, self.target_group_rank, self.rank_scorer)
        if self.fault_target is not None:
            if self.target_group_rank is None or self.rank_scorer is None:
                raise ValueError("`fault_target: group_rank` needs both `target_group_rank` (which group) and "
                                 "`rank_scorer` (what ranks them); without the scorer the rank is undefined")
            if self.n_flips is None:
                # A *rate* over the arm's whole storage cannot describe a draw confined to one group's scales:
                # flips_for_rate would return a count sized for all 57344 values per token while the draw sees
                # only that group's scales, so the arm's stated exposure and its real one would differ.
                raise ValueError("a group-targeted arm states its exposure as `n_flips` (flips inside the chosen "
                                 "group); a rate over the arm's whole storage does not describe that draw")
        elif any(x is not None for x in targeting):
            raise ValueError("`target_group_rank` and `rank_scorer` only mean something with "
                             "`fault_target: group_rank`; set it or drop them")
        return self


Task = Annotated[
    CachePPLTask | LatencyTask | QATask | QuantSensitivityTask | FaultInjectionTask | FaultQATask
    | ProtectionArmTask | RetentionTask,
    Field(discriminator="kind"),
]

_TASK_DATASETS: dict[str, tuple[str, ...]] = {
    "cache_ppl": ("wikitext2",),
    "latency": ("wikitext2",),
    "qa_generate": ("raq_probe", "gsm8k", "bfcl"),
    "quant_sensitivity": ("wikitext2",),
    "fault_injection": ("wikitext2",),
    "fault_qa": ("raq_probe",),
    "protection_arm": ("raq_probe",),
    "retention": ("raq_probe", "bfcl"),
}


class QuantSpec(StrictModel):
    """Uniform integer quantization of one KV part (keys or values)."""

    bits: Literal[2, 3, 4, 8] = 4
    axis: Literal["per_tensor", "per_head", "per_channel", "per_token", "group_channel", "group_token"] = "per_token"
    group_size: int = Field(default=64, gt=0)
    symmetric: bool = False


class QuantizationConfig(StrictModel):
    """Keys and values are configured separately (KIVI: per-channel keys, per-token values).

    ``None`` for a part leaves it in full precision; ``residual_window`` keeps the most recent tokens unquantized.
    """

    keys: QuantSpec | None = None
    values: QuantSpec | None = None
    residual_window: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _check(self) -> QuantizationConfig:
        if self.keys is None and self.values is None:
            raise ValueError("quantization section with neither keys nor values; omit the section instead")
        return self


class ProtectionConfig(StrictModel):
    """Which codec guards which part of the stored representation.

    Codes and quantization metadata are configured separately because M2 found their damage per bit differs by
    ~3000x (D21), and ``data_layers`` expresses a layer-selective policy so the fault-sensitivity map can be
    tested as a protection policy rather than only described.
    """

    data_codec: Literal["none", "parity", "secded"] = "none"
    data_word_bits: int = Field(default=64, gt=1)
    data_layers: tuple[int, ...] | None = Field(default=None, description="None protects every layer")
    # A plausibility check, not a code: two float16 bounds per metadata region (about 3.5 kbit against 596 Mbit)
    # against which every scale and zero point is validated on read. It exists because the measured mechanism of
    # metadata damage is magnitude explosion, not arbitrary corruption: a float16 scale's bit 14 multiplies it by
    # 65536 and bit 13 divides it by 250 (D27), and a range check catches exactly those without any ECC.
    # Defaults to None, not "none": a new field with a non-None default would change the identity
    # hash of every stored run that has a protection section (tests/test_raw_store_integrity.py).
    metadata_guard: Literal["range"] | None = None
    metadata_codec: Literal["none", "parity", "secded"] = "none"
    metadata_word_bits: int = Field(default=16, gt=1)
    due_policy: Literal["erasure_mid", "pass_through"] = "erasure_mid"

    @model_validator(mode="after")
    def _check(self) -> ProtectionConfig:
        if self.data_codec == "none" and self.metadata_codec == "none" and self.metadata_guard is None:
            raise ValueError("protection section with no codec and no guard; omit the section instead")
        return self


class CompressionConfig(StrictModel):
    """Which cached tokens to keep, and who decides (M4).

    ``ratio`` is a fraction of the ORIGINAL cached tokens, never of some already-reduced count: after pruning,
    bits per retained token improves for free and would flatter every policy. ``protect_recent`` keeps a tail
    unconditionally, as KIVI, H2O and SnapKV all do, so a scorer is not judged on whether it rediscovers recency.
    """

    scorer: Literal["recency", "key_norm", "value_norm", "window_attention", "query_attention",
                    "oracle_keep_target", "oracle_keep_sentence", "oracle_keep_random_span",
                    "oracle_drop_target"] = "recency"
    ratio: float = Field(default=1.0, gt=0.0, le=1.0, description="Share of original cached tokens kept")
    protect_recent: int = Field(default=0, ge=0, description="Most recent tokens always kept")
    # Position 0 is an attention sink: dropping it alone fails every answer on this probe (D32). Protecting an
    # initial block is what StreamingLLM and every practical method do, so it is the default rather than a flag
    # each config must remember.
    protect_initial: int = Field(default=4, ge=0, description="Initial tokens always kept (attention sink)")
    window: int = Field(default=64, gt=0, description="Observation window for window_attention")
    # Token or span. D34 measured that the unit matters more than the score: at 25% retention a perfect
    # token-level oracle failed 56% of questions and keeping the fact's own sentence failed 15%. Defaults to
    # None so that adding it left existing retention runs' identities intact.
    # "schema" is M6's unit: a tool catalogue is JSON, so punctuation-delimited sentence spans are meaningless
    # in it, and the natural all-or-nothing block is one function schema. It is the agent-shaped form of D34 --
    # the unit the vehicle's own structure supplies, rather than one inferred from text.
    unit: Literal["token", "span", "schema"] | None = None
    span_aggregate: Literal["mean", "sum", "max"] | None = None


class OutputConfig(StrictModel):
    root: str = "outputs"


class ExperimentConfig(StrictModel):
    schema_version: Literal[1] = SCHEMA_VERSION
    name: str
    seed: int
    model: ModelConfig
    dataset: Dataset
    task: Task
    quantization: QuantizationConfig | None = None
    protection: ProtectionConfig | None = None
    compression: CompressionConfig | None = None
    output: OutputConfig = Field(default_factory=OutputConfig)
    notes: str = ""

    @model_validator(mode="after")
    def _check_lengths(self) -> ExperimentConfig:
        if self.dataset.name not in _TASK_DATASETS[self.task.kind]:
            raise ValueError(f"task '{self.task.kind}' does not support dataset '{self.dataset.name}'")
        if isinstance(self.task, RetentionTask) and self.compression is None:
            raise ValueError("a retention task needs a compression section; without one it is an M0b rerun")
        if isinstance(self.task, FaultInjectionTask | FaultQATask):
            if self.quantization is None:
                raise ValueError(f"{self.task.kind} needs a quantization section: faults land in stored codes")
            if self.quantization.residual_window:
                raise ValueError(f"{self.task.kind} does not support a residual window yet")
        if self.protection is not None and self.quantization is None:
            raise ValueError("protection guards the quantized representation, so quantization must be configured")
        # A `compression` block on a task that never applies one would be worse than ignored: it reaches
        # `_pruning_token`, so the run would be named `..._p25qaS_...` and hashed as pruned while executing
        # unpruned -- a manifest that is internally consistent and wrong (D42). Only the tasks that actually
        # prune may carry the block, and a task that learns to prune is added here deliberately.
        #
        # `protection_arm` joined this list in D43, when it gained a pruning step so that faults and retention
        # could be varied together for the first time. A config with `compression is None` still renders `p0`,
        # so the identities of every arm that ran before that change are unaffected -- checked, not assumed.
        if self.compression is not None and not isinstance(self.task, RetentionTask | ProtectionArmTask):
            raise ValueError(
                f"{self.task.kind} does not apply a compression block, but one is configured. The block would "
                "still enter the experiment id and the config hash, so the run would claim a pruning it never "
                "performed. Remove it, or implement pruning for this task kind and allow it here."
            )
        if isinstance(self.task, LatencyTask) and isinstance(self.dataset, WikiText2Config):
            corpus_tokens = self.dataset.n_chunks * (self.dataset.prefix_len + self.dataset.continuation_len + 1)
            if max(self.task.prompt_lens) > corpus_tokens:
                raise ValueError("latency prompt_lens exceed the tokens available from the configured dataset chunks")
        return self

    def identity_dict(self) -> dict[str, Any]:
        """Fields that define *what* is measured. Output location and free-text notes are excluded.

        ``exclude_none`` keeps a config's identity stable when a later milestone adds an optional section: a run
        without quantization serializes exactly as it did before that section existed, so experiment ids (and the
        raw results already stored under them) stay valid.
        """
        return self.model_dump(mode="json", exclude={"output", "notes"}, exclude_none=True)
