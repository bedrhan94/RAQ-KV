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


Dataset = Annotated[WikiText2Config | RAQProbeConfig | GSM8KConfig, Field(discriminator="name")]


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


Task = Annotated[CachePPLTask | LatencyTask | QATask | QuantSensitivityTask, Field(discriminator="kind")]

_TASK_DATASETS: dict[str, tuple[str, ...]] = {
    "cache_ppl": ("wikitext2",),
    "latency": ("wikitext2",),
    "qa_generate": ("raq_probe", "gsm8k"),
    "quant_sensitivity": ("wikitext2",),
}


class QuantSpec(StrictModel):
    """Uniform integer quantization of one KV part (keys or values)."""

    bits: Literal[2, 3, 4, 8] = 4
    axis: Literal["per_tensor", "per_head", "per_channel", "per_token", "group_channel"] = "per_token"
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
    output: OutputConfig = Field(default_factory=OutputConfig)
    notes: str = ""

    @model_validator(mode="after")
    def _check_lengths(self) -> ExperimentConfig:
        if self.dataset.name not in _TASK_DATASETS[self.task.kind]:
            raise ValueError(f"task '{self.task.kind}' does not support dataset '{self.dataset.name}'")
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
