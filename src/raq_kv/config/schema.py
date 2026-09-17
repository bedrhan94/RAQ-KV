"""Experiment configuration schema.

All models forbid unknown keys: a misspelled field must fail loudly instead of silently falling back to a
default and corrupting a whole sweep. ``SCHEMA_VERSION`` is part of the experiment hash; bump it whenever a
change to this file alters what an existing config means.

M0a scope: model + WikiText-2 + {cache-conditioned perplexity, latency}. Quantization, ECC, faults and pruning
sections are added in their milestones (never as unused placeholders).
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


Task = Annotated[CachePPLTask | LatencyTask, Field(discriminator="kind")]


class OutputConfig(StrictModel):
    root: str = "outputs"


class ExperimentConfig(StrictModel):
    schema_version: Literal[1] = SCHEMA_VERSION
    name: str
    seed: int
    model: ModelConfig
    dataset: WikiText2Config
    task: Task
    output: OutputConfig = Field(default_factory=OutputConfig)
    notes: str = ""

    @model_validator(mode="after")
    def _check_lengths(self) -> ExperimentConfig:
        if isinstance(self.task, LatencyTask):
            corpus_tokens = self.dataset.n_chunks * (self.dataset.prefix_len + self.dataset.continuation_len + 1)
            if max(self.task.prompt_lens) > corpus_tokens:
                raise ValueError("latency prompt_lens exceed the tokens available from the configured dataset chunks")
        return self

    def identity_dict(self) -> dict[str, Any]:
        """Fields that define *what* is measured. Output location and free-text notes are excluded."""
        return self.model_dump(mode="json", exclude={"output", "notes"})
