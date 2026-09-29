# RAQ-KV

Reliability-aware joint token compression, mixed-precision KV-cache quantization
and unequal error protection for LLM agents.

A key-value cache can be shrunk along three axes at once: how many token
positions are kept, how many bits each stored value carries, and how many bits
are spent protecting the representation against memory faults. This repository
holds the measurement harness that treats the three as one allocation problem,
the configuration of every experiment run, and the decision record that ties
each claim to the run behind it.

**Status.** Milestones M0 to M6 are closed. 88 numbered decision records
(`docs/decisions.md`), 21 registered claims with verdicts (`docs/claims.md`),
178 experiment configurations, 293 test functions across 32 files.

## What is measured, and what is not

- **Quantization is simulated.** Codes are reconstructed to the model's working
  precision before use, so accuracy reflects the codec and not a kernel.
  Storage is computed analytically from the declared representation, counting
  codes, scales, zero-points and any check bits.
- **No latency or throughput numbers.** Nothing here supports a wall-clock or
  realised-memory-traffic claim, and none is made. A deployment study would
  need real packed kernels.
- **Faults are single-bit upsets** drawn uniformly over the bits an arm
  actually stores, so a region's share of faults equals its share of storage.
- **Two phases, and never `model.generate()` on a modified cache.** A prompt is
  split into a cached part and a query part; the cached part is prefilled, the
  cache captured, the transform applied to that captured cache, and the query
  scored or decoded through it. Calling a model's own generation helper on a
  modified cache rebuilds state and silently undoes part of the transform.

## Environment

Python 3.11 or newer (locked at 3.12.10), one CUDA GPU. Developed on an
RTX 5060 Ti 16 GB under Windows 10; Linux works the same way. CPU-only runs
are enough for the fast tests and small smoke configurations, not for the GPU
arms.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install --no-deps -e .
python scripts/check_env.py
```

`requirements.txt` is an exact `pip freeze` lock, including
`torch==2.11.0+cu128` and `transformers==5.17.0`, with the PyTorch CUDA 12.8
index declared inline. `pyproject.toml` carries loose lower bounds for
development; the lock is what reproduces a run.

Model weights and datasets are downloaded from the Hugging Face Hub on first
use and are not committed here. The models used are Qwen3-1.7B, Qwen3-4B and
Llama-3.2-3B. The Llama weights require accepting their licence on the Hub.

## Reproducing a result

Every claim is traceable through the same chain:

1. `docs/claims.md` states the claim, its verdict, and the decision that closed
   it.
2. `docs/decisions.md` gives that decision's numbers, the arms compared, and
   the configurations behind them.
3. `configs/experiments/<name>.yaml` is the run. Experiments are defined only
   by YAML; nothing about an arm lives on a command line.

```bash
python scripts/run_experiment.py --config configs/experiments/<name>.yaml --dry-run
python scripts/run_experiment.py --config configs/experiments/<name>.yaml
python scripts/aggregate.py --name <sweep-name>
```

`--dry-run` resolves the config, prints its identity id, and touches nothing.
The `scripts/report_*.py` scripts turn raw records into the summaries quoted in
the decision record.

## Determinism and provenance

- **Every resolved configuration gets a deterministic identity hash.** Every
  non-`None` field enters that hash, so a new field must default to `None` or
  it re-keys every existing run.
- **Runs resume and never overwrite.** Re-running an experiment continues from
  the records already on disk. A run is retired with
  `scripts/supersede_run.py`, which renames and marks it; nothing is deleted.
- **Each run records its own provenance** in `manifest.json`: the resolved
  config, a fingerprint of the data it consumed, library versions, and a hash
  over the contents of `src/raq_kv`, so a code change mid-campaign cannot pass
  unnoticed.

## A reproducibility caveat worth knowing before you re-run anything

**Llama-3.x chat templates write the current date into the prompt.** The
template calls `strftime_now` and emits a `Today Date:` line, so every Llama
run here caches a prompt containing the date it was run on, then quantizes,
faults and scores that cache. No Llama arm is byte-identical when re-run on a
different day. Qwen's template contains nothing of the kind and renders no
date.

Comparisons within one campaign are unaffected, because those arms share a
date, and that is how every result here is built. Where the effect could be
checked directly it changed 2 predictions in 120.
`tests/test_prompt_date_stability.py` holds the invariant under
`xfail(strict=True)`. Pinning the date is a one-line change that re-keys the
stored runs, which is why it was not adopted. See D79 in `docs/decisions.md`.

## Statistical conventions

Applied throughout, and worth knowing before reading any number here:

- Outcomes are **paired per sample and never pooled across trials**, because
  trials within a sample share a prompt.
- A difference counts as established only if **p < 0.05 and the interval
  excludes zero**. Intervals are 10,000-resample bootstrap intervals on paired
  differences, or Wilson intervals for a single proportion.
- **Two arms that both sit at zero are a bound, not an equality**, reported
  with the rule of three.
- Where several related contrasts are tested, Holm's correction is applied.

## Tests

```bash
python -m pytest                     # fast, CPU, tiny random model, no downloads
python -m pytest -m slow tests/slow  # real Qwen3-1.7B on CPU (float32, ~8 GB RAM)
```

GPU-gated tests skip automatically when no CUDA device is present.

## Output layout

Raw experiment outputs are **not** committed. The configurations and the
recorded identity hashes are what reproduce them.

```
outputs/raw/<exp_id>/manifest.json   config, data fingerprint, provenance
outputs/raw/<exp_id>/results.jsonl   one append-only record per unit
outputs/raw/<exp_id>/errors.jsonl    failures (a run stops at the first one)
outputs/csv/<name>_*.csv             regenerated summaries
outputs/reports/env.json             last environment check
```

## Repository layout

```
src/raq_kv/           harness: quantization, faults, retention, allocation, analysis
configs/datasets/     vehicle definitions (17)
configs/experiments/  experiment arms (178)
scripts/              run, aggregate, and per-result analysis scripts
docs/                 research plan, conventions, claims, decision record, prereg
tests/                32 files, 293 test functions
```

Key documents: [research plan](docs/00_research_plan.md),
[conventions](docs/conventions.md) (tensor layout, positions, bit accounting),
[claims](docs/claims.md), [decision record](docs/decisions.md),
[preregistrations](docs/preregistrations/).

## Licence

MIT, see [LICENSE](LICENSE).
