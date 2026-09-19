# RAQ-KV

Reliability-aware joint token compression, mixed-precision KV-cache quantization and unequal error protection
for LLM agents.

- Research plan: [docs/00_research_plan.md](docs/00_research_plan.md)
- Claim ↔ evidence matrix: [docs/claims.md](docs/claims.md)
- Conventions (tensor layout, positions, bits): [docs/conventions.md](docs/conventions.md)
- Decision log: [docs/decisions.md](docs/decisions.md)

**Status:** M0 closed (2026-09-19) — golden baselines measured on GPU; see [docs/m0_exit.md](docs/m0_exit.md) and
`outputs/reports/m0_golden_report.md`. Next: M1 (KV quantization). No quantization, ECC, faults or pruning yet.

## Setup

Python 3.12, one CUDA GPU (developed on RTX 5060 Ti 16 GB, Windows 10; Linux works the same way).

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install --no-deps -e .
python scripts/check_env.py
```

## Tests

```bash
python -m pytest                    # fast unit tests, CPU, tiny random model, no downloads
python -m pytest -m slow tests/slow # real Qwen3-1.7B on CPU (float32, ~8 GB RAM)
```

## Running experiments

Experiments are defined only by YAML (`configs/experiments/`). Every resolved config gets a deterministic id;
re-running resumes and never overwrites raw results.

```bash
python scripts/run_experiment.py --config configs/experiments/m0a_golden_ppl.yaml --dry-run
python scripts/run_experiment.py --config configs/experiments/m0a_golden_ppl.yaml
python scripts/run_experiment.py --config configs/experiments/m0a_cpu_smoke.yaml --set dataset.n_chunks=2
python scripts/aggregate.py --name m0a
```

GPU experiments (`m0a_golden_ppl`, `m0a_latency`) must run on an otherwise idle GPU; `check_env.py` warns if not.

## Output layout

```
outputs/raw/<exp_id>/manifest.json   config, data fingerprint, provenance of every run
outputs/raw/<exp_id>/results.jsonl   one append-only record per unit
outputs/raw/<exp_id>/errors.jsonl    failures (the run stops at the first one)
outputs/csv/<name>_*.csv             regenerated summaries
outputs/reports/env.json             last environment check
```
