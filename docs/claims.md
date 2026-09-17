# Claim ↔ evidence matrix

Status values: `planned` → `running` → `supported` / `refuted` / `inconclusive`.
Thresholds marked **PROPOSED** must be frozen (status `FROZEN`, with date) before M2 starts; they are never
changed after looking at the data. Full rationale: `00_research_plan.md` §I.

| ID | Claim | Experiments | Figures / tables | Decision rule | Threshold status | Status |
|---|---|---|---|---|---|---|
| S0 | Measurement protocol is valid (reference for all Δ) | M0a/M0b golden, protocol tests | Setup table | Exit criteria M0 (plan §G) | n/a | running |
| H1 | Sensitivity is heterogeneous across layer / KV-head / token type at fixed bit position | M1, M2 | Layer and bit heatmaps | Top 10% cells carry ≥ 50% damage (support); ≤ 20% (refute) | PROPOSED | planned |
| H2 | Adaptive allocation beats uniform INT4 + SECDED and magnitude-only at equal total bits (±1%, metadata included) | M3 gate-2, M5 | Pareto fronts, paired table | Paired bootstrap 95% CI excludes 0; Holm-corrected McNemar p < 0.05; ≥ 2 model families | PROPOSED | planned |
| H3 | Joint score beats the best single-signal policy | M5 ablations | Ablation table | Significant at equal budget; every ablation degrades | PROPOSED | planned |
| Q48 | Corr(QuantSens, FaultSens), raw and partial on KV magnitude | M1 + M2, gate-1 | Scatter + CI | Partial ρ ≥ 0.6 shared signal; ≤ 0.3 separate signals | PROPOSED | planned |
| Q49 | Critical token types have higher task-level SDC than filler at matched position / layer / bit | M2 (RAQ-Probe), M6 | Odds-ratio plot | OR > 1 with CI excluding 1 | PROPOSED | planned |
| C4 | Token-level metrics under-estimate agent failures | M6 | Metric-disagreement plot | Weak rank correlation between KL damage and task failure | PROPOSED | planned |
