# DIAL Benchmark Leaderboard

Community comparison of adaptive-gating methods evaluated under the
canonical `dial.benchmark` harness — same envs, same seeds, same cost
accounting (`dial.benchmark.compute_cost`).

## Qwen3-4B-Instruct-2507, 100 episodes × 3 seeds

| Method | HotpotQA | WebShop | FEVER | APPS | TWExpress | Plancraft | Mean SR | Mean Cost |
|---|---|---|---|---|---|---|---|---|
| **DIAL (this repo)**       | **95.2 / 8.02×** | **43.8 / 2.50×** | **49.8 / 16.51×** | **73.0 / 3.84×** | **99.0 / 1.10×** | **23.3 / 1.04×** | **64.0** | **5.50×** |
| always_trigger (bound)     | 97.0 / 10.63× | 43.0 / 5.56× | 50.5 / 27.30× | 76.5 / 5.20× | 99.5 / 1.45× | 22.8 / 1.78× | 64.9 | 8.65× |
| base_only (bound)          | 49.0 / 1.00× | 42.5 / 1.00× | 37.0 / 1.00× | 60.5 / 1.00× | 96.0 / 1.00× | 29.8 / 1.00× | 52.5 | 1.00× |
| _Your method here →_       | _PR welcome_ | | | | | | | |

Format: `SR (%) / Cost (×base)`. Lower cost and higher SR are better.

## How to submit

1. Implement `dial.benchmark.GateInterface` for your method (or run your
   existing repo and convert its outputs to the `dial.benchmark.schema`
   JSON format).
2. Run all 6 envs × 3 seeds × ≥100 episodes per cell, with the canonical
   `dial.benchmark.compute_cost` (no custom cost accounting).
3. Open a PR adding a row to the table above and the produced
   `report.json` under `paper_results/community/<method>.json`.

Submissions must:
- Use the canonical `compute_cost` (no custom cost accounting).
- Run all 6 envs × 3 seeds × ≥100 episodes per cell.
- Disclose any extra training compute beyond the standard 50-episode
  explore phase, in the PR description.
- Include a link to the source code that produced the numbers.

We will spot-check submissions by re-running the gate on a held-out seed
when the source is `GateInterface`-compatible.
