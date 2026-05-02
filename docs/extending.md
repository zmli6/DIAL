# Extending DIAL

This file documents the two extension points: implementing a new gate
(to evaluate your own method against ours) and adding a new environment.

## Plugging in a new gate

### `GateInterface` reference

```python
from dial.benchmark import GateInterface, Decision

class MyGate(GateInterface):
    name = "my_method"  # used as column name in reports

    def setup(self, env_name, backbone, explore_data=None):
        """Called once per (env, seed). Optional."""

    def should_rollout(self, state, signals) -> Decision:
        """Required. Decide whether to invoke the test-time optimizer."""
        ...

    def update(self, state, signals, utility):
        """Called after rollouts; for online-learning gates. Optional."""

    def finalize(self) -> dict:
        """Diagnostics included in the per-env report. Optional."""
        return {}
```

### Universal signals always available

| Key | Type | Definition |
|---|---|---|
| `token_entropy` | float | Entropy of the next-token distribution at the current state |
| `action_entropy` | float | Entropy over candidate actions returned by the proposer |
| `step_count` | int | Step index `t` in the episode |
| `state_length` | int | Token count of the state text |
| `num_avail_actions` | int | Size of the candidate-action set, or 0 if unknown |

### Env-specific signals

Adapters may attach additional signals via `env.last_signals` (a dict of
floats). Currently exposed:

- HotpotQA: `evidence_count`, `has_search_result`
- WebShop: `cart_size`, `is_product_page`, `num_attributes`
- Plancraft: `inventory_size`, `has_target_resource`

Your `should_rollout` will receive these alongside the universal pool;
just check the dict for keys you care about and ignore the rest.

### Pre-collected explore cache

DIAL's randomized 50-episode exploration phase produces a dataset of
`(signals, utility)` records per (env, backbone). To run calibration-based
methods (Platt scaling, isotonic, probes) without re-doing this 50-episode
cost, download the cache:

```bash
bash scripts/data/download_explore_cache.sh
# → paper_results/explore_cache/{hotpotqa,apps,...}_qwen3-4b.json
```

Then run with `--explore-data-dir`:

```bash
python -m dial.benchmark --gate my.py:MyGate \
    --explore-data-dir paper_results/explore_cache/
```

The cache is delivered to `setup()` as a list of dicts:

```python
[{"signals": {"token_entropy": 0.8, "step_count": 3, ...},
  "utility": 1.0,
  "episode": 0, "step": 3,
  "env": "hotpotqa"},
 ...]
```

### Cost accounting

If your gate issues extra LLM calls (vote, confidence query, ...), the
harness cannot see them. Either:

1. Use `dial.benchmark.cost.TokenLedger` from inside `should_rollout` (recommended), or
2. Override `finalize()` to return `{"extra_tokens_per_episode": N}`; the harness will add `N × episodes` to the method-token total before computing `cost / base_only`.

Without this, your reported cost will under-count gate overhead and the comparison will be unfair to methods that pay their cost honestly.

---

## Adding a new environment

1. Create `dial/envs/my_env.py` subclassing `BaseEnv`:

```python
from dial.envs.base import BaseEnv, EnvState

class MyEnv(BaseEnv):
    ENV_TYPE = "my_env"

    def reset(self, seed=None):
        ...
        return obs

    def step(self, action):
        ...
        return obs, reward, terminated, truncated, info

    def get_state(self) -> EnvState:
        ...

    def set_state(self, state: EnvState):
        # Required for paired counterfactual rollouts in the explore phase.
        ...

    def forward_value(self, action) -> float:
        # Cheap per-action heuristic (used as a feature when entropy isn't enough).
        ...

    def greedy_oracle_action(self):
        # Ground-truth or near-optimal action; used for utility labelling.
        ...

    def is_success(self, reward, terminated, info) -> bool:
        ...
```

2. Register it:

```python
# dial/envs/registry.py
ENV_REGISTRY["my_env"] = "dial.envs.my_env.MyEnv"
```

3. Add a config under `configs/envs/my_env.yaml` so users have a known-good
   set of hyperparameters to start from.

4. (Optional) Expose env-specific features by setting `self.last_signals`
   in your `step()` implementation.

After this, both `python -m dial.benchmark --envs my_env ...` and
`python experiments/run_dial.py --env my_env ...` will work.
