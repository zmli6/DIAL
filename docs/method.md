# DIAL: Method overview

This is a self-contained summary of the algorithm. For the formal
treatment (proposition + proof of necessity) see the paper.

## Problem

At each step `t` an LLM agent observes state `s_t`, samples action
`a_t ~ π(s_t)` from a base policy, and transitions. An optional
**test-time optimizer** `T` (rollouts, K-variant voting, ...) can be
invoked at extra cost to produce a (hopefully better) action `T(s_t)`.

The optimizer's **utility** at state `s` is

```
U(T, s) = E[R(τ) | a = T(s)] - E[R(τ) | a = π(s)]
```

The **adaptive gating problem** is to learn `g: S → {0,1}` such that
`g(s) = 1` only when `U(T, s) > 0`. A perfect gate captures all the
positive-utility steps and skips the rest.

## Why fixed-direction gating fails

Existing methods all assume that **higher uncertainty consistently
indicates U > 0**. Our measurement shows this assumption fails:

- The same uncertainty signal flips sign across environments.
- For a fixed environment, the sign can flip across LLM backbones.

We trace this to two coexisting state types:

- **Type I (information-poor):** rollouts can't synthesize evidence
  the agent does not have. High σ → U < 0.
- **Type D (decision-difficult):** rollouts can sample multiple viable
  options and find the best. High σ → U > 0.

Real environments mix the two. The marginal correlation
`ρ(σ, U) ≈ β − (α + β) p_I` depends on the mixture `p_I`, which
in turn depends on what the *backbone* already encodes. So the same
σ value can mean opposite things in different (env, backbone) cells.

**Implication:** any gate that maps σ to a fixed direction must
underperform `base_only` in some (env, backbone) cell.

## DIAL: three stages

```
                Explore               Reason              Learn
              ───────────           ───────────        ───────────
              ε-Bernoulli            LLM proposes       ℓ1-logistic
              random trigger    →    task-specific  →   over univ +
              + paired counter-      features +         LLM features
              factual rollouts       universal pool     → signed weights
```

### 1. Explore (signal-agnostic)

For 50 episodes, at each step trigger T with probability ε=0.5
**independently of any signal**. For each triggered step, fork the
env, run T's action and the base action from the same state, label
which yields higher return. Aggregate `(s, U)` pairs into dataset D.

The Bernoulli trigger eliminates selection bias; the paired design
eliminates baseline bias. The remaining approximation is the rollout
horizon H, treated as a hyperparameter.

### 2. Reason (feature construction)

The candidate feature pool is

```
φ_cand(s) = φ_universal(s) ∪ φ_LLM(s)
```

Universal features (always available, per Appendix `app:universal-features`):

- `step_count` — episode step index t
- `token_entropy` — entropy of next-token distribution
- `action_entropy` — entropy over candidate actions
- `state_length` — token count of state text
- (env-specific, e.g. `num_avail_actions` in WebShop)

LLM features are produced by giving an LLM a structured summary of D
and asking it to write a Python function that extracts 5 floats from
state text. The `\ell_1` penalty in stage 3 automatically discards
useless proposals, so the LLM can propose freely.

### 3. Learn (sparse linear gate)

Standardize features, then fit:

```
w*, b* = argmin Σ ℓ(σ(w·φ + b), U) + (1/C) ||w||_1
```

with C selected by 5-fold CV over `{0.01, 0.03, 0.1, 0.3, 1, 3, 10}`,
optimizing held-out log-loss. Drop features with `w_i* = 0`.

The deployed gate is

```
g(s) = 1 if σ(w*ᵀ φ(s) + b*) > 0.5 else 0
```

— one feature evaluation and one sigmoid per step. Zero LLM overhead
at deployment. The signed weights `w_i*` are interpretable:

- `w_i > 0`: this signal is **Type-D-leaning** in this (env, backbone)
- `w_i < 0`: this signal is **Type-I-leaning** in this (env, backbone)
- `w_i ≈ 0`: signal carries no information here

## Online adaptation (optional)

For drifting environments, set `gate.epsilon_decay = True`. The gate
runs an `ε: 0.1 → 0` greedy override during deployment and refits the
sparse classifier every 30 episodes on the accumulated samples.

This is *not* free — the override means a fraction of overridden steps
in Type-I environments will trigger harmful rollouts. For stationary
environments (the setting of the paper's main results) we recommend
skipping this and relying on the offline-fitted gate.

## Wrong-direction ablation

To verify that direction is the load-bearing piece (and not gate
capacity), we trained DIAL normally, then flipped the sign of all
weights `w* → -w*`. On strong-signal environments this collapses SR
by 23–37 points; on weak-signal Plancraft it has negligible effect.
Damage scales with `|ρ(σ, U)|`, confirming that wrong direction is a
structural failure rather than a calibration bug.
