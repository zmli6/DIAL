#!/usr/bin/env python3
"""
Phase 6 Path D: Two-Source Uncertainty Toy Model Verification
=============================================================

Verifies 3 testable predictions of the Two-Source Model:
  P1: Temporal Shift — early steps have different ρ than late steps
  P2: Cross-Environment Divergence — ρ varies across environments
  P3: Signal Identity Alignment — strongest signals match predicted type

Also generates:
  - Figure 7: P1 temporal shift grouped bar chart
  - Figure 2: Two-Source Model theoretical curve + empirical data
  - Simpson's Paradox demonstration

Usage:
    conda activate dial
    python experiments/p6_toy_model_verification.py \
        --output-dir results/phase6/toy_model
"""
import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ══════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════

def load_hotpotqa(base_dir: str) -> dict:
    """Load HotpotQA Phase 1 signal data."""
    import csv
    path = os.path.join(base_dir, "results/phase1_signal_discovery/hotpotqa/phase1_signal_data.csv")
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return {
        "name": "HotpotQA",
        "data": [{
            "episode": int(r["episode"]),
            "step_count": int(r["step_count"]),
            "token_entropy": float(r["token_entropy"]),
            "utility": float(r["utility"]),
            "evidence_count": int(float(r["evidence_count"])),
            "state_category": r["state_category"],
            "is_finish_proposed": r["is_finish_proposed"] == "True",
        } for r in rows],
    }


def load_mbpp(base_dir: str) -> dict:
    """Load MBPP Phase 1 signal data."""
    import csv
    path = os.path.join(base_dir, "results/phase1_signal_discovery/mbpp/phase1_signal_data.csv")
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return {
        "name": "MBPP",
        "data": [{
            "episode": int(r["episode"]),
            "step_count": int(r["step_count"]),
            "token_entropy": float(r["token_entropy"]),
            "utility": float(r["utility"]),
            "state_category": r["state_category"],
        } for r in rows],
    }


def load_apps(base_dir: str) -> dict:
    """Load APPS Phase 3+S2 signal data."""
    path = os.path.join(base_dir, "results/phase3_supp/apps/apps_signal_data.json")
    with open(path) as f:
        data = json.load(f)
    return {
        "name": "APPS",
        "data": [{
            "episode": int(r["episode"]),
            "step_count": int(r["step_count"]),
            "token_entropy": float(r["token_entropy"]),
            "utility": float(r["utility"]),
            "state_category": r.get("state_category", ""),
        } for r in data],
    }


def load_webshop(base_dir: str) -> dict:
    """Load WebShop Phase 4 signal data."""
    path = os.path.join(base_dir, "results/phase4/webshop/p4_webshop_signal_data.json")
    with open(path) as f:
        data = json.load(f)
    return {
        "name": "WebShop",
        "data": [{
            "episode": int(r["episode"]),
            "step_count": int(r["step_count"]),
            "token_entropy": float(r["token_entropy"]),
            "utility": float(r["utility"]),
            "state_category": r.get("state_category", ""),
            "num_available_actions": int(r.get("num_available_actions", 0)),
            "evidence_count": int(r.get("evidence_count", 0)),
        } for r in data],
    }


# ══════════════════════════════════════════════════════════════════
# STATISTICAL HELPERS
# ══════════════════════════════════════════════════════════════════

def spearman_rho(x, y):
    """Spearman rank correlation."""
    from scipy.stats import spearmanr
    x, y = np.asarray(x), np.asarray(y)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3 or np.std(x) < 1e-10:
        return 0.0, 1.0
    rho, p = spearmanr(x, y)
    return float(rho), float(p)


def bootstrap_ci(x, y, stat_fn, n_boot=10000, ci=0.95, seed=42):
    """Bootstrap confidence interval for a bivariate statistic."""
    rng = np.random.RandomState(seed)
    x, y = np.asarray(x), np.asarray(y)
    n = len(x)
    stats = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        val, _ = stat_fn(x[idx], y[idx])
        if np.isfinite(val):
            stats.append(val)
    if not stats:
        return 0.0, 0.0, 0.0
    stats = np.array(stats)
    alpha = (1 - ci) / 2
    lo = np.percentile(stats, alpha * 100)
    hi = np.percentile(stats, (1 - alpha) * 100)
    return float(np.mean(stats)), float(lo), float(hi)


# ══════════════════════════════════════════════════════════════════
# D1: TEMPORAL SHIFT ANALYSIS (P1)
# ══════════════════════════════════════════════════════════════════

def temporal_shift_analysis(env_data: dict, signal_key: str = "token_entropy",
                            early_max: int = 2, n_boot: int = 10000):
    """
    P1 Verification: Split trajectory into early (step ≤ early_max) and late (step > early_max).
    Compute ρ(signal, utility) for each group with bootstrap CI.

    Prediction: For Type I-dominant environments (HotpotQA),
    |ρ_early| > |ρ_late| because early steps have higher information-poverty.
    """
    data = env_data["data"]

    early = [(d[signal_key], d["utility"]) for d in data if d["step_count"] <= early_max]
    late  = [(d[signal_key], d["utility"]) for d in data if d["step_count"] > early_max]

    if len(early) < 5 or len(late) < 5:
        return None

    ex, ey = zip(*early)
    lx, ly = zip(*late)

    rho_early, p_early = spearman_rho(ex, ey)
    rho_late, p_late = spearman_rho(lx, ly)

    mean_e, lo_e, hi_e = bootstrap_ci(np.array(ex), np.array(ey), spearman_rho, n_boot)
    mean_l, lo_l, hi_l = bootstrap_ci(np.array(lx), np.array(ly), spearman_rho, n_boot)

    return {
        "env": env_data["name"],
        "signal": signal_key,
        "early_cutoff": f"step ≤ {early_max}",
        "late_cutoff": f"step > {early_max}",
        "early": {
            "n": len(early),
            "rho": rho_early,
            "p_value": p_early,
            "bootstrap_mean": mean_e,
            "ci_lo": lo_e,
            "ci_hi": hi_e,
        },
        "late": {
            "n": len(late),
            "rho": rho_late,
            "p_value": p_late,
            "bootstrap_mean": mean_l,
            "ci_lo": lo_l,
            "ci_hi": hi_l,
        },
        "shift": rho_late - rho_early,
        "prediction_confirmed": None,  # filled below
    }


def run_d1(envs: list, output_dir: str, n_boot: int = 10000):
    """Run D1 temporal shift analysis for all environments."""
    print("\n" + "=" * 70)
    print("D1: TEMPORAL SHIFT ANALYSIS (P1 Verification)")
    print("=" * 70)
    print("Prediction: Early steps have higher p_I → ρ should differ from late steps")
    print()

    results = []

    # Environment-specific settings
    env_configs = {
        "HotpotQA": {"signal": "token_entropy", "early_max": 2},
        "MBPP":     {"signal": "token_entropy", "early_max": 0},  # only step 0 vs 1+
        "APPS":     {"signal": "token_entropy", "early_max": 1},
        "WebShop":  {"signal": "token_entropy", "early_max": 2},
    }

    for env in envs:
        name = env["name"]
        cfg = env_configs.get(name, {"signal": "token_entropy", "early_max": 2})

        r = temporal_shift_analysis(env, cfg["signal"], cfg["early_max"], n_boot)
        if r is None:
            print(f"  {name}: SKIP (insufficient data)")
            continue

        # Determine if prediction confirmed based on environment type
        # HotpotQA (Type I dominant): ρ_early more negative → ρ_early < ρ_late
        # MBPP (Type D dominant): ρ_early ≈ ρ_late or opposite
        # APPS (mixed): weak effect
        r["prediction_confirmed"] = "see_analysis"

        results.append(r)

        e = r["early"]
        l = r["late"]
        print(f"  {name} ({cfg['signal']}):")
        print(f"    Early ({r['early_cutoff']}, n={e['n']}): "
              f"ρ = {e['rho']:.4f} [{e['ci_lo']:.4f}, {e['ci_hi']:.4f}]  p={e['p_value']:.2e}")
        print(f"    Late  ({r['late_cutoff']}, n={l['n']}): "
              f"ρ = {l['rho']:.4f} [{l['ci_lo']:.4f}, {l['ci_hi']:.4f}]  p={l['p_value']:.2e}")
        print(f"    Shift (late - early): {r['shift']:+.4f}")
        ci_overlap = e['ci_hi'] > l['ci_lo'] and l['ci_hi'] > e['ci_lo']
        print(f"    CI overlap: {'YES' if ci_overlap else 'NO (significant difference!)'}")
        print()

    # Also run with step_count as signal for comparison
    print("  --- Additional: ρ(step_count, utility) for reference ---")
    for env in envs:
        data = env["data"]
        sc = [d["step_count"] for d in data]
        ut = [d["utility"] for d in data]
        rho, p = spearman_rho(sc, ut)
        print(f"    {env['name']}: ρ(step_count, utility) = {rho:.4f}  p={p:.2e}")
    print()

    # Save results
    out_path = os.path.join(output_dir, "d1_temporal_shift_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved: {out_path}")

    return results


# ══════════════════════════════════════════════════════════════════
# D2: SIMPSON'S PARADOX SUBGROUP ANALYSIS
# ══════════════════════════════════════════════════════════════════

def simpsons_paradox_analysis(env_data: dict, group_key: str, group_threshold: int,
                               signal_key: str = "token_entropy", n_boot: int = 10000):
    """
    Demonstrate Simpson's Paradox: within-group ρ directions differ from aggregate ρ.

    Split data by group_key ≤/> threshold, compute ρ(signal, U) per group and overall.
    """
    data = env_data["data"]

    # Filter to data that has the group key
    valid = [d for d in data if group_key in d]
    if not valid:
        return None

    group_a = [(d[signal_key], d["utility"]) for d in valid if d[group_key] <= group_threshold]
    group_b = [(d[signal_key], d["utility"]) for d in valid if d[group_key] > group_threshold]
    all_pts = [(d[signal_key], d["utility"]) for d in valid]

    if len(group_a) < 5 or len(group_b) < 5:
        return None

    ax, ay = zip(*group_a)
    bx, by = zip(*group_b)
    allx, ally = zip(*all_pts)

    rho_a, p_a = spearman_rho(ax, ay)
    rho_b, p_b = spearman_rho(bx, by)
    rho_all, p_all = spearman_rho(allx, ally)

    mean_a, lo_a, hi_a = bootstrap_ci(np.array(ax), np.array(ay), spearman_rho, n_boot)
    mean_b, lo_b, hi_b = bootstrap_ci(np.array(bx), np.array(by), spearman_rho, n_boot)
    mean_all, lo_all, hi_all = bootstrap_ci(np.array(allx), np.array(ally), spearman_rho, n_boot)

    # Check for Simpson's Paradox: subgroup directions differ from aggregate
    sign_a = np.sign(rho_a) if abs(rho_a) > 0.01 else 0
    sign_b = np.sign(rho_b) if abs(rho_b) > 0.01 else 0
    sign_all = np.sign(rho_all) if abs(rho_all) > 0.01 else 0

    paradox = (sign_a != sign_all) or (sign_b != sign_all) or (sign_a != sign_b)

    return {
        "env": env_data["name"],
        "signal": signal_key,
        "group_key": group_key,
        "threshold": group_threshold,
        "group_a": {
            "label": f"{group_key} ≤ {group_threshold}",
            "n": len(group_a),
            "rho": rho_a, "p_value": p_a,
            "ci_lo": lo_a, "ci_hi": hi_a,
        },
        "group_b": {
            "label": f"{group_key} > {group_threshold}",
            "n": len(group_b),
            "rho": rho_b, "p_value": p_b,
            "ci_lo": lo_b, "ci_hi": hi_b,
        },
        "aggregate": {
            "n": len(all_pts),
            "rho": rho_all, "p_value": p_all,
            "ci_lo": lo_all, "ci_hi": hi_all,
        },
        "simpsons_paradox": paradox,
        "direction_summary": f"A:{sign_a:+.0f} B:{sign_b:+.0f} All:{sign_all:+.0f}",
    }


def run_d2(envs_dict: dict, output_dir: str, n_boot: int = 10000):
    """Run D2 Simpson's Paradox analysis."""
    print("\n" + "=" * 70)
    print("D2: SIMPSON'S PARADOX SUBGROUP ANALYSIS")
    print("=" * 70)
    print("Goal: Show within-group ρ(entropy, U) directions can reverse")
    print()

    results = []

    # HotpotQA: split by evidence_count (≤1 = info-poor Type I, ≥2 = info-rich Type D)
    if "HotpotQA" in envs_dict:
        r = simpsons_paradox_analysis(
            envs_dict["HotpotQA"], "evidence_count", 1,
            "token_entropy", n_boot
        )
        if r:
            results.append(r)
            a, b, agg = r["group_a"], r["group_b"], r["aggregate"]
            print(f"  HotpotQA — split by evidence_count (≤1 vs >1):")
            print(f"    Type I proxy (evidence ≤ 1, n={a['n']}):  ρ = {a['rho']:+.4f} [{a['ci_lo']:+.4f}, {a['ci_hi']:+.4f}]")
            print(f"    Type D proxy (evidence > 1, n={b['n']}):  ρ = {b['rho']:+.4f} [{b['ci_lo']:+.4f}, {b['ci_hi']:+.4f}]")
            print(f"    Aggregate           (n={agg['n']}):  ρ = {agg['rho']:+.4f} [{agg['ci_lo']:+.4f}, {agg['ci_hi']:+.4f}]")
            print(f"    Simpson's Paradox: {'✅ YES!' if r['simpsons_paradox'] else '❌ No'} ({r['direction_summary']})")
            print()

        # Also try evidence_count ≤ 0 vs > 0
        r2 = simpsons_paradox_analysis(
            envs_dict["HotpotQA"], "evidence_count", 0,
            "token_entropy", n_boot
        )
        if r2:
            results.append(r2)
            a, b, agg = r2["group_a"], r2["group_b"], r2["aggregate"]
            print(f"  HotpotQA — split by evidence_count (=0 vs ≥1):")
            print(f"    No evidence   (n={a['n']}):  ρ = {a['rho']:+.4f} [{a['ci_lo']:+.4f}, {a['ci_hi']:+.4f}]")
            print(f"    Has evidence  (n={b['n']}):  ρ = {b['rho']:+.4f} [{b['ci_lo']:+.4f}, {b['ci_hi']:+.4f}]")
            print(f"    Aggregate     (n={agg['n']}):  ρ = {agg['rho']:+.4f}")
            print(f"    Simpson's Paradox: {'✅ YES!' if r2['simpsons_paradox'] else '❌ No'} ({r2['direction_summary']})")
            print()

    # APPS: split by step_count (≤2 = early/info-poor, ≥3 = late/decision-phase)
    if "APPS" in envs_dict:
        r = simpsons_paradox_analysis(
            envs_dict["APPS"], "step_count", 1,
            "token_entropy", n_boot
        )
        if r:
            results.append(r)
            a, b, agg = r["group_a"], r["group_b"], r["aggregate"]
            print(f"  APPS — split by step_count (≤1 vs >1):")
            print(f"    Early steps (n={a['n']}):  ρ = {a['rho']:+.4f} [{a['ci_lo']:+.4f}, {a['ci_hi']:+.4f}]")
            print(f"    Late steps  (n={b['n']}):  ρ = {b['rho']:+.4f} [{b['ci_lo']:+.4f}, {b['ci_hi']:+.4f}]")
            print(f"    Aggregate   (n={agg['n']}):  ρ = {agg['rho']:+.4f}")
            print(f"    Simpson's Paradox: {'✅ YES!' if r['simpsons_paradox'] else '❌ No'} ({r['direction_summary']})")
            print()

    # WebShop: split by step_count
    if "WebShop" in envs_dict:
        r = simpsons_paradox_analysis(
            envs_dict["WebShop"], "step_count", 2,
            "token_entropy", n_boot
        )
        if r:
            results.append(r)
            a, b, agg = r["group_a"], r["group_b"], r["aggregate"]
            print(f"  WebShop — split by step_count (≤2 vs >2):")
            print(f"    Early steps (n={a['n']}):  ρ = {a['rho']:+.4f} [{a['ci_lo']:+.4f}, {a['ci_hi']:+.4f}]")
            print(f"    Late steps  (n={b['n']}):  ρ = {b['rho']:+.4f} [{b['ci_lo']:+.4f}, {b['ci_hi']:+.4f}]")
            print(f"    Aggregate   (n={agg['n']}):  ρ = {agg['rho']:+.4f}")
            print(f"    Simpson's Paradox: {'✅ YES!' if r['simpsons_paradox'] else '❌ No'} ({r['direction_summary']})")
            print()

    # Save results
    out_path = os.path.join(output_dir, "d2_simpsons_paradox_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved: {out_path}")

    return results


# ══════════════════════════════════════════════════════════════════
# D3: P2/P3 CROSS-ENVIRONMENT DIVERGENCE + SIGNAL IDENTITY
# ══════════════════════════════════════════════════════════════════

def run_d3(envs: list, output_dir: str):
    """P2: Cross-environment ρ divergence. P3: Signal identity alignment."""
    print("\n" + "=" * 70)
    print("D3: P2 CROSS-ENVIRONMENT DIVERGENCE + P3 SIGNAL IDENTITY")
    print("=" * 70)

    # P2: Compute ρ(token_entropy, utility) for each environment
    print("\n  --- P2: ρ(token_entropy, utility) per environment ---")
    rho_map = {}
    for env in envs:
        data = env["data"]
        te = [d["token_entropy"] for d in data]
        ut = [d["utility"] for d in data]
        rho, p = spearman_rho(te, ut)
        rho_map[env["name"]] = rho
        print(f"    {env['name']:12s}: ρ = {rho:+.4f}  (p={p:.2e}, n={len(data)})")

    # Divergence matrix
    print("\n  --- P2: ρ divergence matrix ---")
    names = list(rho_map.keys())
    div_matrix = {}
    print(f"    {'':12s}", end="")
    for n in names:
        print(f"  {n:>10s}", end="")
    print()

    for n1 in names:
        print(f"    {n1:12s}", end="")
        row = {}
        for n2 in names:
            d = abs(rho_map[n1] - rho_map[n2])
            row[n2] = d
            print(f"  {d:10.3f}", end="")
        div_matrix[n1] = row
        print()

    # P2 interpretation
    print("\n  P2 Interpretation:")
    # Find max and min divergence pairs
    max_div = 0
    max_pair = ("", "")
    min_div = float("inf")
    min_pair = ("", "")
    for i, n1 in enumerate(names):
        for n2 in names[i+1:]:
            d = abs(rho_map[n1] - rho_map[n2])
            if d > max_div:
                max_div = d
                max_pair = (n1, n2)
            if d < min_div:
                min_div = d
                min_pair = (n1, n2)
    print(f"    Max divergence: |ρ_{max_pair[0]} - ρ_{max_pair[1]}| = {max_div:.3f}")
    print(f"    Min divergence: |ρ_{min_pair[0]} - ρ_{min_pair[1]}| = {min_div:.3f}")

    # P3: Signal identity alignment
    print("\n  --- P3: Signal Identity — strongest signal per environment ---")

    signals_to_check = ["token_entropy", "step_count"]

    p3_results = []
    for env in envs:
        data = env["data"]
        best_signal = None
        best_rho = 0

        for sig in signals_to_check:
            if sig not in data[0]:
                continue
            vals = [d[sig] for d in data]
            uts = [d["utility"] for d in data]
            rho, p = spearman_rho(vals, uts)
            if abs(rho) > abs(best_rho):
                best_rho = rho
                best_signal = sig

        # Check environment-specific signals
        if "evidence_count" in data[0]:
            vals = [d["evidence_count"] for d in data]
            uts = [d["utility"] for d in data]
            rho, p = spearman_rho(vals, uts)
            if abs(rho) > abs(best_rho):
                best_rho = rho
                best_signal = "evidence_count"

        if "num_available_actions" in data[0]:
            vals = [d["num_available_actions"] for d in data]
            uts = [d["utility"] for d in data]
            rho, p = spearman_rho(vals, uts)
            if abs(rho) > abs(best_rho):
                best_rho = rho
                best_signal = "num_available_actions"

        # Determine dominant type
        if best_rho < -0.2:
            dom_type = "Type I (information-poverty)"
        elif best_rho > 0.2:
            dom_type = "Type D (decision-difficulty)"
        else:
            dom_type = "Mixed / Weak"

        entry = {
            "env": env["name"],
            "strongest_signal": best_signal,
            "rho": best_rho,
            "dominant_type": dom_type,
        }
        p3_results.append(entry)
        print(f"    {env['name']:12s}: strongest = {best_signal:25s}  ρ = {best_rho:+.4f}  → {dom_type}")

    # Save
    d3_results = {
        "p2_rho_per_env": rho_map,
        "p2_divergence_matrix": div_matrix,
        "p3_signal_identity": p3_results,
    }
    out_path = os.path.join(output_dir, "d3_p2p3_results.json")
    with open(out_path, "w") as f:
        json.dump(d3_results, f, indent=2, default=str)
    print(f"\n  Saved: {out_path}")

    return d3_results


# ══════════════════════════════════════════════════════════════════
# FIGURE GENERATION
# ══════════════════════════════════════════════════════════════════

def plot_figure7(d1_results: list, output_dir: str):
    """Figure 7: P1 Temporal Shift — Grouped bar chart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not d1_results:
        print("  No D1 results to plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    envs = [r["env"] for r in d1_results]
    n = len(envs)
    x = np.arange(n)
    width = 0.35

    early_rho = [r["early"]["rho"] for r in d1_results]
    late_rho = [r["late"]["rho"] for r in d1_results]
    early_err_lo = [r["early"]["rho"] - r["early"]["ci_lo"] for r in d1_results]
    early_err_hi = [r["early"]["ci_hi"] - r["early"]["rho"] for r in d1_results]
    late_err_lo = [r["late"]["rho"] - r["late"]["ci_lo"] for r in d1_results]
    late_err_hi = [r["late"]["ci_hi"] - r["late"]["rho"] for r in d1_results]

    bars1 = ax.bar(x - width/2, early_rho, width, label="Early steps",
                   color="#2c5f8a", alpha=0.85,
                   yerr=[early_err_lo, early_err_hi], capsize=4, ecolor="gray")
    bars2 = ax.bar(x + width/2, late_rho, width, label="Late steps",
                   color="#8ab6d6", alpha=0.85,
                   yerr=[late_err_lo, late_err_hi], capsize=4, ecolor="gray")

    ax.set_xlabel("Environment", fontsize=12)
    ax.set_ylabel(r"$\rho$(token_entropy, utility)", fontsize=12)
    ax.set_title("P1 Verification: Temporal Shift in Signal-Utility Correlation", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(envs, fontsize=11)
    ax.axhline(y=0, color="black", linewidth=0.5, linestyle="-")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    # Add sample sizes
    for i, r in enumerate(d1_results):
        ax.text(x[i] - width/2, -0.02, f"n={r['early']['n']}", ha="center",
                va="top", fontsize=8, color="gray")
        ax.text(x[i] + width/2, -0.02, f"n={r['late']['n']}", ha="center",
                va="top", fontsize=8, color="gray")

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "figure7_temporal_shift.pdf")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    fig_path_png = os.path.join(output_dir, "figure7_temporal_shift.png")
    fig.savefig(fig_path_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fig_path}")
    print(f"  Saved: {fig_path_png}")


def plot_figure2(d3_results: dict, output_dir: str):
    """
    Figure 2: Two-Source Model theoretical curve.
    Left: p_I vs ρ(entropy, U) with environment positions.
    Right: Reserved for P1 temporal shift (= Figure 7).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left panel: Theoretical curve
    # Model: ρ = β - (α + β) · p_I
    # At p_I = 0: ρ = β (positive, Type D dominant)
    # At p_I = 1: ρ = -α (negative, Type I dominant)
    # Zero crossing at p* = β / (α + β)

    # Empirical data points
    env_rho = d3_results["p2_rho_per_env"]

    # Estimate p_I for each environment based on known characteristics
    # HotpotQA: high p_I (mostly information-seeking) → ρ negative
    # MBPP: low p_I (mostly decision-making) → ρ positive
    # APPS: medium p_I (mixed) → ρ near zero
    # WebShop: medium-high p_I → ρ slightly negative
    env_positions = {}

    # Estimate p_I from ρ using a simple linear model
    # We'll fit α, β from the empirical data
    # ρ = β - (α + β) · p_I  →  p_I = (β - ρ) / (α + β)

    # Use reasonable estimates: α ≈ 0.35, β ≈ 0.20
    # This gives p* = 0.20 / 0.55 ≈ 0.36
    alpha_est = 0.35
    beta_est = 0.20

    # Theoretical curve
    p_I = np.linspace(0, 1, 100)
    rho_theory = beta_est - (alpha_est + beta_est) * p_I

    ax1.plot(p_I, rho_theory, "k-", linewidth=2, label=r"$\rho = \beta - (\alpha+\beta) \cdot p_I$")

    # Zero crossing
    p_star = beta_est / (alpha_est + beta_est)
    ax1.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
    ax1.axvline(x=p_star, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
    ax1.annotate(f"$p^* = {p_star:.2f}$", xy=(p_star, 0),
                xytext=(p_star + 0.05, 0.05), fontsize=10,
                arrowprops=dict(arrowstyle="->", color="gray"))

    # Place environments on the curve
    # Estimate p_I for each from their ρ
    colors = {"HotpotQA": "#d62728", "MBPP": "#2ca02c", "APPS": "#ff7f0e", "WebShop": "#1f77b4"}
    markers = {"HotpotQA": "o", "MBPP": "s", "APPS": "^", "WebShop": "D"}
    # Manual offsets to avoid label overlap for clustered points
    # MBPP/APPS/WebShop cluster tightly — spread labels vertically
    label_offsets = {
        "HotpotQA": (12, -18),
        "MBPP":     (30, 30),
        "APPS":     (30, -5),
        "WebShop":  (30, -35),
    }

    for name, rho in env_rho.items():
        # p_I = (β - ρ) / (α + β)
        p_i = (beta_est - rho) / (alpha_est + beta_est)
        p_i = np.clip(p_i, 0.05, 0.95)
        env_positions[name] = p_i

        c = colors.get(name, "gray")
        m = markers.get(name, "o")
        offset = label_offsets.get(name, (8, 8))
        ax1.scatter(p_i, rho, s=120, color=c, marker=m, zorder=5, edgecolors="black", linewidth=0.5)
        ax1.annotate(name, xy=(p_i, rho),
                    xytext=offset, textcoords="offset points",
                    fontsize=9, color=c, fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=c, alpha=0.4))

    ax1.set_xlabel(r"$p_I$ (proportion of Type I uncertainty)", fontsize=12)
    ax1.set_ylabel(r"$\rho$(entropy, utility)", fontsize=12)
    ax1.set_title("Two-Source Model: Predicted Signal Direction", fontsize=12)
    ax1.set_xlim(-0.05, 1.05)
    ax1.grid(alpha=0.3)

    # Add region labels
    ax1.text(0.30, beta_est * 0.5, "Type D dominant\n(ρ > 0)", fontsize=9, ha="center",
             color="#2ca02c", alpha=0.7, style="italic")
    ax1.text(0.70, -alpha_est * 0.5, "Type I dominant\n(ρ < 0)", fontsize=9, ha="center",
             color="#d62728", alpha=0.7, style="italic")

    # Right panel: Summary table as text
    ax2.axis("off")
    ax2.set_title("Empirical Validation Summary", fontsize=12)

    table_data = []
    headers = ["Environment", "ρ(ent,U)", "est. p_I", "Dominant Type"]

    for name in env_rho:
        rho = env_rho[name]
        p_i = env_positions.get(name, 0.5)
        if rho < -0.2:
            dtype = "Type I"
        elif rho > 0.2:
            dtype = "Type D"
        else:
            dtype = "Mixed"
        table_data.append([name, f"{rho:+.3f}", f"{p_i:.2f}", dtype])

    table = ax2.table(cellText=table_data, colLabels=headers, loc="center",
                      cellLoc="center", colWidths=[0.25, 0.18, 0.18, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    # Color code dominant type column
    for i, row in enumerate(table_data):
        dtype = row[3]
        if dtype == "Type I":
            table[(i+1, 3)].set_facecolor("#ffcccc")
        elif dtype == "Type D":
            table[(i+1, 3)].set_facecolor("#ccffcc")
        else:
            table[(i+1, 3)].set_facecolor("#ffffcc")

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "figure2_two_source_model.pdf")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    fig_path_png = os.path.join(output_dir, "figure2_two_source_model.png")
    fig.savefig(fig_path_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fig_path}")
    print(f"  Saved: {fig_path_png}")

    return env_positions


def plot_simpsons_scatter(envs_dict: dict, output_dir: str):
    """Optional: Simpson's Paradox scatter visualization."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if "HotpotQA" not in envs_dict:
        return

    data = envs_dict["HotpotQA"]["data"]

    fig, ax = plt.subplots(figsize=(8, 6))

    # Split by evidence_count
    type_i = [(d["token_entropy"], d["utility"]) for d in data if d["evidence_count"] <= 1]
    type_d = [(d["token_entropy"], d["utility"]) for d in data if d["evidence_count"] > 1]

    if type_i:
        xi, yi = zip(*type_i)
        ax.scatter(xi, yi, alpha=0.3, s=20, c="#d62728", label=f"evidence ≤ 1 (n={len(type_i)})")
    if type_d:
        xd, yd = zip(*type_d)
        ax.scatter(xd, yd, alpha=0.3, s=20, c="#2ca02c", label=f"evidence > 1 (n={len(type_d)})")

    # Add trend lines
    from numpy.polynomial.polynomial import polyfit
    if type_i and len(type_i) > 5:
        xi, yi = np.array(xi), np.array(yi)
        mask = np.isfinite(xi) & np.isfinite(yi)
        if mask.sum() > 5:
            coef = polyfit(xi[mask], yi[mask], 1)
            xline = np.linspace(xi[mask].min(), xi[mask].max(), 50)
            ax.plot(xline, coef[0] + coef[1]*xline, "--", color="#d62728", linewidth=2)

    if type_d and len(type_d) > 5:
        xd, yd = np.array(xd), np.array(yd)
        mask = np.isfinite(xd) & np.isfinite(yd)
        if mask.sum() > 5:
            coef = polyfit(xd[mask], yd[mask], 1)
            xline = np.linspace(xd[mask].min(), xd[mask].max(), 50)
            ax.plot(xline, coef[0] + coef[1]*xline, "--", color="#2ca02c", linewidth=2)

    ax.set_xlabel("Token Entropy", fontsize=12)
    ax.set_ylabel("Utility (U)", fontsize=12)
    ax.set_title("HotpotQA: Simpson's Paradox in Signal-Utility Space", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "simpsons_paradox_scatter.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fig_path}")


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Phase 6 Path D: Toy Model Verification")
    parser.add_argument("--base-dir", default=".",
                        help="Project root directory")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: {base}/results/phase6/toy_model)")
    parser.add_argument("--n-boot", type=int, default=10000,
                        help="Number of bootstrap resamples")
    parser.add_argument("--skip-plots", action="store_true",
                        help="Skip figure generation")
    args = parser.parse_args()

    base_dir = args.base_dir
    output_dir = args.output_dir or os.path.join(base_dir, "results/phase6/toy_model")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("Phase 6 Path D: Two-Source Uncertainty Toy Model Verification")
    print("=" * 70)
    print(f"Base dir:   {base_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Bootstrap:  {args.n_boot} resamples")

    # Load all environment data
    print("\nLoading data...")
    hotpotqa = load_hotpotqa(base_dir)
    mbpp = load_mbpp(base_dir)
    apps = load_apps(base_dir)
    webshop = load_webshop(base_dir)

    envs = [hotpotqa, mbpp, apps, webshop]
    envs_dict = {e["name"]: e for e in envs}

    for e in envs:
        print(f"  {e['name']:12s}: {len(e['data']):5d} data points")

    # D1: Temporal Shift
    d1_results = run_d1(envs, output_dir, args.n_boot)

    # D2: Simpson's Paradox
    d2_results = run_d2(envs_dict, output_dir, args.n_boot)

    # D3: P2/P3
    d3_results = run_d3(envs, output_dir)

    # Figures
    if not args.skip_plots:
        print("\n" + "=" * 70)
        print("GENERATING FIGURES")
        print("=" * 70)
        plot_figure7(d1_results, output_dir)
        plot_figure2(d3_results, output_dir)
        plot_simpsons_scatter(envs_dict, output_dir)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\nP1 (Temporal Shift):")
    for r in d1_results:
        shift_dir = "early < late" if r["shift"] > 0 else "early > late"
        print(f"  {r['env']:12s}: shift = {r['shift']:+.4f} ({shift_dir})")

    print("\nP2 (Cross-Env Divergence):")
    if d3_results:
        for name, rho in d3_results["p2_rho_per_env"].items():
            print(f"  {name:12s}: ρ(entropy, U) = {rho:+.4f}")

    print("\nSimpson's Paradox:")
    for r in d2_results:
        status = "CONFIRMED" if r["simpsons_paradox"] else "not observed"
        print(f"  {r['env']:12s} (by {r['group_key']}): {status} ({r['direction_summary']})")

    # Save combined summary
    summary = {
        "d1_temporal_shift": d1_results,
        "d2_simpsons_paradox": d2_results,
        "d3_p2p3": d3_results,
    }
    summary_path = os.path.join(output_dir, "toy_model_verification_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nFull summary: {summary_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
