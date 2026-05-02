#!/usr/bin/env python3
"""
Phase 1: Signal Discovery — Multi-Indicator Analysis
=====================================================

Runs the full analysis pipeline on collected Phase 1 data:

1. For each (signal, environment) pair, compute 5 correlation metrics:
   - Pearson r          (linear)
   - Spearman ρ         (monotone non-linear)
   - Mutual Information  (arbitrary non-linear, incl. U-shape)
   - Piecewise Pearson r (split at median, detects U-shape)
   - η² (Eta-squared)   (for categorical signals)

2. Rank signals by MI across environments.

3. Scatter plots + LOWESS curves for top signals.

4. Signal Comparison Matrix (the key Phase 1 deliverable).

5. Go / No-Go decision.

Usage:
    python experiments/phase1_analysis.py \\
        --data-dir results/phase1_signal_discovery \\
        --config configs/phase1_signal_discovery.yaml

    # Only generate plots (skip heavy MI computation):
    python experiments/phase1_analysis.py \\
        --data-dir results/phase1_signal_discovery --plots-only
"""
import argparse
import json
import math
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from dial.utils.legacy import NumpyEncoder

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ══════════════════════════════════════════════════════════════════
# CORRELATION METRICS
# ══════════════════════════════════════════════════════════════════

def pearson_r(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Pearson correlation coefficient with p-value."""
    from scipy.stats import pearsonr
    if len(x) < 3 or np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return 0.0, 1.0
    r, p = pearsonr(x, y)
    return float(r), float(p)


def spearman_rho(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Spearman rank correlation with p-value."""
    from scipy.stats import spearmanr
    if len(x) < 3:
        return 0.0, 1.0
    rho, p = spearmanr(x, y)
    return float(rho), float(p)


def mutual_information(x: np.ndarray, y: np.ndarray, n_neighbors: int = 5) -> float:
    """
    Mutual information estimated via k-NN (Kraskov estimator).

    Returns MI in nats (natural units).
    """
    from sklearn.feature_selection import mutual_info_regression
    if len(x) < 10:
        return 0.0
    x_2d = x.reshape(-1, 1)
    # mutual_info_regression returns MI in nats
    mi = mutual_info_regression(
        x_2d, y,
        n_neighbors=min(n_neighbors, len(x) - 1),
        random_state=42,
    )
    return float(mi[0])


def piecewise_pearson(x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """
    Split data at median of x and compute Pearson r in each half.

    This detects U-shape: if r_left < 0 and r_right > 0 (or vice versa),
    the relationship is non-monotone.
    """
    if len(x) < 6:
        return {
            "left_r": 0.0, "left_p": 1.0, "left_n": 0,
            "right_r": 0.0, "right_p": 1.0, "right_n": 0,
            "shape": "insufficient_data",
        }

    median_x = np.median(x)
    left_mask = x <= median_x
    right_mask = x > median_x

    left_r, left_p = pearson_r(x[left_mask], y[left_mask])
    right_r, right_p = pearson_r(x[right_mask], y[right_mask])

    # Classify shape
    if abs(left_r) < 0.1 and abs(right_r) < 0.1:
        shape = "flat"
    elif left_r < -0.1 and right_r > 0.1:
        shape = "U-shape"
    elif left_r > 0.1 and right_r < -0.1:
        shape = "inverse-U"
    elif left_r > 0 and right_r > 0:
        shape = "monotone_increasing"
    elif left_r < 0 and right_r < 0:
        shape = "monotone_decreasing"
    else:
        shape = "mixed"

    return {
        "left_r": float(left_r),
        "left_p": float(left_p),
        "left_n": int(np.sum(left_mask)),
        "right_r": float(right_r),
        "right_p": float(right_p),
        "right_n": int(np.sum(right_mask)),
        "split_point": float(median_x),
        "shape": shape,
    }


def eta_squared(categories: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Eta-squared (η²) — effect size for categorical predictor.

    η² = SS_between / SS_total
    Also returns F-test p-value.
    """
    from scipy.stats import f_oneway
    unique_cats = np.unique(categories)
    if len(unique_cats) < 2 or len(y) < 3:
        return 0.0, 1.0

    groups = [y[categories == c] for c in unique_cats if np.sum(categories == c) > 0]
    groups = [g for g in groups if len(g) > 0]

    if len(groups) < 2:
        return 0.0, 1.0

    # SS_total
    grand_mean = np.mean(y)
    ss_total = np.sum((y - grand_mean) ** 2)
    if ss_total < 1e-10:
        return 0.0, 1.0

    # SS_between
    ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
    eta2 = float(ss_between / ss_total)

    # F-test
    try:
        f_stat, p_val = f_oneway(*groups)
        p_val = float(p_val)
    except Exception:
        p_val = 1.0

    return eta2, p_val


# ══════════════════════════════════════════════════════════════════
# ANALYSIS PIPELINE
# ══════════════════════════════════════════════════════════════════

# Continuous signals (Pearson, Spearman, MI, Piecewise)
CONTINUOUS_SIGNALS = ["step_count", "token_entropy", "evidence_count", "test_pass_rate"]
# Categorical signals (η², MI via encoding)
CATEGORICAL_SIGNALS = ["state_category", "action_type"]


def analyze_single_env(
    df: pd.DataFrame,
    env_name: str,
    analysis_cfg: Dict,
) -> Dict[str, Any]:
    """
    Run full multi-indicator analysis for one environment.

    Returns per-signal results dict.
    """
    results = {}
    y = df["utility"].values

    mi_k = analysis_cfg.get("mi_n_neighbors", 5)

    # ── Continuous signals ──
    for sig in CONTINUOUS_SIGNALS:
        if sig not in df.columns:
            continue
        vals = df[sig].dropna()
        if len(vals) < 10:
            continue

        x = vals.values.astype(float)
        y_sig = df.loc[vals.index, "utility"].values

        r, r_p = pearson_r(x, y_sig)
        rho, rho_p = spearman_rho(x, y_sig)
        mi = mutual_information(x, y_sig, n_neighbors=mi_k)
        pw = piecewise_pearson(x, y_sig)

        results[sig] = {
            "type": "continuous",
            "n": len(x),
            "pearson_r": r,
            "pearson_p": r_p,
            "spearman_rho": rho,
            "spearman_p": rho_p,
            "mutual_information": mi,
            "piecewise": pw,
            "shape": pw["shape"],
        }

    # ── Categorical signals ──
    for sig in CATEGORICAL_SIGNALS:
        if sig not in df.columns:
            continue
        vals = df[sig].dropna()
        if len(vals) < 10:
            continue

        cats = vals.values
        y_sig = df.loc[vals.index, "utility"].values

        eta2, eta2_p = eta_squared(cats, y_sig)

        # MI via label encoding
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        x_encoded = le.fit_transform(cats).astype(float)
        mi = mutual_information(x_encoded, y_sig, n_neighbors=mi_k)

        # Per-category breakdown
        unique_cats = np.unique(cats)
        per_cat = {}
        for c in unique_cats:
            mask = cats == c
            cu = y_sig[mask]
            per_cat[str(c)] = {
                "count": int(np.sum(mask)),
                "mean_utility": float(np.mean(cu)),
                "std_utility": float(np.std(cu)),
                "positive_ratio": float(np.mean(cu > 0)),
            }

        results[sig] = {
            "type": "categorical",
            "n": len(cats),
            "eta_squared": eta2,
            "eta_squared_p": eta2_p,
            "mutual_information": mi,
            "per_category": per_cat,
            "shape": "categorical",
        }

    return results


def compare_environments(
    results_by_env: Dict[str, Dict],
) -> Dict[str, Any]:
    """
    Build the Signal Comparison Matrix: for each signal,
    compare metrics across environments.
    """
    all_signals = set()
    for env_results in results_by_env.values():
        all_signals.update(env_results.keys())

    comparison = {}
    for sig in sorted(all_signals):
        sig_row = {}
        for env_name, env_results in results_by_env.items():
            if sig in env_results:
                r = env_results[sig]
                if r["type"] == "continuous":
                    sig_row[env_name] = {
                        "pearson_r": r["pearson_r"],
                        "spearman_rho": r["spearman_rho"],
                        "MI": r["mutual_information"],
                        "shape": r["shape"],
                        "piecewise_left_r": r["piecewise"]["left_r"],
                        "piecewise_right_r": r["piecewise"]["right_r"],
                    }
                else:  # categorical
                    sig_row[env_name] = {
                        "eta_squared": r["eta_squared"],
                        "MI": r["mutual_information"],
                        "shape": "categorical",
                    }
            else:
                sig_row[env_name] = {"note": "N/A"}

        # Check if shapes differ across environments
        shapes = [sig_row[e].get("shape") for e in sig_row
                  if sig_row[e].get("shape") not in (None, "N/A", "categorical")]
        shape_differs = len(set(shapes)) > 1 if len(shapes) > 1 else False

        comparison[sig] = {
            "per_env": sig_row,
            "shape_differs": shape_differs,
            "shapes": shapes,
        }

    return comparison


# ══════════════════════════════════════════════════════════════════
# VISUALIZATION
# ══════════════════════════════════════════════════════════════════

def generate_signal_plots(
    data_by_env: Dict[str, pd.DataFrame],
    results_by_env: Dict[str, Dict],
    output_dir: str,
    lowess_frac: float = 0.3,
):
    """Generate scatter + LOWESS plots for each (signal, env) pair."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("  ⚠ matplotlib/seaborn not available, skipping plots")
        return

    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    envs = list(data_by_env.keys())
    signals = CONTINUOUS_SIGNALS

    for sig in signals:
        n_envs = sum(1 for e in envs
                     if sig in data_by_env[e].columns
                     and data_by_env[e][sig].notna().sum() > 0)
        if n_envs == 0:
            continue

        fig, axes = plt.subplots(1, max(n_envs, 1), figsize=(7 * n_envs, 5),
                                 squeeze=False)
        fig.suptitle(f"Signal: {sig} vs Utility", fontsize=14)

        ax_idx = 0
        for env_name in envs:
            df = data_by_env[env_name]
            if sig not in df.columns or df[sig].notna().sum() < 5:
                continue

            ax = axes[0][ax_idx]
            mask = df[sig].notna()
            x = df.loc[mask, sig].values.astype(float)
            y = df.loc[mask, "utility"].values

            # Scatter
            ax.scatter(x, y, alpha=0.3, s=12, color="steelblue", label="data")

            # LOWESS
            try:
                from statsmodels.nonparametric.smoothers_lowess import lowess
                smoothed = lowess(y, x, frac=lowess_frac, return_sorted=True)
                ax.plot(smoothed[:, 0], smoothed[:, 1], color="red",
                        linewidth=2.5, label="LOWESS")
            except ImportError:
                # Fallback: binned mean
                n_bins = min(20, len(x) // 5)
                if n_bins > 1:
                    bins = np.linspace(x.min(), x.max(), n_bins + 1)
                    bin_centers = (bins[:-1] + bins[1:]) / 2
                    bin_means = []
                    for i in range(n_bins):
                        bmask = (x >= bins[i]) & (x < bins[i + 1])
                        if bmask.sum() > 0:
                            bin_means.append(y[bmask].mean())
                        else:
                            bin_means.append(np.nan)
                    ax.plot(bin_centers, bin_means, color="red",
                            linewidth=2, label="binned mean")

            ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
            ax.set_xlabel(sig)
            ax.set_ylabel("Utility U")
            ax.set_title(f"{env_name.upper()}")

            # Add correlation text
            if sig in results_by_env.get(env_name, {}):
                r = results_by_env[env_name][sig]
                if r["type"] == "continuous":
                    info_text = (
                        f"r={r['pearson_r']:.3f}  ρ={r['spearman_rho']:.3f}\n"
                        f"MI={r['mutual_information']:.4f}  shape={r['shape']}"
                    )
                    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                            fontsize=8, verticalalignment="top",
                            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

            ax.legend(fontsize=8)
            ax_idx += 1

        plt.tight_layout()
        save_path = os.path.join(plot_dir, f"signal_{sig}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"    Plot: {save_path}")

    # ── Categorical signal plots (box plots) ──
    for sig in CATEGORICAL_SIGNALS:
        n_envs = sum(1 for e in envs
                     if sig in data_by_env[e].columns
                     and data_by_env[e][sig].notna().sum() > 0)
        if n_envs == 0:
            continue

        fig, axes = plt.subplots(1, max(n_envs, 1), figsize=(7 * n_envs, 5),
                                 squeeze=False)
        fig.suptitle(f"Signal: {sig} vs Utility", fontsize=14)

        ax_idx = 0
        for env_name in envs:
            df = data_by_env[env_name]
            if sig not in df.columns or df[sig].notna().sum() < 5:
                continue

            ax = axes[0][ax_idx]
            mask = df[sig].notna()
            subset = df.loc[mask, [sig, "utility"]]

            cats = sorted(subset[sig].unique())
            data_by_cat = [subset.loc[subset[sig] == c, "utility"].values for c in cats]
            data_by_cat = [d for d in data_by_cat if len(d) > 0]
            cat_labels = [str(c) for c, d in zip(cats, [subset.loc[subset[sig] == c, "utility"] for c in cats]) if len(d) > 0]

            if data_by_cat:
                bp = ax.boxplot(data_by_cat, tick_labels=cat_labels, patch_artist=True)
                colors = sns.color_palette("pastel", len(data_by_cat))
                for patch, color in zip(bp["boxes"], colors):
                    patch.set_facecolor(color)

            ax.axhline(y=0, color="red", linestyle="--", linewidth=0.8)
            ax.set_xlabel(sig)
            ax.set_ylabel("Utility U")
            ax.set_title(f"{env_name.upper()}")

            # Add η² text
            if sig in results_by_env.get(env_name, {}):
                r = results_by_env[env_name][sig]
                info_text = f"η²={r['eta_squared']:.4f}  MI={r['mutual_information']:.4f}"
                ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                        fontsize=8, verticalalignment="top",
                        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

            ax_idx += 1

        plt.tight_layout()
        save_path = os.path.join(plot_dir, f"signal_{sig}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"    Plot: {save_path}")

    # ── Summary heatmap: MI across (signal, env) ──
    all_signals = sorted(set(
        s for env_r in results_by_env.values() for s in env_r.keys()
    ))
    mi_matrix = []
    for sig in all_signals:
        row = []
        for env_name in envs:
            if sig in results_by_env.get(env_name, {}):
                row.append(results_by_env[env_name][sig]["mutual_information"])
            else:
                row.append(0.0)
        mi_matrix.append(row)

    if mi_matrix and len(envs) > 0:
        fig, ax = plt.subplots(figsize=(max(8, len(envs) * 3), max(4, len(all_signals) * 0.6)))
        mi_arr = np.array(mi_matrix)
        im = ax.imshow(mi_arr, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(len(envs)))
        ax.set_xticklabels([e.upper() for e in envs])
        ax.set_yticks(range(len(all_signals)))
        ax.set_yticklabels(all_signals)
        for i in range(len(all_signals)):
            for j in range(len(envs)):
                ax.text(j, i, f"{mi_arr[i, j]:.4f}", ha="center", va="center",
                        fontsize=9, color="black" if mi_arr[i, j] < 0.05 else "white")
        plt.colorbar(im, label="Mutual Information (nats)")
        ax.set_title("Signal Comparison Matrix: MI(signal, utility)")
        plt.tight_layout()
        heatmap_path = os.path.join(plot_dir, "signal_comparison_heatmap.png")
        plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"    Heatmap: {heatmap_path}")


def generate_phase0_comparison_plot(
    hotpotqa_df: Optional[pd.DataFrame],
    output_dir: str,
):
    """Compare Phase 0 (N=3) vs Phase 1 (N=5) utility distributions for HotpotQA."""
    if hotpotqa_df is None:
        return

    # Try to load Phase 0 data
    phase0_path = "results/phase0_idea_validation/hotpotqa/phase0_utility_data.json"
    if not os.path.exists(phase0_path):
        print("  ⚠ Phase 0 data not found, skipping comparison plot")
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        with open(phase0_path) as f:
            phase0_data = json.load(f)

        p0_utils = [dp["utility"] for dp in phase0_data]
        p1_utils = hotpotqa_df["utility"].values

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Phase 0 (N=3) vs Phase 1 (N=5): HotpotQA Utility Distribution")

        # Histogram comparison
        axes[0].hist(p0_utils, bins=30, alpha=0.6, color="steelblue",
                     label=f"Phase 0 (N=3, n={len(p0_utils)})", density=True)
        axes[0].hist(p1_utils, bins=30, alpha=0.6, color="coral",
                     label=f"Phase 1 (N=5, n={len(p1_utils)})", density=True)
        axes[0].set_xlabel("Utility U")
        axes[0].set_ylabel("Density")
        axes[0].set_title("Utility Distribution Comparison")
        axes[0].legend()

        # Per-step comparison
        p0_steps = {}
        for dp in phase0_data:
            s = dp["step"]
            p0_steps.setdefault(s, []).append(dp["utility"])

        p1_steps = {}
        for _, row in hotpotqa_df.iterrows():
            s = int(row["step"])
            p1_steps.setdefault(s, []).append(row["utility"])

        p0_s = sorted(p0_steps.keys())
        p1_s = sorted(p1_steps.keys())
        axes[1].plot(p0_s, [np.mean(p0_steps[s]) for s in p0_s],
                     "o-", color="steelblue", label="Phase 0 (N=3)")
        axes[1].plot(p1_s, [np.mean(p1_steps[s]) for s in p1_s],
                     "s-", color="coral", label="Phase 1 (N=5)")
        axes[1].set_xlabel("Step")
        axes[1].set_ylabel("Mean Utility")
        axes[1].set_title("Per-Step Mean Utility")
        axes[1].legend()

        plt.tight_layout()
        save_path = os.path.join(output_dir, "plots", "phase0_vs_phase1_comparison.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"    Comparison plot: {save_path}")

    except Exception as e:
        print(f"  ⚠ Phase 0 comparison plot failed: {e}")


# ══════════════════════════════════════════════════════════════════
# GO / NO-GO DECISION
# ══════════════════════════════════════════════════════════════════

def make_go_nogo_decision(
    comparison: Dict,
    results_by_env: Dict[str, Dict],
    go_nogo_cfg: Dict,
) -> Dict[str, Any]:
    """
    Apply Phase 1 Go / No-Go criteria.

    ✅ GO (ideal): ≥1 signal with MI > 0.05 or |ρ| > 0.2 in BOTH envs,
                   AND shapes differ across envs → C2 evidence
    ✅ GO (acceptable): ≥1 signal in ≥1 env
    ⚠️ WEAK: signals weak (MI < 0.03, |ρ| < 0.2) in both
    ❌ NO-GO: all signals MI < 0.01 and |ρ| < 0.1
    """
    go_mi = go_nogo_cfg.get("go_mi_threshold", 0.05)
    go_rho = go_nogo_cfg.get("go_spearman_threshold", 0.2)
    nogo_mi = go_nogo_cfg.get("nogo_mi_threshold", 0.01)
    nogo_rho = go_nogo_cfg.get("nogo_spearman_threshold", 0.1)

    envs = list(results_by_env.keys())
    strong_signals = {}  # signal → list of envs where it's strong

    for sig, comp in comparison.items():
        for env_name in envs:
            env_data = comp["per_env"].get(env_name, {})
            mi = env_data.get("MI", 0)
            rho = abs(env_data.get("spearman_rho", 0))

            if mi > go_mi or rho > go_rho:
                strong_signals.setdefault(sig, []).append(env_name)

    # Check for cross-env strong signals
    cross_env_strong = {s: e for s, e in strong_signals.items() if len(e) >= 2}

    # Check for shape differences
    shape_differs = any(comp["shape_differs"] for comp in comparison.values())

    # Decision
    if cross_env_strong and shape_differs:
        decision = "GO"
        emoji = "✅"
        description = (
            f"Strong signals found across environments: {list(cross_env_strong.keys())}. "
            f"Shape differences detected → C2 evidence exists. "
            f"Proceed to Phase 2: Gate Learning."
        )
        level = "ideal"
    elif cross_env_strong:
        decision = "GO"
        emoji = "✅"
        description = (
            f"Strong signals found across environments: {list(cross_env_strong.keys())}. "
            f"No shape difference detected (weaker C2). "
            f"Proceed to Phase 2, but a fixed-direction gate may work."
        )
        level = "acceptable"
    elif strong_signals:
        decision = "GO"
        emoji = "✅"
        description = (
            f"Signals found in some environments: {list(strong_signals.keys())}. "
            f"Proceed to Phase 2 with caution."
        )
        level = "acceptable_weak"
    elif any(
        env_data.get("MI", 0) > nogo_mi or abs(env_data.get("spearman_rho", 0)) > nogo_rho
        for comp in comparison.values()
        for env_data in comp["per_env"].values()
        if isinstance(env_data, dict) and "MI" in env_data
    ):
        decision = "WEAK"
        emoji = "⚠️"
        description = (
            "Signals exist but are weak (MI < 0.05, |ρ| < 0.2). "
            "Consider adding more signals or trying different environments."
        )
        level = "weak"
    else:
        decision = "NO_GO"
        emoji = "❌"
        description = (
            "No signals found above noise level. "
            "All MI < 0.01 and |ρ| < 0.1. "
            "Reconsider optimizer definition or environment selection."
        )
        level = "nogo"

    return {
        "decision": decision,
        "emoji": emoji,
        "level": level,
        "description": description,
        "strong_signals": {s: e for s, e in strong_signals.items()},
        "cross_env_strong": {s: e for s, e in cross_env_strong.items()},
        "shape_differs": shape_differs,
    }


# ══════════════════════════════════════════════════════════════════
# REPORT
# ══════════════════════════════════════════════════════════════════

def generate_analysis_report(
    results_by_env: Dict[str, Dict],
    comparison: Dict,
    decision: Dict,
    output_dir: str,
) -> str:
    """Generate the full Phase 1 analysis report."""
    envs = list(results_by_env.keys())

    lines = [
        "# Phase 1: Signal Discovery — Analysis Report",
        "",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Environments**: {', '.join(envs)}",
        "",
        "---",
        "",
        f"## Decision: {decision['emoji']} {decision['decision']} ({decision['level']})",
        "",
        f"> {decision['description']}",
        "",
        "---",
        "",
        "## 1. Signal Comparison Matrix",
        "",
    ]

    # Build the comparison table
    header = "| Signal |"
    sep = "|--------|"
    for env_name in envs:
        header += f" {env_name} Pearson | {env_name} Spearman | {env_name} MI | {env_name} Shape |"
        sep += "---------|-----------|------|-------|"
    lines.append(header)
    lines.append(sep)

    for sig, comp in sorted(comparison.items()):
        row = f"| **{sig}** |"
        for env_name in envs:
            env_data = comp["per_env"].get(env_name, {})
            if "note" in env_data:
                row += f" {env_data['note']} | — | — | — |"
            elif "pearson_r" in env_data:
                row += (f" {env_data['pearson_r']:.3f} | "
                        f"{env_data['spearman_rho']:.3f} | "
                        f"{env_data['MI']:.4f} | "
                        f"{env_data['shape']} |")
            elif "eta_squared" in env_data:
                row += (f" η²={env_data['eta_squared']:.4f} | "
                        f"— | "
                        f"{env_data['MI']:.4f} | "
                        f"categorical |")
            else:
                row += " — | — | — | — |"
        # Mark if shapes differ
        if comp.get("shape_differs"):
            row += " **⚡ SHAPE DIFFERS** |"
        lines.append(row)

    lines.extend([
        "",
        "---",
        "",
        "## 2. Per-Environment Details",
        "",
    ])

    for env_name in envs:
        lines.append(f"### {env_name.upper()}")
        lines.append("")

        env_results = results_by_env[env_name]
        for sig, r in sorted(env_results.items()):
            lines.append(f"#### {sig}")
            if r["type"] == "continuous":
                lines.extend([
                    f"- Pearson r = {r['pearson_r']:.4f} (p={r['pearson_p']:.4f})",
                    f"- Spearman ρ = {r['spearman_rho']:.4f} (p={r['spearman_p']:.4f})",
                    f"- MI = {r['mutual_information']:.4f} nats",
                    f"- Shape: **{r['shape']}**",
                    f"  - Left half: r={r['piecewise']['left_r']:.3f} (n={r['piecewise']['left_n']})",
                    f"  - Right half: r={r['piecewise']['right_r']:.3f} (n={r['piecewise']['right_n']})",
                ])
            else:
                lines.extend([
                    f"- η² = {r['eta_squared']:.4f} (p={r['eta_squared_p']:.4f})",
                    f"- MI = {r['mutual_information']:.4f} nats",
                ])
                for cat, info in sorted(r.get("per_category", {}).items()):
                    lines.append(
                        f"  - {cat}: N={info['count']}, "
                        f"mean_U={info['mean_utility']:.3f}, "
                        f"U>0={info['positive_ratio']:.0%}"
                    )
            lines.append("")

    # ── Shape difference analysis (C2 evidence) ──
    lines.extend([
        "---",
        "",
        "## 3. C2 Evidence: Shape Differences Across Environments",
        "",
    ])

    shape_diff_found = False
    for sig, comp in comparison.items():
        if comp.get("shape_differs"):
            shape_diff_found = True
            lines.append(f"### ⚡ {sig}")
            for env_name in envs:
                env_data = comp["per_env"].get(env_name, {})
                shape = env_data.get("shape", "N/A")
                lines.append(f"  - {env_name}: **{shape}**")
            lines.append("")

    if not shape_diff_found:
        lines.append("No shape differences detected across environments.")
        lines.append("")
        lines.append(
            "This weakens C2 (\"direction differs across environments\"), but does not "
            "invalidate the overall approach. A fixed-direction gate may suffice."
        )
        lines.append("")

    # ── Go / No-Go ──
    lines.extend([
        "---",
        "",
        "## 4. Go / No-Go Decision",
        "",
        f"**Decision**: {decision['emoji']} **{decision['decision']}** ({decision['level']})",
        "",
        f"> {decision['description']}",
        "",
        f"- Strong signals: {decision['strong_signals']}",
        f"- Cross-env strong: {decision['cross_env_strong']}",
        f"- Shape differs: {decision['shape_differs']}",
        "",
    ])

    report_text = "\n".join(lines)
    report_path = os.path.join(output_dir, "phase1_analysis_report.md")
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"  Analysis report: {report_path}")
    return report_path


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Phase 1: Signal Discovery — Multi-Indicator Analysis"
    )
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory with Phase 1 collected data")
    parser.add_argument("--config", type=str, default=None,
                        help="Config YAML for analysis parameters")
    parser.add_argument("--plots-only", action="store_true",
                        help="Only generate plots, skip heavy computation")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # Load config
    analysis_cfg = {}
    go_nogo_cfg = {}
    lowess_frac = 0.3
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        analysis_cfg = cfg.get("analysis", {})
        go_nogo_cfg = cfg.get("go_nogo", {})
        lowess_frac = analysis_cfg.get("lowess_frac", 0.3)

    output_dir = args.data_dir

    print()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  Phase 1: Signal Discovery — Analysis                           ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()

    # ── Load data ──
    data_by_env: Dict[str, pd.DataFrame] = {}

    for env_name in ["hotpotqa", "mbpp"]:
        data_path = os.path.join(output_dir, env_name, "phase1_signal_data.csv")
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            data_by_env[env_name] = df
            print(f"  Loaded {env_name}: {len(df)} data points from {data_path}")
        else:
            json_path = os.path.join(output_dir, env_name, "phase1_signal_data.json")
            if os.path.exists(json_path):
                with open(json_path) as f:
                    raw = json.load(f)
                df = pd.DataFrame(raw)
                data_by_env[env_name] = df
                print(f"  Loaded {env_name}: {len(df)} data points from {json_path}")

    if not data_by_env:
        print("  ❌ No data found! Run phase1_signal_discovery.py first.")
        return 1

    print()

    # ── Run analysis ──
    results_by_env = {}
    for env_name, df in data_by_env.items():
        print(f"  Analyzing {env_name.upper()}...")
        results = analyze_single_env(df, env_name, analysis_cfg)
        results_by_env[env_name] = results

        # Print quick summary
        for sig, r in sorted(results.items()):
            if r["type"] == "continuous":
                print(f"    {sig}: r={r['pearson_r']:.3f}, ρ={r['spearman_rho']:.3f}, "
                      f"MI={r['mutual_information']:.4f}, shape={r['shape']}")
            else:
                print(f"    {sig}: η²={r['eta_squared']:.4f}, "
                      f"MI={r['mutual_information']:.4f}")

    # ── Compare environments ──
    print()
    print("  Building Signal Comparison Matrix...")
    comparison = compare_environments(results_by_env)

    # Save analysis results
    analysis_results_path = os.path.join(output_dir, "phase1_analysis_results.json")
    with open(analysis_results_path, "w") as f:
        json.dump({
            "per_env": {
                env: {
                    sig: {k: v for k, v in r.items() if k != "per_category"}
                    for sig, r in env_r.items()
                }
                for env, env_r in results_by_env.items()
            },
            "comparison": comparison,
        }, f, indent=2, cls=NumpyEncoder)
    print(f"  Analysis results: {analysis_results_path}")

    # ── Plots ──
    print()
    print("  Generating plots...")
    generate_signal_plots(data_by_env, results_by_env, output_dir, lowess_frac)

    # Phase 0 vs Phase 1 comparison
    hotpotqa_df = data_by_env.get("hotpotqa")
    generate_phase0_comparison_plot(hotpotqa_df, output_dir)

    # ── Go / No-Go ──
    print()
    decision = make_go_nogo_decision(comparison, results_by_env, go_nogo_cfg)

    print(f"  Decision: {decision['emoji']} {decision['decision']} ({decision['level']})")
    print(f"  {decision['description']}")

    # Save decision
    decision_path = os.path.join(output_dir, "phase1_decision.json")
    with open(decision_path, "w") as f:
        json.dump(decision, f, indent=2, cls=NumpyEncoder)

    # ── Report ──
    print()
    generate_analysis_report(results_by_env, comparison, decision, output_dir)

    print()
    print("═" * 65)
    print(f"  ✅ Phase 1 Analysis Complete: {decision['emoji']} {decision['decision']}")
    if decision["decision"] == "GO":
        print("  Next: Phase 2 — Gate Learning")
    print("═" * 65)
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
