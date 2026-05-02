#!/usr/bin/env python3
"""
Phase 7: Regularizer Ablation (Offline)
========================================

Companion to the SR-based deployment ablation. Fits 5 regularizer variants
on the same exploration data DIAL collected during its probe phase
(`results/phase6/hidden_states_multi/{env}/seed_{seed}/multi_layer_data.npz`),
and reports CV-AUC + #non-zero coefficients per variant.

This isolates the regularizer choice from the deployment loop (which is
expensive; see scripts/phase6/run_regularizer_ablation.sbatch). The signed-
weight diagnostic and the #nz claim in the paper rely on offline sparsity
properties, so the AUC numbers here support the paper's argument even if
the SR runs are not yet complete.

Variants:
  - l1       : DIAL default — LogisticRegressionCV(penalty='l1')
  - l2       : Ridge — LogisticRegressionCV(penalty='l2')
  - none     : LogisticRegression(penalty=None) on full pool
  - elastic  : LogisticRegressionCV(penalty='elasticnet', l1_ratio=0.5)
  - mi_top3  : top-3 features by MI, then unregularized LR

Pool composition (≈30 features, mirroring DIAL's candidate pool):
  - 4-6 universal signals (entropy, step_count, evidence_count, num_avail_actions, is_finish)
  - 2 derived (entropy_sq, step_x_entropy)
  - 20 hidden-state PCA components

Usage:
    python experiments/p7_regularizer_ablation_offline.py
    python experiments/p7_regularizer_ablation_offline.py --envs hotpotqa webshop plancraft
"""
import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("p7-reg-ablation")

DATA_DIR = "results/phase6/hidden_states_multi"
OUTPUT_DIR = "results/phase7/regularizer_ablation"
ENVS = ["hotpotqa", "webshop", "plancraft"]
SEEDS = [42]  # phase6 only collected seed_42 for hidden_states_multi
EXTRA_RESAMPLE_SEEDS = [123, 456]  # bootstrap resamples for stability
N_FOLDS = 5
N_PCA = 20


def load_npz(env: str, seed: int = 42):
    path = os.path.join(DATA_DIR, env, f"seed_{seed}", "multi_layer_data.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return np.load(path, allow_pickle=True)


def build_feature_pool(data) -> tuple[np.ndarray, list[str]]:
    """Concatenate raw signals + derived + hidden-state PCA into a (N, F) matrix."""
    signals = data["signals"]  # (N, n_sig)
    keys = list(data["signal_keys"])
    hidden = data["hidden_states"]  # (N, 2560)

    feat_cols = []
    feat_names = []

    # Raw signals (drop "utility" key if present — that's the target leak)
    for i, k in enumerate(keys):
        if k.lower() == "utility":
            continue
        feat_cols.append(signals[:, i])
        feat_names.append(k)

    # Derived
    if "token_entropy" in feat_names:
        ent = signals[:, keys.index("token_entropy")]
        feat_cols.append(ent ** 2)
        feat_names.append("entropy_sq")
        if "step_count" in feat_names:
            sc = signals[:, keys.index("step_count")]
            feat_cols.append(sc * ent)
            feat_names.append("step_x_entropy")

    # Hidden-state PCA (20 components, fit on the pool)
    scaler_h = StandardScaler()
    h_scaled = scaler_h.fit_transform(hidden)
    n_pca = min(N_PCA, h_scaled.shape[0], h_scaled.shape[1])
    pca = PCA(n_components=n_pca, random_state=42)
    h_pca = pca.fit_transform(h_scaled)
    for i in range(n_pca):
        feat_cols.append(h_pca[:, i])
        feat_names.append(f"h_pca_{i}")

    X = np.column_stack(feat_cols).astype(np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, feat_names


def cv_auc(model_factory, X, y, n_folds=N_FOLDS, seed=42):
    """Return mean CV-AUC + std + mean #nz weights across folds."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    aucs, nzs = [], []
    coefs_accum = []
    for tr, va in skf.split(X, y):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr])
        X_va = scaler.transform(X[va])
        X_tr = np.nan_to_num(X_tr, nan=0, posinf=0, neginf=0)
        X_va = np.nan_to_num(X_va, nan=0, posinf=0, neginf=0)

        model = model_factory()
        model.fit(X_tr, y[tr])
        if hasattr(model, "predict_proba"):
            try:
                p = model.predict_proba(X_va)[:, 1]
            except Exception:
                p = model.decision_function(X_va)
        else:
            p = model.decision_function(X_va)
        if len(np.unique(y[va])) == 2:
            aucs.append(roc_auc_score(y[va], p))
        coef = np.abs(model.coef_[0]) if hasattr(model, "coef_") else None
        if coef is not None:
            nzs.append(int((coef > 1e-6).sum()))
            coefs_accum.append(coef)
    nz_mean = float(np.mean(nzs)) if nzs else None
    avg_coef = np.mean(coefs_accum, axis=0) if coefs_accum else None
    return {
        "auc_mean": float(np.mean(aucs)) if aucs else None,
        "auc_std": float(np.std(aucs)) if aucs else None,
        "nz_mean": nz_mean,
        "n_folds_used": len(aucs),
        "avg_abs_coef": avg_coef.tolist() if avg_coef is not None else None,
    }


def make_factories(pos_count: int):
    """Build (name -> factory) dict. Factory must return a fresh sklearn estimator."""
    cv = max(2, min(5, pos_count))

    def f_l1():
        return LogisticRegressionCV(
            penalty="l1", solver="saga", cv=cv,
            max_iter=2000, class_weight="balanced", random_state=42,
        )

    def f_l2():
        return LogisticRegressionCV(
            penalty="l2", solver="lbfgs", cv=cv,
            max_iter=2000, class_weight="balanced", random_state=42,
        )

    def f_none():
        return LogisticRegression(
            penalty=None, solver="lbfgs",
            max_iter=2000, class_weight="balanced", random_state=42,
        )

    def f_elastic():
        return LogisticRegressionCV(
            penalty="elasticnet", solver="saga", cv=cv,
            l1_ratios=[0.5], max_iter=2000,
            class_weight="balanced", random_state=42,
        )

    return {
        "l1": f_l1,
        "l2": f_l2,
        "none": f_none,
        "elastic": f_elastic,
    }


def fit_mi_top3(X, y, feat_names, n_folds=N_FOLDS, seed=42):
    """Hard MI top-3 selection, then unregularized logistic. Re-selects per fold."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    aucs, picks = [], []
    for tr, va in skf.split(X, y):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr])
        X_va = scaler.transform(X[va])
        X_tr = np.nan_to_num(X_tr, nan=0, posinf=0, neginf=0)
        X_va = np.nan_to_num(X_va, nan=0, posinf=0, neginf=0)
        try:
            mi = mutual_info_classif(X_tr, y[tr], random_state=42)
        except Exception:
            mi = np.zeros(X_tr.shape[1])
        top = np.argsort(mi)[::-1][:3]
        picks.append([feat_names[i] for i in top])
        model = LogisticRegression(
            penalty=None, solver="lbfgs", max_iter=2000,
            class_weight="balanced", random_state=42,
        )
        model.fit(X_tr[:, top], y[tr])
        p = model.predict_proba(X_va[:, top])[:, 1]
        if len(np.unique(y[va])) == 2:
            aucs.append(roc_auc_score(y[va], p))
    return {
        "auc_mean": float(np.mean(aucs)) if aucs else None,
        "auc_std": float(np.std(aucs)) if aucs else None,
        "nz_mean": 3.0,
        "n_folds_used": len(aucs),
        "selections_per_fold": picks,
    }


def fit_l1_full(X, y, feat_names, pos_count: int, seed=42):
    """Single full-data L1 fit — for reporting selected features at the env level."""
    cv = max(2, min(5, pos_count))
    scaler = StandardScaler()
    Xs = np.nan_to_num(scaler.fit_transform(X), nan=0, posinf=0, neginf=0)
    model = LogisticRegressionCV(
        penalty="l1", solver="saga", cv=cv,
        max_iter=2000, class_weight="balanced", random_state=seed,
    )
    model.fit(Xs, y)
    coef = model.coef_[0]
    selected = [feat_names[i] for i in np.where(np.abs(coef) > 1e-6)[0]]
    return {"selected": selected, "n_selected": len(selected),
            "C": float(model.C_[0]) if hasattr(model, "C_") else None}


def run_env(env: str, seed: int):
    logger.info(f"\n=== {env.upper()} (seed {seed}) ===")
    data = load_npz(env, seed)
    X, feat_names = build_feature_pool(data)
    U = data["utilities"]
    y = (U > 0).astype(int)
    pos_count = int(y.sum())
    pos_rate = float(y.mean())
    logger.info(
        f"  N={X.shape[0]}, F={X.shape[1]}, pos_rate={pos_rate:.3f} "
        f"(pos_count={pos_count})"
    )

    if pos_count < 2 or pos_count > len(y) - 2:
        logger.warning(f"  Insufficient class balance: pos_count={pos_count}; skipping.")
        return {"env": env, "seed": seed, "skipped": True,
                "pos_rate": pos_rate, "n_samples": int(X.shape[0])}

    factories = make_factories(pos_count)

    results = {
        "env": env,
        "seed": seed,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "pos_rate": pos_rate,
        "feature_names": feat_names,
        "variants": {},
    }

    for name, fac in factories.items():
        logger.info(f"  fitting {name} ...")
        try:
            r = cv_auc(fac, X, y)
        except Exception as e:
            logger.warning(f"    {name} failed: {e}")
            r = {"error": str(e)}
        results["variants"][name] = r
        if "auc_mean" in r and r["auc_mean"] is not None:
            logger.info(
                f"    {name:8s}  AUC={r['auc_mean']:.3f}±{r['auc_std']:.3f}  "
                f"#nz={r['nz_mean']:.1f}"
            )

    logger.info("  fitting mi_top3 ...")
    results["variants"]["mi_top3"] = fit_mi_top3(X, y, feat_names)
    rt = results["variants"]["mi_top3"]
    if rt["auc_mean"] is not None:
        logger.info(f"    mi_top3   AUC={rt['auc_mean']:.3f}±{rt['auc_std']:.3f}  #nz=3.0")

    # Full-data L1 selection (for the env-level "selected features" report)
    results["l1_full_data_selection"] = fit_l1_full(X, y, feat_names, pos_count)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", nargs="+", default=ENVS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=OUTPUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    all_results = {}
    for env in args.envs:
        try:
            all_results[env] = run_env(env, args.seed)
        except FileNotFoundError as e:
            logger.warning(f"  {env}: data missing ({e}); skipping.")
            all_results[env] = {"env": env, "skipped": True, "reason": "data missing"}

    # Pretty summary table
    lines = [
        "",
        "=" * 70,
        "Regularizer Ablation Summary (offline; CV-AUC over 5 folds)",
        "=" * 70,
        f"{'Env':<11s} {'Reg':<10s} {'AUC':<14s} {'#nz':<7s}",
        "-" * 70,
    ]
    for env, r in all_results.items():
        if r.get("skipped"):
            lines.append(f"{env:<11s}  (skipped: {r.get('reason','no data')})")
            continue
        for vname, vr in r["variants"].items():
            if vr.get("auc_mean") is None:
                continue
            lines.append(
                f"{env:<11s} {vname:<10s} "
                f"{vr['auc_mean']:.3f}±{vr['auc_std']:.3f}  "
                f"{vr['nz_mean']:.1f}"
            )
        lines.append("-" * 70)
    summary = "\n".join(lines)
    print(summary)

    out_path = os.path.join(args.output, "results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    with open(os.path.join(args.output, "summary.txt"), "w") as f:
        f.write(summary + "\n")
    logger.info(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
