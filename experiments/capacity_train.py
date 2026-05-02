#!/usr/bin/env python3
"""
Phase 6 B2: Offline Probe Training + B3 GO/NO-GO + B6 Scientific Analysis
==========================================================================

Trains 4 probe architectures on Phase 6 multi-layer hidden state data,
evaluates GO/NO-GO, and runs scientific analyses (layer-wise probing,
cross-env transfer, learning curves).

Probe architectures:
  P1: Linear Regression        h(2560) → U        MSE         ~2.5K params
  P2: PCA(50) + LR Classifier  PCA(h)  → P(trig)  BCE         <1K params
  P3: Small MLP Regression     h(2560) → U        MSE         ~82K params
  P4: Small MLP Classifier     h(2560) → P(trig)  w-BCE       ~82K params

Also:
  B6.1: Layer-wise probing (9 layers × 3 envs → AUC per layer)
  B6.2: Cross-env transfer matrix (3×3 AUC)
  B6.3: Learning curve (AUC vs N_episodes)

Usage:
    python experiments/p6_b2_probe_training.py
"""
import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("DIAL")

# ══════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════

DATA_DIR = "results/phase6/hidden_states_multi"
OUTPUT_DIR = "results/phase6/probe_training"
ENVS = ["hotpotqa", "apps", "webshop"]
SEED = 42
N_CV_FOLDS = 5
LAYER_INDICES = [0, 4, 8, 12, 16, 20, 24, 28, 31]

# B3 GO/NO-GO thresholds
GO_R2 = 0.10
GO_AUC = 0.70


# ══════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════

def load_multi_layer_data(env: str, seed: int = 42) -> Dict:
    """Load multi-layer hidden state data for one environment."""
    path = os.path.join(DATA_DIR, env, f"seed_{seed}", "multi_layer_data.npz")
    data = np.load(path, allow_pickle=True)
    return {
        "hidden_states_multi": data["hidden_states_multi"],   # (N, n_layers, d)
        "hidden_states": data["hidden_states"],               # (N, d) - last layer
        "utilities": data["utilities"],                       # (N,)
        "signals": data["signals"],                           # (N, n_signals)
        "signal_keys": list(data["signal_keys"]),
    }


# ══════════════════════════════════════════════════════════════════
# PROBE IMPLEMENTATIONS
# ══════════════════════════════════════════════════════════════════

def train_p1_linear_regression(X: np.ndarray, y: np.ndarray, n_folds: int = 5) -> Dict:
    """P1: Linear Regression  h → U_pred (MSE)."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    r2s, rhos, aucs = [], [], []
    labels = (y > 0).astype(int)

    for train_idx, val_idx in kf.split(X):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_va = scaler.transform(X[val_idx])

        model = LinearRegression()
        model.fit(X_tr, y[train_idx])
        pred = model.predict(X_va)

        r2s.append(r2_score(y[val_idx], pred))
        rho, _ = stats.spearmanr(pred, y[val_idx])
        rhos.append(rho)
        if len(np.unique(labels[val_idx])) == 2:
            aucs.append(roc_auc_score(labels[val_idx], pred))

    return {
        "name": "P1_LinearRegression",
        "type": "regression",
        "params": X.shape[1] + 1,
        "r2_mean": float(np.mean(r2s)),
        "r2_std": float(np.std(r2s)),
        "rho_mean": float(np.mean(rhos)),
        "rho_std": float(np.std(rhos)),
        "auc_mean": float(np.mean(aucs)) if aucs else None,
        "auc_std": float(np.std(aucs)) if aucs else None,
    }


def train_p2_pca_lr_classifier(X: np.ndarray, y: np.ndarray, n_pca: int = 50, n_folds: int = 5) -> Dict:
    """P2: PCA(50) + Logistic Regression Classifier  PCA(h) → P(trigger)."""
    labels = (y > 0).astype(int)
    if len(np.unique(labels)) < 2:
        return {"name": "P2_PCA_LR", "error": "single_class", "auc_mean": 0.5}

    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    r2s, rhos, aucs = [], [], []

    for train_idx, val_idx in kf.split(X, labels):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_va = scaler.transform(X[val_idx])

        pca = PCA(n_components=min(n_pca, X_tr.shape[1], X_tr.shape[0]))
        X_tr_pca = pca.fit_transform(X_tr)
        X_va_pca = pca.transform(X_va)

        model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        model.fit(X_tr_pca, labels[train_idx])
        prob = model.predict_proba(X_va_pca)[:, 1]

        if len(np.unique(labels[val_idx])) == 2:
            aucs.append(roc_auc_score(labels[val_idx], prob))
        rho, _ = stats.spearmanr(prob, y[val_idx])
        rhos.append(rho)
        # R² from probability as utility proxy
        r2s.append(r2_score(y[val_idx], prob * y[labels == 1].mean() if labels.sum() > 0 else prob))

    return {
        "name": "P2_PCA_LR",
        "type": "classifier",
        "n_pca": pca.n_components_ if 'pca' in dir() else n_pca,
        "params": n_pca + 1,
        "r2_mean": float(np.mean(r2s)),
        "r2_std": float(np.std(r2s)),
        "rho_mean": float(np.mean(rhos)),
        "rho_std": float(np.std(rhos)),
        "auc_mean": float(np.mean(aucs)) if aucs else None,
        "auc_std": float(np.std(aucs)) if aucs else None,
    }


def train_p3_mlp_regression(X: np.ndarray, y: np.ndarray, n_folds: int = 5) -> Dict:
    """P3: Small MLP Regression  h → U_pred (MSE).  2560→64→1."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    r2s, rhos, aucs = [], [], []
    labels = (y > 0).astype(int)

    for train_idx, val_idx in kf.split(X):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_va = scaler.transform(X[val_idx])

        model = nn.Sequential(
            nn.Linear(X.shape[1], 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        X_t = torch.tensor(X_tr, dtype=torch.float32)
        y_t = torch.tensor(y[train_idx], dtype=torch.float32).unsqueeze(1)
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=64, shuffle=True)

        best_loss, patience_cnt, best_state = float("inf"), 0, None
        for epoch in range(150):
            model.train()
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                loss = nn.functional.mse_loss(model(xb), yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                X_va_t = torch.tensor(X_va, dtype=torch.float32).to(device)
                val_loss = nn.functional.mse_loss(
                    model(X_va_t),
                    torch.tensor(y[val_idx], dtype=torch.float32).unsqueeze(1).to(device),
                ).item()

            scheduler.step(val_loss)
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= 15:
                    break

        if best_state:
            model.load_state_dict(best_state)
            model = model.to(device)

        model.eval()
        with torch.no_grad():
            pred = model(torch.tensor(X_va, dtype=torch.float32).to(device)).cpu().numpy().ravel()

        r2s.append(r2_score(y[val_idx], pred))
        rho, _ = stats.spearmanr(pred, y[val_idx])
        rhos.append(rho)
        if len(np.unique(labels[val_idx])) == 2:
            aucs.append(roc_auc_score(labels[val_idx], pred))

    n_params = sum(p.numel() for p in model.parameters())
    return {
        "name": "P3_MLP_Regression",
        "type": "regression",
        "architecture": f"{X.shape[1]}→64→1",
        "params": n_params,
        "r2_mean": float(np.mean(r2s)),
        "r2_std": float(np.std(r2s)),
        "rho_mean": float(np.mean(rhos)),
        "rho_std": float(np.std(rhos)),
        "auc_mean": float(np.mean(aucs)) if aucs else None,
        "auc_std": float(np.std(aucs)) if aucs else None,
    }


def train_p4_mlp_classifier(X: np.ndarray, y: np.ndarray, n_folds: int = 5) -> Dict:
    """P4: Small MLP Classifier  h → P(trigger) (weighted BCE).  2560→64→1."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    labels = (y > 0).astype(int)
    if len(np.unique(labels)) < 2:
        return {"name": "P4_MLP_Classifier", "error": "single_class", "auc_mean": 0.5}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    r2s, rhos, aucs = [], [], []

    # Class weight
    pos_rate = labels.mean()
    pos_weight = torch.tensor([(1 - pos_rate) / max(pos_rate, 0.01)]).to(device)

    for train_idx, val_idx in kf.split(X, labels):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_va = scaler.transform(X[val_idx])

        model = nn.Sequential(
            nn.Linear(X.shape[1], 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        X_t = torch.tensor(X_tr, dtype=torch.float32)
        y_t = torch.tensor(labels[train_idx], dtype=torch.float32).unsqueeze(1)
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=64, shuffle=True)

        best_loss, patience_cnt, best_state = float("inf"), 0, None
        for epoch in range(150):
            model.train()
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                loss = criterion(model(xb), yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                X_va_t = torch.tensor(X_va, dtype=torch.float32).to(device)
                val_loss = criterion(
                    model(X_va_t),
                    torch.tensor(labels[val_idx], dtype=torch.float32).unsqueeze(1).to(device),
                ).item()

            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= 15:
                    break

        if best_state:
            model.load_state_dict(best_state)
            model = model.to(device)

        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(X_va, dtype=torch.float32).to(device)).cpu().numpy().ravel()
            prob = 1 / (1 + np.exp(-logits))

        rho, _ = stats.spearmanr(prob, y[val_idx])
        rhos.append(rho)
        if len(np.unique(labels[val_idx])) == 2:
            aucs.append(roc_auc_score(labels[val_idx], prob))
        r2s.append(r2_score(y[val_idx], prob * y[labels == 1].mean() if labels.sum() > 0 else prob))

    n_params = sum(p.numel() for p in model.parameters())
    return {
        "name": "P4_MLP_Classifier",
        "type": "classifier",
        "architecture": f"{X.shape[1]}→64→1",
        "params": n_params,
        "pos_rate": float(pos_rate),
        "r2_mean": float(np.mean(r2s)),
        "r2_std": float(np.std(r2s)),
        "rho_mean": float(np.mean(rhos)),
        "rho_std": float(np.std(rhos)),
        "auc_mean": float(np.mean(aucs)) if aucs else None,
        "auc_std": float(np.std(aucs)) if aucs else None,
    }


def train_handcrafted_lr_baseline(signals: np.ndarray, y: np.ndarray, n_folds: int = 5) -> Dict:
    """Handcrafted 5-feature LR baseline for comparison."""
    labels = (y > 0).astype(int)
    if len(np.unique(labels)) < 2:
        return {"name": "Handcrafted_LR", "error": "single_class"}

    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    r2s, rhos, aucs = [], [], []

    for train_idx, val_idx in kf.split(signals, labels):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(signals[train_idx])
        X_va = scaler.transform(signals[val_idx])

        model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        model.fit(X_tr, labels[train_idx])
        prob = model.predict_proba(X_va)[:, 1]

        if len(np.unique(labels[val_idx])) == 2:
            aucs.append(roc_auc_score(labels[val_idx], prob))
        rho, _ = stats.spearmanr(prob, y[val_idx])
        rhos.append(rho)
        r2s.append(r2_score(y[val_idx], prob * y[labels == 1].mean() if labels.sum() > 0 else prob))

    return {
        "name": "Handcrafted_LR",
        "type": "classifier",
        "n_features": signals.shape[1],
        "r2_mean": float(np.mean(r2s)),
        "r2_std": float(np.std(r2s)),
        "rho_mean": float(np.mean(rhos)),
        "rho_std": float(np.std(rhos)),
        "auc_mean": float(np.mean(aucs)) if aucs else None,
        "auc_std": float(np.std(aucs)) if aucs else None,
    }


# ══════════════════════════════════════════════════════════════════
# B6.1: LAYER-WISE PROBING
# ══════════════════════════════════════════════════════════════════

def layer_wise_probing(hidden_multi: np.ndarray, y: np.ndarray, n_folds: int = 5) -> List[Dict]:
    """Train linear probe on each layer independently, return AUC per layer."""
    labels = (y > 0).astype(int)
    if len(np.unique(labels)) < 2:
        return []

    results = []
    n_layers = hidden_multi.shape[1]

    for layer_idx in range(n_layers):
        X_layer = hidden_multi[:, layer_idx, :]  # (N, d)
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        aucs = []

        for train_idx, val_idx in kf.split(X_layer, labels):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_layer[train_idx])
            X_va = scaler.transform(X_layer[val_idx])

            model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
            model.fit(X_tr, labels[train_idx])
            prob = model.predict_proba(X_va)[:, 1]

            if len(np.unique(labels[val_idx])) == 2:
                aucs.append(roc_auc_score(labels[val_idx], prob))

        results.append({
            "layer_position": layer_idx,
            "layer_index": LAYER_INDICES[layer_idx] if layer_idx < len(LAYER_INDICES) else layer_idx,
            "auc_mean": float(np.mean(aucs)) if aucs else 0.5,
            "auc_std": float(np.std(aucs)) if aucs else 0.0,
        })

    return results


# ══════════════════════════════════════════════════════════════════
# B6.2: CROSS-ENV TRANSFER MATRIX
# ══════════════════════════════════════════════════════════════════

def cross_env_transfer(all_data: Dict[str, Dict]) -> Dict:
    """Train on env A, eval on env B. Returns 3×3 AUC matrix."""
    envs = list(all_data.keys())
    matrix = {}

    for train_env in envs:
        matrix[train_env] = {}
        X_tr = all_data[train_env]["hidden_states"]
        y_tr = all_data[train_env]["utilities"]
        labels_tr = (y_tr > 0).astype(int)

        if len(np.unique(labels_tr)) < 2:
            for eval_env in envs:
                matrix[train_env][eval_env] = 0.5
            continue

        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        model.fit(X_tr_scaled, labels_tr)

        for eval_env in envs:
            X_ev = all_data[eval_env]["hidden_states"]
            y_ev = all_data[eval_env]["utilities"]
            labels_ev = (y_ev > 0).astype(int)

            X_ev_scaled = scaler.transform(X_ev)
            prob = model.predict_proba(X_ev_scaled)[:, 1]

            if len(np.unique(labels_ev)) >= 2:
                auc = roc_auc_score(labels_ev, prob)
            else:
                auc = 0.5
            matrix[train_env][eval_env] = float(auc)

    return matrix


# ══════════════════════════════════════════════════════════════════
# B6.3: LEARNING CURVE
# ══════════════════════════════════════════════════════════════════

def learning_curve_analysis(X: np.ndarray, y: np.ndarray, n_repeats: int = 5) -> List[Dict]:
    """AUC vs number of training samples."""
    labels = (y > 0).astype(int)
    if len(np.unique(labels)) < 2:
        return []

    N = len(X)
    # Use fractions of total data
    fractions = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    results = []

    for frac in fractions:
        n_train = max(int(N * frac * 0.8), 10)  # 80% for train
        n_test = max(int(N * 0.2), 10)
        aucs = []

        for rep in range(n_repeats):
            rng = np.random.RandomState(42 + rep)
            idx = rng.permutation(N)
            test_idx = idx[:n_test]
            train_pool = idx[n_test:]
            train_idx = train_pool[:min(n_train, len(train_pool))]

            if len(np.unique(labels[train_idx])) < 2 or len(np.unique(labels[test_idx])) < 2:
                continue

            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[train_idx])
            X_te = scaler.transform(X[test_idx])

            model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
            model.fit(X_tr, labels[train_idx])
            prob = model.predict_proba(X_te)[:, 1]
            aucs.append(roc_auc_score(labels[test_idx], prob))

        results.append({
            "fraction": frac,
            "n_train": n_train,
            "auc_mean": float(np.mean(aucs)) if aucs else 0.5,
            "auc_std": float(np.std(aucs)) if aucs else 0.0,
        })

    return results


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.random.seed(SEED)

    # Load all data
    logger.info("=" * 60)
    logger.info("Phase 6 B2: Probe Training + B6 Scientific Analysis")
    logger.info("=" * 60)

    all_data = {}
    for env in ENVS:
        logger.info(f"\nLoading {env}...")
        data = load_multi_layer_data(env)
        all_data[env] = data
        n = len(data["utilities"])
        pos = (data["utilities"] > 0).sum()
        logger.info(f"  Steps: {n}, Positive: {pos} ({100*pos/n:.1f}%), "
                    f"Hidden: {data['hidden_states'].shape}, "
                    f"Multi: {data['hidden_states_multi'].shape}")

    # ── B2: Train 4 probes on each environment ──
    all_results = {}
    for env in ENVS:
        logger.info(f"\n{'='*60}")
        logger.info(f"B2: Training probes on {env.upper()}")
        logger.info(f"{'='*60}")

        data = all_data[env]
        X = data["hidden_states"]  # Last layer
        y = data["utilities"]
        signals = data["signals"]

        env_results = {"n_steps": len(y), "pos_rate": float((y > 0).mean())}

        # P1: Linear Regression
        logger.info(f"  P1: Linear Regression...")
        env_results["P1"] = train_p1_linear_regression(X, y, N_CV_FOLDS)
        logger.info(f"    R²={env_results['P1']['r2_mean']:.4f}±{env_results['P1']['r2_std']:.4f}, "
                    f"AUC={env_results['P1'].get('auc_mean', 'N/A')}")

        # P2: PCA + LR
        logger.info(f"  P2: PCA(50) + LR Classifier...")
        env_results["P2"] = train_p2_pca_lr_classifier(X, y, n_pca=50, n_folds=N_CV_FOLDS)
        logger.info(f"    AUC={env_results['P2'].get('auc_mean', 'N/A')}")

        # P3: MLP Regression
        logger.info(f"  P3: MLP Regression (2560→64→1)...")
        env_results["P3"] = train_p3_mlp_regression(X, y, N_CV_FOLDS)
        logger.info(f"    R²={env_results['P3']['r2_mean']:.4f}±{env_results['P3']['r2_std']:.4f}, "
                    f"AUC={env_results['P3'].get('auc_mean', 'N/A')}")

        # P4: MLP Classifier
        logger.info(f"  P4: MLP Classifier (2560→64→1, weighted BCE)...")
        env_results["P4"] = train_p4_mlp_classifier(X, y, N_CV_FOLDS)
        logger.info(f"    AUC={env_results['P4'].get('auc_mean', 'N/A')}")

        # Handcrafted LR baseline
        logger.info(f"  Baseline: Handcrafted LR ({signals.shape[1]} features)...")
        env_results["baseline_handcrafted"] = train_handcrafted_lr_baseline(signals, y, N_CV_FOLDS)
        logger.info(f"    AUC={env_results['baseline_handcrafted'].get('auc_mean', 'N/A')}")

        # B6.1: Layer-wise probing
        logger.info(f"  B6.1: Layer-wise probing (9 layers)...")
        env_results["layer_wise"] = layer_wise_probing(data["hidden_states_multi"], y, N_CV_FOLDS)
        for lw in env_results["layer_wise"]:
            logger.info(f"    Layer {lw['layer_index']:2d}: AUC={lw['auc_mean']:.4f}")

        # B6.3: Learning curve
        logger.info(f"  B6.3: Learning curve...")
        env_results["learning_curve"] = learning_curve_analysis(X, y)
        for lc in env_results["learning_curve"]:
            logger.info(f"    N={lc['n_train']:4d} ({lc['fraction']:.0%}): AUC={lc['auc_mean']:.4f}")

        all_results[env] = env_results

    # ── B6.2: Cross-env transfer ──
    logger.info(f"\n{'='*60}")
    logger.info(f"B6.2: Cross-Environment Transfer Matrix")
    logger.info(f"{'='*60}")
    transfer_matrix = cross_env_transfer(all_data)

    header_label = "Train\\Eval"
    logger.info(f"\n  {header_label:>12s} | {'HotpotQA':>10s} | {'APPS':>10s} | {'WebShop':>10s}")
    sep = "-"
    logger.info(f"  {sep*12}-+-{sep*10}-+-{sep*10}-+-{sep*10}")
    for train_env in ENVS:
        row = f"  {train_env:>12s} |"
        for eval_env in ENVS:
            auc = transfer_matrix[train_env][eval_env]
            marker = " *" if train_env == eval_env else ""
            row += f" {auc:>8.4f}{marker} |"
        logger.info(row)

    all_results["transfer_matrix"] = transfer_matrix

    # ── B3: GO/NO-GO Decision ──
    logger.info(f"\n{'='*60}")
    logger.info(f"B3: GO/NO-GO Decision")
    logger.info(f"{'='*60}")

    go_decision = {"per_env": {}, "overall": None}
    for env in ENVS:
        r = all_results[env]
        best_r2 = max(
            r["P1"]["r2_mean"],
            r["P3"]["r2_mean"],
        )
        best_auc = max(
            r["P1"].get("auc_mean") or 0,
            r["P2"].get("auc_mean") or 0,
            r["P3"].get("auc_mean") or 0,
            r["P4"].get("auc_mean") or 0,
        )
        baseline_auc = r["baseline_handcrafted"].get("auc_mean") or 0

        env_go = best_r2 > GO_R2 or best_auc > GO_AUC
        go_decision["per_env"][env] = {
            "best_r2": best_r2,
            "best_auc": best_auc,
            "baseline_auc": baseline_auc,
            "probe_vs_baseline": best_auc - baseline_auc,
            "go": env_go,
        }

        status = "✅ GO" if env_go else "❌ NO-GO"
        logger.info(f"  {env:>10s}: best R²={best_r2:.4f}, best AUC={best_auc:.4f}, "
                    f"baseline AUC={baseline_auc:.4f}, Δ={best_auc - baseline_auc:+.4f} → {status}")

    n_go = sum(1 for v in go_decision["per_env"].values() if v["go"])
    overall_go = n_go >= 2
    go_decision["overall"] = {
        "n_go": n_go,
        "n_total": len(ENVS),
        "go": overall_go,
        "decision": "GO" if overall_go else "NO-GO",
    }

    logger.info(f"\n  Overall: {n_go}/{len(ENVS)} envs GO → {'✅ GO' if overall_go else '❌ NO-GO'}")
    all_results["go_decision"] = go_decision

    # ── Summary Table ──
    logger.info(f"\n{'='*60}")
    logger.info(f"SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"\n  {'Method':>25s} | {'HotpotQA AUC':>14s} | {'APPS AUC':>14s} | {'WebShop AUC':>14s}")
    logger.info(f"  {'-'*25}-+-{'-'*14}-+-{'-'*14}-+-{'-'*14}")
    for method in ["P1", "P2", "P3", "P4", "baseline_handcrafted"]:
        row = f"  {method:>25s} |"
        for env in ENVS:
            auc = all_results[env][method].get("auc_mean")
            row += f" {auc:>12.4f}   |" if auc is not None else f" {'N/A':>12s}   |"
        logger.info(row)

    # ── Save results ──
    output_path = os.path.join(OUTPUT_DIR, "b2_probe_results.json")

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, cls=NpEncoder)
    logger.info(f"\nResults saved to {output_path}")

    return all_results


if __name__ == "__main__":
    main()
