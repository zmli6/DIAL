"""
Probe Models for Phase 5 — Hidden State & Text Embedding VoC Prediction.

Contains two probe architectures for predicting rollout utility (VoC)
from different state representations:

  5.1A  HiddenStateProbe  — MLP on LLM last-layer hidden states (d=2560)
  5.1B  TextEmbeddingProbe — MLP on sentence-transformer embeddings (d=384)

Both use the same ``VOCProbeHead`` MLP with Gaussian NLL loss
(predicts mean + log-variance of utility distribution).

GO/NO-GO thresholds:
  5.1A: R² > 0.15 in ≥ 2/3 environments → GO
  5.1B: R² > 0.10 in ≥ 2/3 environments → GO
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("DIAL")


# ══════════════════════════════════════════════════════════════════
# VOC PROBE HEAD (shared MLP architecture)
# ══════════════════════════════════════════════════════════════════

class VOCProbeHead:
    """
    MLP probe head that predicts utility distribution parameters.

    Architecture: input_dim → d_hidden → d_hidden//2 → 2 (mean, logvar)
    Loss: Gaussian negative log-likelihood.
    ~690K params for d_in=2560, d_hidden=256.

    Parameters
    ----------
    input_dim : int
        Dimensionality of input features.
    d_hidden : int
        Hidden layer size.
    lr : float
        Learning rate.
    weight_decay : float
        L2 regularization.
    """

    def __init__(
        self,
        input_dim: int,
        d_hidden: int = 256,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        import torch
        import torch.nn as nn

        self.input_dim = input_dim
        self.d_hidden = d_hidden
        self.lr = lr
        self.weight_decay = weight_decay

        self.model = nn.Sequential(
            nn.Linear(input_dim, d_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_hidden, d_hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_hidden // 2, 2),  # mean, logvar
        )

        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info(
            f"[VOCProbeHead] input_dim={input_dim}, d_hidden={d_hidden}, "
            f"params={n_params:,}"
        )

    def to(self, device):
        self.model = self.model.to(device)
        return self

    def train_probe(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        batch_size: int = 64,
        val_fraction: float = 0.15,
        patience: int = 15,
        device: str = "cuda",
    ) -> Dict[str, Any]:
        """
        Train the probe on (features, utility) pairs.

        Parameters
        ----------
        X : np.ndarray, shape (N, input_dim)
        y : np.ndarray, shape (N,)
        epochs : int
        batch_size : int
        val_fraction : float
        patience : int
            Early stopping patience.
        device : str

        Returns
        -------
        dict with training_log, best_epoch, final_val_loss
        """
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        self.model = self.model.to(device)

        # Train/val split
        N = len(X)
        n_val = max(int(N * val_fraction), 1)
        indices = np.random.permutation(N)
        val_idx, train_idx = indices[:n_val], indices[n_val:]

        X_train = torch.tensor(X[train_idx], dtype=torch.float32)
        y_train = torch.tensor(y[train_idx], dtype=torch.float32)
        X_val = torch.tensor(X[val_idx], dtype=torch.float32)
        y_val = torch.tensor(y[val_idx], dtype=torch.float32)

        train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=batch_size, shuffle=True,
        )

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5,
        )

        training_log = []
        best_val_loss = float("inf")
        best_epoch = 0
        best_state = None
        wait = 0

        for epoch in range(epochs):
            # Train
            self.model.train()
            train_losses = []
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = self.model(xb)
                loss = self._gaussian_nll(out, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            # Validate
            self.model.eval()
            with torch.no_grad():
                out_val = self.model(X_val.to(device))
                val_loss = self._gaussian_nll(out_val, y_val.to(device)).item()

                # R² on validation
                pred_mean = out_val[:, 0].cpu().numpy()
                y_val_np = y_val.numpy()
                ss_res = np.sum((y_val_np - pred_mean) ** 2)
                ss_tot = np.sum((y_val_np - y_val_np.mean()) ** 2)
                r2 = 1.0 - ss_res / max(ss_tot, 1e-8)

            scheduler.step(val_loss)

            log_entry = {
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)),
                "val_loss": val_loss,
                "val_r2": float(r2),
            }
            training_log.append(log_entry)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    logger.info(f"[VOCProbeHead] Early stopping at epoch {epoch}")
                    break

            if epoch % 20 == 0 or epoch == epochs - 1:
                logger.info(
                    f"[VOCProbeHead] Epoch {epoch}: "
                    f"train_loss={np.mean(train_losses):.4f}, "
                    f"val_loss={val_loss:.4f}, val_r2={r2:.4f}"
                )

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model = self.model.to(device)

        return {
            "training_log": training_log,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "final_val_r2": training_log[best_epoch]["val_r2"] if training_log else 0.0,
        }

    @staticmethod
    def _gaussian_nll(output, target):
        """Gaussian negative log-likelihood loss."""
        import torch
        mean = output[:, 0]
        logvar = output[:, 1]
        # Clamp logvar for stability
        logvar = torch.clamp(logvar, min=-10, max=10)
        var = torch.exp(logvar)
        nll = 0.5 * (logvar + (target - mean) ** 2 / var)
        return nll.mean()

    def predict(self, X: np.ndarray, device: str = "cuda") -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict mean and variance of utility.

        Returns
        -------
        means : np.ndarray, shape (N,)
        variances : np.ndarray, shape (N,)
        """
        import torch
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(device)
            out = self.model(X_t)
            means = out[:, 0].cpu().numpy()
            logvars = out[:, 1].cpu().numpy()
            variances = np.exp(np.clip(logvars, -10, 10))
        return means, variances

    def save(self, path: str):
        """Save model state dict."""
        import torch
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "state_dict": self.model.state_dict(),
            "input_dim": self.input_dim,
            "d_hidden": self.d_hidden,
        }, path)

    def load(self, path: str, device: str = "cuda"):
        """Load model state dict."""
        import torch
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model = self.model.to(device)


# ══════════════════════════════════════════════════════════════════
# HIDDEN STATE PROBE (5.1A)
# ══════════════════════════════════════════════════════════════════

class HiddenStateProbe:
    """
    Probe that predicts VoC from LLM hidden states.

    Uses mean-pooled last-layer hidden states (d_model dimensions)
    as input features for a ``VOCProbeHead`` MLP.

    Parameters
    ----------
    input_dim : int
        Hidden state dimensionality (e.g. 2560 for Qwen3-4B).
    d_hidden : int
        MLP hidden layer size.
    pooling : str
        Pooling strategy: ``"mean"`` or ``"last_token"``.
    utility_threshold : float
        Threshold for GO/NO-GO gate decisions (U > threshold → trigger).
    """

    def __init__(
        self,
        input_dim: int = 2560,
        d_hidden: int = 256,
        pooling: str = "mean",
        utility_threshold: float = 0.05,
        lr: float = 1e-3,
    ):
        self.input_dim = input_dim
        self.d_hidden = d_hidden
        self.pooling = pooling
        self.utility_threshold = utility_threshold

        self.probe = VOCProbeHead(
            input_dim=input_dim, d_hidden=d_hidden, lr=lr,
        )
        self._train_result = None

    def train_probe(
        self,
        hidden_states: np.ndarray,
        utilities: np.ndarray,
        epochs: int = 100,
        batch_size: int = 64,
        device: str = "cuda",
    ) -> Dict[str, Any]:
        """
        Train the probe on collected hidden states and utilities.

        Parameters
        ----------
        hidden_states : np.ndarray, shape (N, input_dim)
        utilities : np.ndarray, shape (N,)

        Returns
        -------
        dict with training log and metrics.
        """
        assert hidden_states.shape[1] == self.input_dim, (
            f"Expected input_dim={self.input_dim}, got {hidden_states.shape[1]}"
        )

        result = self.probe.train_probe(
            hidden_states, utilities,
            epochs=epochs, batch_size=batch_size, device=device,
        )
        self._train_result = result
        return result

    def evaluate(
        self,
        hidden_states: np.ndarray,
        utilities: np.ndarray,
        device: str = "cuda",
    ) -> Dict[str, float]:
        """
        Evaluate probe performance.

        Returns
        -------
        dict with r2, mae, gate_accuracy, gate_precision, gate_recall
        """
        pred_means, pred_vars = self.probe.predict(hidden_states, device)

        # R²
        ss_res = np.sum((utilities - pred_means) ** 2)
        ss_tot = np.sum((utilities - utilities.mean()) ** 2)
        r2 = 1.0 - ss_res / max(ss_tot, 1e-8)

        # MAE
        mae = float(np.mean(np.abs(utilities - pred_means)))

        # Gate accuracy: predict trigger when predicted mean > threshold
        pred_trigger = pred_means > self.utility_threshold
        true_trigger = utilities > self.utility_threshold
        gate_acc = float(np.mean(pred_trigger == true_trigger))

        # Precision / Recall
        tp = np.sum(pred_trigger & true_trigger)
        fp = np.sum(pred_trigger & ~true_trigger)
        fn = np.sum(~pred_trigger & true_trigger)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)

        return {
            "r2": float(r2),
            "mae": mae,
            "gate_accuracy": gate_acc,
            "gate_precision": float(precision),
            "gate_recall": float(recall),
        }

    def make_gate_decision(
        self,
        hidden_state: np.ndarray,
        device: str = "cuda",
    ) -> bool:
        """
        Decide whether to trigger a rollout based on a single hidden state.

        Returns True if predicted utility mean exceeds the threshold.
        """
        X = hidden_state.reshape(1, -1)
        means, _ = self.probe.predict(X, device)
        return bool(means[0] > self.utility_threshold)

    def save(self, path: str):
        self.probe.save(path)

    def load(self, path: str, device: str = "cuda"):
        self.probe.load(path, device)


# ══════════════════════════════════════════════════════════════════
# TEXT EMBEDDING PROBE (5.1B)
# ══════════════════════════════════════════════════════════════════

class TextEmbeddingProbe:
    """
    Probe that predicts VoC from sentence-transformer text embeddings.

    Uses ``all-MiniLM-L6-v2`` (d=384) to encode state descriptions,
    then feeds embeddings into a ``VOCProbeHead`` MLP.

    Parameters
    ----------
    model_name : str
        Sentence transformer model name.
    d_hidden : int
        MLP hidden layer size.
    utility_threshold : float
        Threshold for GO/NO-GO gate decisions.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        d_hidden: int = 128,
        utility_threshold: float = 0.05,
        lr: float = 1e-3,
    ):
        self.model_name = model_name
        self.d_hidden = d_hidden
        self.utility_threshold = utility_threshold
        self._encoder = None
        self._embedding_dim = None
        self.lr = lr

        self.probe = None  # Initialized after first encode call
        self._train_result = None

    def _ensure_encoder(self):
        """Lazy-load sentence transformer."""
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(self.model_name)
            # Determine embedding dimension
            test_emb = self._encoder.encode(["test"], show_progress_bar=False)
            self._embedding_dim = test_emb.shape[1]
            logger.info(
                f"[TextEmbeddingProbe] Loaded {self.model_name}, "
                f"d={self._embedding_dim}"
            )

    def encode_texts(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """
        Encode a list of text descriptions into embeddings.

        Returns
        -------
        np.ndarray, shape (N, embedding_dim)
        """
        self._ensure_encoder()
        embeddings = self._encoder.encode(
            texts, batch_size=batch_size, show_progress_bar=False,
            normalize_embeddings=True,
        )
        return np.array(embeddings, dtype=np.float32)

    def train_probe(
        self,
        texts: List[str],
        utilities: np.ndarray,
        epochs: int = 100,
        batch_size: int = 64,
        device: str = "cuda",
    ) -> Dict[str, Any]:
        """
        Train the probe on text descriptions and utilities.

        Encodes texts first, then trains the MLP probe.
        """
        embeddings = self.encode_texts(texts)

        if self.probe is None:
            self.probe = VOCProbeHead(
                input_dim=self._embedding_dim,
                d_hidden=self.d_hidden,
                lr=self.lr,
            )

        result = self.probe.train_probe(
            embeddings, utilities,
            epochs=epochs, batch_size=batch_size, device=device,
        )
        self._train_result = result
        return result

    def train_probe_from_embeddings(
        self,
        embeddings: np.ndarray,
        utilities: np.ndarray,
        epochs: int = 100,
        batch_size: int = 64,
        device: str = "cuda",
    ) -> Dict[str, Any]:
        """Train using pre-computed embeddings."""
        if self.probe is None:
            self.probe = VOCProbeHead(
                input_dim=embeddings.shape[1],
                d_hidden=self.d_hidden,
                lr=self.lr,
            )

        result = self.probe.train_probe(
            embeddings, utilities,
            epochs=epochs, batch_size=batch_size, device=device,
        )
        self._train_result = result
        return result

    def evaluate(
        self,
        texts: Optional[List[str]] = None,
        embeddings: Optional[np.ndarray] = None,
        utilities: np.ndarray = None,
        device: str = "cuda",
    ) -> Dict[str, float]:
        """Evaluate probe performance on text or pre-computed embeddings."""
        if embeddings is None:
            assert texts is not None
            embeddings = self.encode_texts(texts)

        pred_means, pred_vars = self.probe.predict(embeddings, device)

        # R²
        ss_res = np.sum((utilities - pred_means) ** 2)
        ss_tot = np.sum((utilities - utilities.mean()) ** 2)
        r2 = 1.0 - ss_res / max(ss_tot, 1e-8)

        # MAE
        mae = float(np.mean(np.abs(utilities - pred_means)))

        # Gate accuracy
        pred_trigger = pred_means > self.utility_threshold
        true_trigger = utilities > self.utility_threshold
        gate_acc = float(np.mean(pred_trigger == true_trigger))

        tp = np.sum(pred_trigger & true_trigger)
        fp = np.sum(pred_trigger & ~true_trigger)
        fn = np.sum(~pred_trigger & true_trigger)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)

        return {
            "r2": float(r2),
            "mae": mae,
            "gate_accuracy": gate_acc,
            "gate_precision": float(precision),
            "gate_recall": float(recall),
        }

    def make_gate_decision(
        self,
        text: str,
        device: str = "cuda",
    ) -> bool:
        """Decide whether to trigger a rollout based on a text description."""
        emb = self.encode_texts([text])
        means, _ = self.probe.predict(emb, device)
        return bool(means[0] > self.utility_threshold)

    def save(self, path: str):
        if self.probe is not None:
            self.probe.save(path)

    def load(self, path: str, device: str = "cuda"):
        if self.probe is None:
            self._ensure_encoder()
            self.probe = VOCProbeHead(
                input_dim=self._embedding_dim,
                d_hidden=self.d_hidden,
                lr=self.lr,
            )
        self.probe.load(path, device)
