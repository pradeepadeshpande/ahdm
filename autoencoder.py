"""
autoencoder.py
==============
Denoising Autoencoder (DAE) for anomaly scoring on DP-protected cloud logs.

Architecture
------------
    Encoder: d → 128 → 64 → 32  (ReLU, BatchNorm, Dropout p=0.1)
    Bottleneck: 32-dim latent
    Decoder: 32 → 64 → 128 → d  (ReLU; final Sigmoid)

Anomaly score: per-sample MSE reconstruction error s_AE(x) = ||x - x̂||²

Usage
-----
    from autoencoder import DenoisingAutoencoder

    dae = DenoisingAutoencoder(input_dim=15)
    dae.fit(X_train_dp)
    scores = dae.anomaly_scores(X_test_dp)

Author : Pradeep Deshpande
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# ── Try importing PyTorch; fall back to numpy-only stub if unavailable ────────
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
    TORCH_IMPORT_ERROR = None
except (ImportError, OSError) as _exc:       # OSError: Windows DLL load failures
    TORCH_AVAILABLE = False
    TORCH_IMPORT_ERROR = f"{type(_exc).__name__}: {_exc}"
    logger.warning("PyTorch could not be imported (%s).", TORCH_IMPORT_ERROR)


# ─────────────────────────────────────────────────────────────────────────────
#  PyTorch model definition
# ─────────────────────────────────────────────────────────────────────────────

if TORCH_AVAILABLE:

    class _DAEModule(nn.Module):
        """Internal PyTorch module for the Denoising Autoencoder."""

        def __init__(self, input_dim: int, dropout: float = 0.1) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 32),
                nn.ReLU(),
            )
            self.decoder = nn.Sequential(
                nn.Linear(32, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, input_dim),
                nn.Sigmoid(),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            z = self.encoder(x)
            return self.decoder(z)

        def encode(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.encoder(x)


# ─────────────────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────────────────

class DenoisingAutoencoder:
    """
    Denoising Autoencoder for anomaly detection on DP-protected data.

    Parameters
    ----------
    input_dim   : int    — number of input features
    lr          : float  — Adam learning rate (default 1e-3)
    epochs      : int    — max training epochs (default 50)
    batch_size  : int    — mini-batch size (default 64)
    dropout     : float  — dropout probability (default 0.1)
    patience    : int    — early stopping patience (default 10)
    noise_factor: float  — additional Gaussian noise for denoising (default 0.1)
    device      : str    — 'cpu', 'cuda', or 'auto'
    random_state: int
    """

    def __init__(
        self,
        input_dim: int,
        lr: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 64,
        dropout: float = 0.1,
        patience: int = 10,
        noise_factor: float = 0.1,
        device: str = "auto",
        random_state: int = 42,
        allow_fallback: bool = False,
    ) -> None:
        self.input_dim    = input_dim
        self.lr           = lr
        self.epochs       = epochs
        self.batch_size   = batch_size
        self.dropout      = dropout
        self.patience     = patience
        self.noise_factor = noise_factor
        self.random_state = random_state
        self.is_fitted_   = False
        self.train_losses_: list[float] = []
        self.val_losses_:   list[float] = []

        self.backend = "torch" if TORCH_AVAILABLE else "pca-fallback"
        if not TORCH_AVAILABLE:
            if not allow_fallback:
                raise ImportError(
                    "PyTorch is required for the Denoising Autoencoder. Install it, or "
                    "pass allow_fallback=True to knowingly use the PCA stand-in "
                    "(results then do NOT correspond to the DAE described in the paper)."
                )
            logger.warning("Using NumPy PCA-based fallback (install PyTorch for full DAE).")
            return

        if device == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        torch.manual_seed(random_state)
        self._model = _DAEModule(input_dim, dropout).to(self._device)
        self._criterion = nn.MSELoss()
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)
        logger.info("DAE initialised on %s", self._device)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _to_tensor(self, X: np.ndarray) -> "torch.Tensor":
        return torch.tensor(X, dtype=torch.float32).to(self._device)

    def _add_noise(self, x: "torch.Tensor") -> "torch.Tensor":
        """Add Gaussian noise to simulate DP-like corruption during training."""
        return x + self.noise_factor * torch.randn_like(x)

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        verbose: int = 1,
    ) -> "DenoisingAutoencoder":
        """
        Train the DAE on DP-protected training data.

        Parameters
        ----------
        X_train  : (n, d) numpy array — DP-protected training features (normalised [0,1])
        X_val    : (m, d) numpy array — optional validation set for early stopping
        verbose  : 0 = silent, 1 = epoch summary

        Returns
        -------
        self
        """
        if not TORCH_AVAILABLE:
            self._fit_pca_fallback(X_train)
            return self

        X_train = np.nan_to_num(X_train).astype(np.float32)
        train_tensor = self._to_tensor(X_train)
        train_loader = DataLoader(
            TensorDataset(train_tensor, train_tensor),
            batch_size=self.batch_size,
            shuffle=True,
        )

        has_val = X_val is not None
        if has_val:
            X_val = np.nan_to_num(X_val).astype(np.float32)
            val_tensor = self._to_tensor(X_val)

        best_val_loss = float("inf")
        patience_counter = 0
        best_weights = None

        for epoch in range(1, self.epochs + 1):
            # ── Training ──────────────────────────────────────────────────────
            self._model.train()
            epoch_loss = 0.0
            for x_batch, _ in train_loader:
                x_noisy = self._add_noise(x_batch)
                x_recon = self._model(x_noisy)
                loss = self._criterion(x_recon, x_batch)
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()
                epoch_loss += loss.item() * len(x_batch)
            epoch_loss /= len(X_train)
            self.train_losses_.append(epoch_loss)

            # ── Validation ────────────────────────────────────────────────────
            if has_val:
                self._model.eval()
                with torch.no_grad():
                    val_recon = self._model(val_tensor)
                    val_loss = self._criterion(val_recon, val_tensor).item()
                self.val_losses_.append(val_loss)

                if val_loss < best_val_loss - 1e-6:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_weights = {
                        k: v.clone() for k, v in self._model.state_dict().items()
                    }
                else:
                    patience_counter += 1

                if verbose and epoch % 10 == 0:
                    logger.info(
                        "Epoch %3d | train_loss=%.5f | val_loss=%.5f | patience=%d",
                        epoch, epoch_loss, val_loss, patience_counter,
                    )

                if patience_counter >= self.patience:
                    logger.info("Early stopping at epoch %d.", epoch)
                    break
            else:
                if verbose and epoch % 10 == 0:
                    logger.info("Epoch %3d | train_loss=%.5f", epoch, epoch_loss)

        # Restore best weights
        if best_weights is not None:
            self._model.load_state_dict(best_weights)

        self.is_fitted_ = True
        return self

    # ── NumPy PCA fallback ────────────────────────────────────────────────────

    def _fit_pca_fallback(self, X: np.ndarray) -> None:
        """Minimal PCA-based fallback when PyTorch is unavailable."""
        from sklearn.decomposition import PCA
        n_comp = min(32, X.shape[1])
        self._pca = PCA(n_components=n_comp, random_state=self.random_state)
        self._pca.fit(X)
        self.is_fitted_ = True
        logger.info("PCA fallback fitted (n_components=%d).", n_comp)

    def _score_pca_fallback(self, X: np.ndarray) -> np.ndarray:
        X_proj = self._pca.transform(X)
        X_recon = self._pca.inverse_transform(X_proj)
        return np.mean((X - X_recon) ** 2, axis=1)

    # ── Anomaly scores ────────────────────────────────────────────────────────

    def anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Compute per-sample reconstruction MSE: s_AE(x) = ||x - x̂||².

        Parameters
        ----------
        X : (n, d) numpy array — DP-protected features (same scale as training)

        Returns
        -------
        scores : (n,) numpy array — higher = more anomalous
        """
        if not self.is_fitted_:
            raise RuntimeError("Call fit() before anomaly_scores().")

        if not TORCH_AVAILABLE:
            return self._score_pca_fallback(X)

        X = np.nan_to_num(X).astype(np.float32)
        self._model.eval()
        with torch.no_grad():
            X_tensor = self._to_tensor(X)
            X_recon  = self._model(X_tensor).cpu().numpy()
        scores = np.mean((X - X_recon) ** 2, axis=1)
        return scores

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """Return reconstructed features (for qualitative inspection)."""
        if not self.is_fitted_:
            raise RuntimeError("Call fit() before reconstruct().")
        if not TORCH_AVAILABLE:
            return self._pca.inverse_transform(self._pca.transform(X))
        X = np.nan_to_num(X).astype(np.float32)
        self._model.eval()
        with torch.no_grad():
            return self._model(self._to_tensor(X)).cpu().numpy()

    def latent_representation(self, X: np.ndarray) -> np.ndarray:
        """Return 32-dim bottleneck representation."""
        if not TORCH_AVAILABLE:
            return self._pca.transform(X)
        X = np.nan_to_num(X).astype(np.float32)
        self._model.eval()
        with torch.no_grad():
            return self._model.encode(self._to_tensor(X)).cpu().numpy()
