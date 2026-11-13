"""
ahdm.py
=======
Adaptive Hybrid Detection Model (AHDM)

Architecture
------------
  Stage 1 : Isolation Forest  → anomaly score s_IF  (path-length based)
  Stage 2 : Denoising AE      → anomaly score s_AE  (reconstruction MSE)
  Stage 3 : Meta-Learner (LR) → learns optimal fusion weight α
  Stage 4 : Adaptive threshold T_p = percentile(s_fused, p*)

Decision: y_hat = 1 if s_fused > T_p else 0

Usage
-----
    from ahdm import AHDM

    model = AHDM(input_dim=15, n_estimators=100, contamination=0.11)
    model.fit(X_train_dp, y_val=y_val, X_val=X_val_dp)
    y_pred = model.predict(X_test_dp)
    scores = model.decision_scores(X_test_dp)
    model.print_summary()

Author : Pradeep Deshpande
"""

from __future__ import annotations

import numpy as np
import logging
from typing import Optional, Tuple, Dict

from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

from autoencoder import DenoisingAutoencoder

logger = logging.getLogger(__name__)


class AHDM:
    """
    Adaptive Hybrid Detection Model (AHDM).

    Fuses Isolation Forest and Denoising Autoencoder scores via a
    Logistic Regression stacking meta-learner.  Threshold is selected
    by maximising F1-score on a validation split.

    Parameters
    ----------
    input_dim      : int   — number of input features (required for DAE)
    n_estimators   : int   — Isolation Forest trees (default 100)
    contamination  : float or 'auto' — expected anomaly fraction
    ae_epochs      : int   — DAE training epochs (default 50)
    ae_dropout     : float — DAE dropout probability
    ae_noise_factor: float — Gaussian noise factor added during DAE training
    meta_C         : float — LR regularisation (default 1.0)
    threshold_percentile : float or None
        If float, fixed anomaly score percentile for thresholding.
        If None, threshold is tuned on validation set.
    alpha_init     : float — initial IF weight (before meta-learner trains)
    random_state   : int
    """

    def __init__(
        self,
        input_dim: int,
        n_estimators: int = 100,
        contamination: float | str = "auto",
        ae_epochs: int = 50,
        ae_dropout: float = 0.1,
        ae_noise_factor: float = 0.1,
        meta_C: float = 1.0,
        threshold_percentile: Optional[float] = None,
        alpha_init: float = 0.5,
        random_state: int = 42,
        allow_fallback: bool = False,
    ) -> None:
        self.input_dim             = input_dim
        self.n_estimators          = n_estimators
        self.contamination         = contamination
        self.ae_epochs             = ae_epochs
        self.ae_dropout            = ae_dropout
        self.ae_noise_factor       = ae_noise_factor
        self.meta_C                = meta_C
        self.threshold_percentile  = threshold_percentile
        self.alpha_init            = alpha_init
        self.random_state          = random_state

        # Learned weights (set during fit)
        self.alpha_: float = alpha_init
        self.threshold_: float = 0.5
        self.is_fitted_: bool = False

        # Component models
        self._if = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
        )
        self._dae = DenoisingAutoencoder(
            input_dim=input_dim,
            epochs=ae_epochs,
            dropout=ae_dropout,
            noise_factor=ae_noise_factor,
            random_state=random_state,
            allow_fallback=allow_fallback,
        )
        self._meta = LogisticRegression(
            C=meta_C, solver="lbfgs", max_iter=500, random_state=random_state
        )
        self._score_scaler = MinMaxScaler()   # normalise raw scores to [0,1]

        # Diagnostics
        self._fit_stats: Dict = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: int = 1,
    ) -> "AHDM":
        """
        Train AHDM on DP-protected training data.

        Parameters
        ----------
        X_train : (n, d) array — training features (DP-protected, scaled [0,1])
        X_val   : (m, d) array — optional validation features for threshold tuning
        y_val   : (m,)   array — optional validation labels (0=normal, 1=anomaly)
                  Required for meta-learner training and threshold tuning.
        verbose : 0 = silent, 1 = progress logs

        Returns
        -------
        self
        """
        X_train = np.nan_to_num(X_train.astype(np.float32))

        # ── Stage 1: Train Isolation Forest ───────────────────────────────────
        logger.info("Training Isolation Forest (%d trees)…", self.n_estimators)
        self._if.fit(X_train)

        # ── Stage 2: Train Denoising Autoencoder ─────────────────────────────
        logger.info("Training Denoising Autoencoder (max %d epochs)…", self.ae_epochs)
        X_val_dae = X_val if X_val is not None else None
        self._dae.fit(X_train, X_val=X_val_dae, verbose=verbose)

        # ── Raw scores on training data ───────────────────────────────────────
        s_if_train = self._if_scores(X_train)
        s_ae_train = self._dae.anomaly_scores(X_train)

        # Normalise scores to [0,1] for stable fusion
        stacked_train = np.column_stack([s_if_train, s_ae_train])
        stacked_train = self._score_scaler.fit_transform(stacked_train)
        s_if_n, s_ae_n = stacked_train[:, 0], stacked_train[:, 1]

        # ── Stage 3: Meta-Learner ─────────────────────────────────────────────
        if X_val is not None and y_val is not None:
            X_val = np.nan_to_num(X_val.astype(np.float32))
            s_if_val = self._if_scores(X_val)
            s_ae_val = self._dae.anomaly_scores(X_val)
            stacked_val = self._score_scaler.transform(
                np.column_stack([s_if_val, s_ae_val])
            )
            logger.info("Fitting Meta-Learner on validation scores…")
            self._meta.fit(stacked_val, y_val.astype(int))
            # Extract learned alpha from LR coefficients
            coef = np.clip(self._meta.coef_[0], 0.0, None)  # [w_IF, w_AE]; both scores
            # point "higher = more anomalous", so a negative weight is noise: clip to 0.
            total = coef.sum()
            self.alpha_ = float(coef[0] / total) if total > 1e-9 else 0.5
            logger.info(
                "Meta-Learner converged: α=%.3f (IF weight), (1-α)=%.3f (AE weight)",
                self.alpha_, 1.0 - self.alpha_,
            )

            # ── Stage 4: Adaptive threshold ───────────────────────────────────
            if self.threshold_percentile is not None:
                s_fused_val = (
                    self.alpha_ * stacked_val[:, 0]
                    + (1.0 - self.alpha_) * stacked_val[:, 1]
                )
                self.threshold_ = float(
                    np.percentile(s_fused_val, self.threshold_percentile)
                )
            else:
                self.threshold_ = self._tune_threshold(stacked_val, y_val)
            logger.info("Adaptive threshold set to %.4f", self.threshold_)
        else:
            # No validation data: use contamination-based threshold
            s_fused_train = (
                self.alpha_ * s_if_n + (1.0 - self.alpha_) * s_ae_n
            )
            cont = (
                0.1 if self.contamination == "auto"
                else float(self.contamination)
            )
            self.threshold_ = float(
                np.percentile(s_fused_train, 100 * (1.0 - cont))
            )
            logger.info(
                "Threshold set from contamination (%.0f%%): %.4f",
                cont * 100, self.threshold_,
            )

        self._fit_stats = {
            "n_train": len(X_train),
            "alpha": self.alpha_,
            "threshold": self.threshold_,
        }
        self.is_fitted_ = True
        return self

    # ── Scoring ───────────────────────────────────────────────────────────────

    def decision_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Compute fused anomaly scores for all samples.

        Returns
        -------
        scores : (n,) array — higher = more anomalous
        """
        self._check_fitted()
        X = np.nan_to_num(X.astype(np.float32))
        s_if = self._if_scores(X)
        s_ae = self._dae.anomaly_scores(X)
        stacked = self._score_scaler.transform(np.column_stack([s_if, s_ae]))
        s_fused = self.alpha_ * stacked[:, 0] + (1.0 - self.alpha_) * stacked[:, 1]
        return s_fused

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Binary prediction: 0 = Normal, 1 = Anomaly.
        """
        return (self.decision_scores(X) >= self.threshold_).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Soft probability estimates using the meta-learner (if fitted).
        Returns (n, 2) array: [P(normal), P(anomaly)].
        """
        self._check_fitted()
        X = np.nan_to_num(X.astype(np.float32))
        s_if = self._if_scores(X)
        s_ae = self._dae.anomaly_scores(X)
        stacked = self._score_scaler.transform(np.column_stack([s_if, s_ae]))
        try:
            return self._meta.predict_proba(stacked)
        except Exception:
            scores = self.alpha_ * stacked[:, 0] + (1.0 - self.alpha_) * stacked[:, 1]
            p_anom = scores / (scores.max() + 1e-9)
            return np.column_stack([1.0 - p_anom, p_anom])

    # ── Component scores (public helpers) ─────────────────────────────────────

    def if_scores(self, X: np.ndarray) -> np.ndarray:
        """Raw (normalised) Isolation Forest anomaly scores."""
        return self._if_scores(X)

    def ae_scores(self, X: np.ndarray) -> np.ndarray:
        """Raw DAE reconstruction error scores."""
        return self._dae.anomaly_scores(X)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _if_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Return Isolation Forest anomaly scores in [0,1] range.
        sklearn returns negative scores; we flip and normalise.
        """
        raw = -self._if.score_samples(X)   # higher = more anomalous
        return raw

    def _tune_threshold(
        self, stacked: np.ndarray, y_val: np.ndarray
    ) -> float:
        """
        Grid-search threshold percentile [80, 99] to maximise F1.
        """
        from sklearn.metrics import f1_score
        best_t, best_f1 = 0.5, -1.0
        s_fused = self.alpha_ * stacked[:, 0] + (1.0 - self.alpha_) * stacked[:, 1]
        for pct in np.linspace(80, 99, 40):
            t = float(np.percentile(s_fused, pct))
            preds = (s_fused >= t).astype(int)
            f1 = f1_score(y_val, preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        logger.info("Optimal threshold %.4f → F1=%.4f", best_t, best_f1)
        return best_t

    def _check_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError("AHDM: call fit() before predict().")

    # ── Summary ───────────────────────────────────────────────────────────────

    def print_summary(self) -> None:
        """Print a concise model summary."""
        print("\n" + "=" * 55)
        print("  ADAPTIVE HYBRID DETECTION MODEL (AHDM) SUMMARY")
        print("=" * 55)
        print(f"  Input dim        : {self.input_dim}")
        print(f"  IF estimators    : {self.n_estimators}")
        print(f"  DAE architecture : {self.input_dim}→128→64→32→64→128→{self.input_dim}")
        print(f"  Fusion weight α  : {self.alpha_:.4f}  (IF)  |  {1-self.alpha_:.4f}  (AE)")
        print(f"  Decision threshold: {self.threshold_:.6f}")
        print(f"  Training samples : {self._fit_stats.get('n_train', '?')}")
        print("=" * 55 + "\n")
