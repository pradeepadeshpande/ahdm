"""
dp_engine.py
============
Differential Privacy Engine — Adaptive Laplace Mechanism for Cloud Logs.

Implements:
  - Sensitivity classification (High / Moderate / Low)
  - Per-attribute adaptive epsilon allocation
  - Laplace noise perturbation
  - (epsilon, delta)-DP moments-accountant composition tracking

Usage
-----
    from dp_engine import DPEngine

    engine = DPEngine(base_epsilon=1.0, delta=1e-5)
    engine.register_schema(schema)          # declare attribute sensitivity
    X_dp = engine.perturb(X_raw, columns)  # perturb a DataFrame
    engine.print_budget_summary()

Author : Pradeep Deshpande
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import warnings
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ── Sensitivity levels ────────────────────────────────────────────────────────

class SensitivityLevel:
    HIGH     = "high"       # user ID, email, IP — epsilon multiplier 0.5x
    MODERATE = "moderate"   # session dur, tx amount — epsilon multiplier 1.0x
    LOW      = "low"        # log level, event type — epsilon multiplier 2.0x


@dataclass
class AttributeSpec:
    name: str
    sensitivity: str = SensitivityLevel.MODERATE
    delta_f: Optional[float] = None   # custom global sensitivity; None = estimate from data


# ── Main engine ───────────────────────────────────────────────────────────────

class DPEngine:
    """
    Adaptive Differential Privacy Engine using the Laplace mechanism.

    Parameters
    ----------
    base_epsilon : float
        Base privacy budget for MODERATE sensitivity attributes.
    delta : float
        Failure probability for (epsilon, delta)-DP composition.
    sensitivity_percentile : float
        Data percentile used to estimate global sensitivity Δf
        when not explicitly provided (default 99th).
    clip_output : bool
        Whether to clip perturbed values to [0, original_max].
    random_state : int or None
    """

    _EPSILON_MULTIPLIERS: Dict[str, float] = {
        SensitivityLevel.HIGH:     0.5,
        SensitivityLevel.MODERATE: 1.0,
        SensitivityLevel.LOW:      2.0,
    }

    def __init__(
        self,
        base_epsilon: float = 1.0,
        delta: float = 1e-5,
        sensitivity_percentile: float = 99.0,
        clip_output: bool = True,
        random_state: Optional[int] = 42,
    ) -> None:
        if base_epsilon <= 0:
            raise ValueError("base_epsilon must be positive.")
        self.base_epsilon          = base_epsilon
        self.delta                 = delta
        self.sensitivity_percentile = sensitivity_percentile
        self.clip_output           = clip_output
        self.rng                   = np.random.default_rng(random_state)

        # Schema: attribute_name -> AttributeSpec
        self._schema: Dict[str, AttributeSpec] = {}

        # Budget accounting: moments accountant (simplified log-moments)
        self._privacy_losses: List[float] = []     # per-query epsilon
        self._total_epsilon_used: float = 0.0

    # ── Schema registration ───────────────────────────────────────────────────

    def register_attribute(
        self,
        name: str,
        sensitivity: str = SensitivityLevel.MODERATE,
        delta_f: Optional[float] = None,
    ) -> None:
        """Register a single attribute with its sensitivity class."""
        self._schema[name] = AttributeSpec(name, sensitivity, delta_f)
        logger.info("Registered attribute '%s' [%s]", name, sensitivity)

    def register_schema(self, schema: Dict[str, str]) -> None:
        """
        Register multiple attributes at once.

        Parameters
        ----------
        schema : dict
            {attribute_name: sensitivity_level}
        """
        for name, sens in schema.items():
            self.register_attribute(name, sens)

    # ── Sensitivity estimation ────────────────────────────────────────────────

    def _estimate_delta_f(self, series: pd.Series) -> float:
        """Estimate global sensitivity as the p-th percentile of the range."""
        lo = np.percentile(series.dropna(), 100 - self.sensitivity_percentile)
        hi = np.percentile(series.dropna(), self.sensitivity_percentile)
        delta_f = float(hi - lo)
        if delta_f <= 0:
            warnings.warn(
                f"Estimated Δf=0 for attribute; skipping DP protection.",
                RuntimeWarning,
            )
        return delta_f

    # ── Epsilon per attribute ─────────────────────────────────────────────────

    def _effective_epsilon(self, attr_name: str) -> float:
        """Return the effective epsilon for a registered attribute."""
        spec = self._schema.get(attr_name)
        if spec is None:
            logger.warning(
                "Attribute '%s' not in schema; using base epsilon.", attr_name
            )
            return self.base_epsilon
        mult = self._EPSILON_MULTIPLIERS.get(spec.sensitivity, 1.0)
        return self.base_epsilon * mult

    # ── Laplace noise ─────────────────────────────────────────────────────────

    def _laplace_noise(self, n: int, delta_f: float, epsilon: float) -> np.ndarray:
        """Sample Laplace noise: Lap(0, Δf/ε)."""
        scale = delta_f / epsilon
        return self.rng.laplace(loc=0.0, scale=scale, size=n)

    # ── Core perturbation ─────────────────────────────────────────────────────

    def perturb_series(
        self,
        series: pd.Series,
        attr_name: str,
    ) -> pd.Series:
        """
        Perturb a single numeric Series with Laplace noise.

        Returns
        -------
        pd.Series  (DP-protected values, same index as input)
        """
        spec = self._schema.get(attr_name)

        # Determine Δf
        if spec and spec.delta_f is not None:
            delta_f = spec.delta_f
        else:
            delta_f = self._estimate_delta_f(series)

        if delta_f <= 0:
            return series.copy()   # skip zero-sensitivity attributes

        eps = self._effective_epsilon(attr_name)
        noise = self._laplace_noise(len(series), delta_f, eps)
        perturbed = series.values.astype(float) + noise

        if self.clip_output:
            original_max = float(series.max())
            perturbed = np.clip(perturbed, 0.0, original_max * 1.5)

        # Track budget
        self._privacy_losses.append(eps)
        self._total_epsilon_used += eps

        return pd.Series(perturbed, index=series.index, name=series.name)

    def perturb(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Perturb all registered numeric columns in a DataFrame.

        Parameters
        ----------
        df      : input DataFrame (unchanged)
        columns : explicit list of columns to perturb; if None, uses schema

        Returns
        -------
        pd.DataFrame with perturbed columns; non-sensitive columns copied as-is.
        """
        df_dp = df.copy()
        target_cols = columns if columns else list(self._schema.keys())

        for col in target_cols:
            if col not in df.columns:
                logger.warning("Column '%s' not found in DataFrame; skipping.", col)
                continue
            if not pd.api.types.is_numeric_dtype(df[col]):
                logger.info("Column '%s' is non-numeric; skipping DP.", col)
                continue

            spec = self._schema.get(col)
            if spec and spec.sensitivity == SensitivityLevel.LOW:
                logger.info("Column '%s' is LOW sensitivity; skipping DP.", col)
                continue

            df_dp[col] = self.perturb_series(df[col], col)
            logger.info(
                "Perturbed column '%s' [ε=%.2f]", col, self._effective_epsilon(col)
            )

        return df_dp

    # ── Matrix perturbation (used by run_experiments.py) ──────────────────────

    def allocate_epsilons(self, columns: List[str]) -> Dict[str, float]:
        """Per-attribute epsilon_j = base_epsilon * multiplier(sensitivity class)."""
        return {c: self._effective_epsilon(c) for c in columns}

    def perturb_matrix(self, X: np.ndarray, columns: List[str],
                       allocation: str = "adaptive") -> np.ndarray:
        """
        Local Laplace perturbation of a feature matrix already scaled to [0, 1].

        * Every attribute is perturbed, including LOW-sensitivity ones
          (they simply receive a larger epsilon).
        * Bounds [0, 1] are treated as public, so the global sensitivity of each
          attribute is exactly 1 (no data-dependent estimate).
        * Output is clipped back to [0, 1] (post-processing, free of privacy cost).
        * allocation: 'adaptive' (sensitivity-class budgets, default), 'uniform'
          (epsilon_j = base_epsilon for all attributes) or 'matched' (uniform
          epsilon_j with the same per-record total as 'adaptive'); the last two
          exist only for the ablation study.
        * Accounting: a record releases len(columns) noisy attributes, so by
          sequential composition its privacy loss is sum_j epsilon_j. Each
          record is released once, so parallel composition applies across
          records.  Call once per record set (train / val / test).
        """
        eps = self.allocate_epsilons(columns)
        if allocation == "uniform":          # ablation: every attribute gets the base epsilon
            eps = {c: self.base_epsilon for c in columns}
        elif allocation == "matched":        # ablation: uniform split of the SAME per-record total
            total = sum(eps.values())
            eps = {c: total / len(columns) for c in columns}
        elif allocation != "adaptive":
            raise ValueError(f"unknown allocation '{allocation}'")
        X = np.clip(np.asarray(X, dtype=float), 0.0, 1.0)
        out = np.empty_like(X)
        for j, c in enumerate(columns):
            out[:, j] = X[:, j] + self.rng.laplace(0.0, 1.0 / eps[c], size=X.shape[0])
        self._record_epsilon = float(sum(eps.values()))
        return np.clip(out, 0.0, 1.0)

    @property
    def record_epsilon(self) -> float:
        """Per-record privacy loss of the last perturb_matrix call (sum of epsilon_j)."""
        return getattr(self, "_record_epsilon", 0.0)

    # ── Budget accounting ─────────────────────────────────────────────────────

    def composed_epsilon(self, method: str = "advanced") -> float:
        """
        Compute composed epsilon over all queries.

        method : 'basic'    → simple sum (loose bound)
                 'advanced' → sqrt(2k ln(1/δ)) * ε   (moments accountant approx.)
        """
        if not self._privacy_losses:
            return 0.0
        k = len(self._privacy_losses)
        eps_max = max(self._privacy_losses)
        if method == "basic":
            return sum(self._privacy_losses)
        # Advanced composition (homogeneous upper bound using eps_max).  It is only
        # a bound, and can exceed the basic sum, so report the tighter of the two.
        advanced = (
            np.sqrt(2 * k * np.log(1.0 / self.delta)) * eps_max
            + k * eps_max * (np.exp(eps_max) - 1)
        )
        return float(min(advanced, sum(self._privacy_losses)))

    def print_budget_summary(self) -> None:
        """Print a summary of the privacy budget consumption."""
        print("\n" + "=" * 55)
        print("  DIFFERENTIAL PRIVACY BUDGET SUMMARY")
        print("=" * 55)
        print(f"  Base epsilon         : {self.base_epsilon:.3f}")
        print(f"  Delta                : {self.delta:.2e}")
        print(f"  # Attributes perturbed: {len(self._privacy_losses)}")
        print(f"  Total ε (naive sum)  : {self._total_epsilon_used:.3f}")
        print(f"  Total ε (advanced)   : {self.composed_epsilon('advanced'):.3f}")
        print("=" * 55 + "\n")
