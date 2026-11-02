"""Classical statistical anomaly detector (WP5 baseline).

This module implements a conventional, non-learning baseline for anomaly
detection. It is intentionally designed to be *scalable and flexible* rather
than hard-wired to a single rule:

* The scoring method is pluggable (see ``SCORING_STRATEGIES``). Adding a new
  statistical method only requires registering a new strategy, without
  touching the detector logic or the public interface.
* The decision threshold can be established in several documented ways
  (data-driven percentile, a multiple of the score's standard deviation, an
  expected contamination rate, or a fixed value).

The detector follows the shared :class:`AnomalyDetector` interface and the
project convention ``0 = normal``, ``1 = anomalous``. Higher anomaly scores
indicate more anomalous observations, consistent with the AI/ML detector
(WP6) so the two can be compared fairly by WP7.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from interface import AnomalyDetector

# A scoring strategy maps fitted statistics + input X to a 1D array of scores
# (one score per observation, higher = more anomalous).
ScoringStrategy = Callable[["StatisticalAnomalyDetector", np.ndarray], np.ndarray]

_EPSILON = 1e-12


class StatisticalAnomalyDetector(AnomalyDetector):
    """Multivariate statistical anomaly detector.

    Parameters
    ----------
    scoring:
        Name of the registered scoring strategy. Built-in options:

        * ``"robust_zscore"`` (default): per-feature robust z-score based on
          the median and the Median Absolute Deviation (MAD). The per-feature
          deviations are aggregated into a single score (see ``aggregation``).
          Robust to outliers in the training data.
        * ``"zscore"``: classic per-feature z-score using mean and standard
          deviation, aggregated the same way.
        * ``"mahalanobis"``: multivariate Mahalanobis distance using the
          empirical covariance matrix. Accounts for correlations between
          features.

    threshold_method:
        How the decision threshold on the anomaly score is established:

        * ``"percentile"`` (default): threshold = the ``percentile`` of the
          scores observed on the (assumed mostly normal) training data.
        * ``"sigma"``: threshold = mean(score) + ``sigma`` * std(score)
          over the training scores.
        * ``"contamination"``: threshold chosen so that approximately the
          ``contamination`` fraction of the training data is flagged.
        * ``"fixed"``: use ``fixed_threshold`` directly.

    percentile:
        Percentile (0-100) used when ``threshold_method="percentile"``.
    sigma:
        Multiplier used when ``threshold_method="sigma"``.
    contamination:
        Expected fraction of anomalies (0-1) when
        ``threshold_method="contamination"``.
    fixed_threshold:
        Threshold used when ``threshold_method="fixed"``.
    aggregation:
        How per-feature deviations are combined for the per-feature scoring
        strategies: ``"max"`` (most sensitive, flags a single bad feature) or
        ``"mean"`` (averages deviations). Ignored for ``"mahalanobis"``.
    """

    SCORING_STRATEGIES: dict[str, ScoringStrategy] = {}

    def __init__(
        self,
        scoring: str = "robust_zscore",
        threshold_method: str = "percentile",
        percentile: float = 99.0,
        sigma: float = 3.0,
        contamination: float = 0.05,
        fixed_threshold: float | None = None,
        aggregation: str = "max",
    ) -> None:
        if scoring not in self.SCORING_STRATEGIES:
            raise ValueError(
                f"Unknown scoring strategy {scoring!r}. "
                f"Available: {sorted(self.SCORING_STRATEGIES)}"
            )
        if threshold_method not in {"percentile", "sigma", "contamination", "fixed"}:
            raise ValueError(f"Unknown threshold_method {threshold_method!r}.")
        if aggregation not in {"max", "mean"}:
            raise ValueError(f"Unknown aggregation {aggregation!r}.")
        if not 0.0 < percentile <= 100.0:
            raise ValueError("percentile must be in (0, 100].")
        if not 0.0 <= contamination < 1.0:
            raise ValueError("contamination must be in [0, 1).")
        if threshold_method == "fixed" and fixed_threshold is None:
            raise ValueError(
                "fixed_threshold must be provided when threshold_method='fixed'."
            )

        self.scoring = scoring
        self.threshold_method = threshold_method
        self.percentile = percentile
        self.sigma = sigma
        self.contamination = contamination
        self.fixed_threshold = fixed_threshold
        self.aggregation = aggregation

        # Fitted statistics (set in fit()).
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.median_: np.ndarray | None = None
        self.mad_: np.ndarray | None = None
        self.cov_inv_: np.ndarray | None = None
        self.n_features_: int | None = None
        self.threshold_: float | None = None
        self._is_fitted = False

    # ------------------------------------------------------------------ #
    # Input validation
    # ------------------------------------------------------------------ #
    @staticmethod
    def _validate_input(X: np.ndarray) -> np.ndarray:
        """Validate and convert input features to a finite 2D float array."""
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("Input must be a 2D array of shape (n_samples, n_features).")

        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("Input must contain at least one sample and one feature.")

        if not np.isfinite(X).all():
            raise ValueError(
                "Input contains NaN or infinite values. "
                "Preprocess the data before detection."
            )

        return X

    def _check_feature_count(self, X: np.ndarray) -> None:
        if self.n_features_ is not None and X.shape[1] != self.n_features_:
            raise ValueError(
                f"Expected {self.n_features_} features, received {X.shape[1]}. "
                "Feature order and count must match the fitted data."
            )

    # ------------------------------------------------------------------ #
    # Fit
    # ------------------------------------------------------------------ #
    def fit(self, X: np.ndarray) -> "StatisticalAnomalyDetector":
        """Learn the normal-behaviour statistics from development data.

        The training data is assumed to be predominantly normal. Robust
        statistics (median/MAD) reduce the influence of any contamination.
        """
        X = self._validate_input(X)
        self.n_features_ = X.shape[1]

        # Classic statistics.
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)

        # Robust statistics.
        self.median_ = np.median(X, axis=0)
        # MAD scaled by 1.4826 approximates the standard deviation for
        # normally distributed data.
        self.mad_ = 1.4826 * np.median(np.abs(X - self.median_), axis=0)

        # Covariance inverse for Mahalanobis. Regularise lightly for stability.
        if self.scoring == "mahalanobis":
            cov = np.cov(X, rowvar=False)
            cov = np.atleast_2d(cov)
            cov += np.eye(cov.shape[0]) * _EPSILON
            self.cov_inv_ = np.linalg.pinv(cov)

        self._is_fitted = True

        # Establish the decision threshold from the training scores.
        train_scores = self._raw_scores(X)
        self.threshold_ = self._compute_threshold(train_scores)
        return self

    # ------------------------------------------------------------------ #
    # Scoring
    # ------------------------------------------------------------------ #
    def _raw_scores(self, X: np.ndarray) -> np.ndarray:
        """Dispatch to the configured scoring strategy."""
        strategy = self.SCORING_STRATEGIES[self.scoring]
        return strategy(self, X)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly scores (higher = more anomalous)."""
        if not self._is_fitted:
            raise RuntimeError("Call fit() before score_samples().")
        X = self._validate_input(X)
        self._check_feature_count(X)
        return self._raw_scores(X)

    def _aggregate(self, per_feature: np.ndarray) -> np.ndarray:
        if self.aggregation == "max":
            return per_feature.max(axis=1)
        return per_feature.mean(axis=1)

    def _score_robust_zscore(self, X: np.ndarray) -> np.ndarray:
        assert self.median_ is not None and self.mad_ is not None
        deviations = np.abs(X - self.median_) / (self.mad_ + _EPSILON)
        return self._aggregate(deviations)

    def _score_zscore(self, X: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None and self.std_ is not None
        deviations = np.abs(X - self.mean_) / (self.std_ + _EPSILON)
        return self._aggregate(deviations)

    def _score_mahalanobis(self, X: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None and self.cov_inv_ is not None
        centered = X - self.mean_
        # sqrt of the squared Mahalanobis distance, computed row-wise.
        md_sq = np.einsum("ij,jk,ik->i", centered, self.cov_inv_, centered)
        md_sq = np.clip(md_sq, 0.0, None)
        return np.sqrt(md_sq)

    # ------------------------------------------------------------------ #
    # Threshold
    # ------------------------------------------------------------------ #
    def _compute_threshold(self, train_scores: np.ndarray) -> float:
        if self.threshold_method == "percentile":
            return float(np.percentile(train_scores, self.percentile))
        if self.threshold_method == "sigma":
            return float(train_scores.mean() + self.sigma * train_scores.std())
        if self.threshold_method == "contamination":
            # Flag the top `contamination` fraction of training scores.
            pct = 100.0 * (1.0 - self.contamination)
            return float(np.percentile(train_scores, pct))
        # fixed
        assert self.fixed_threshold is not None
        return float(self.fixed_threshold)

    # ------------------------------------------------------------------ #
    # Predict
    # ------------------------------------------------------------------ #
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict per observation. Returns 0 (normal) or 1 (anomalous)."""
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict().")
        assert self.threshold_ is not None
        scores = self.score_samples(X)
        return np.where(scores > self.threshold_, 1, 0).astype(int)

    # ------------------------------------------------------------------ #
    # Metadata
    # ------------------------------------------------------------------ #
    def get_metadata(self) -> dict[str, Any]:
        """Return detector configuration and fitted status."""
        return {
            "name": self.__class__.__name__,
            "type": "classical",
            "scoring": self.scoring,
            "aggregation": self.aggregation,
            "threshold_method": self.threshold_method,
            "percentile": self.percentile,
            "sigma": self.sigma,
            "contamination": self.contamination,
            "fixed_threshold": self.fixed_threshold,
            "n_features": self.n_features_,
            "threshold": self.threshold_,
            "is_fitted": self._is_fitted,
            "available_scoring_strategies": sorted(self.SCORING_STRATEGIES),
        }


# Register the built-in scoring strategies. Adding a new method is as simple
# as writing a function/method and registering it here (scalable by design).
StatisticalAnomalyDetector.SCORING_STRATEGIES.update(
    {
        "robust_zscore": StatisticalAnomalyDetector._score_robust_zscore,
        "zscore": StatisticalAnomalyDetector._score_zscore,
        "mahalanobis": StatisticalAnomalyDetector._score_mahalanobis,
    }
)
