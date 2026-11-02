"""Synthetic dataset generator for developing and testing the WP5 detector.

IMPORTANT
---------
The data produced here is **synthetic**. It does NOT come from the ESA
robotic platform, a simulator, or any real sensor. It exists only so the
classical detector can be developed, exercised, and demonstrated before the
real dataset from WP4 is available. Any results obtained on this data describe
behaviour on synthetic signals, not real robotic performance (see
``docs/threat_model.md`` section 9 and ``docs/data_schema.md`` section 9).

The generator produces observations that loosely mirror the candidate anomaly
scenarios from the threat model:

* normal operation: measurements inside expected ranges with mild noise;
* trajectory / value deviation: an offset pushes features out of range;
* sensor spike: an abrupt large jump in one or more features.

Features are intentionally abstract (``feature_0 .. feature_{m-1}``) because
the real feature set is still an open decision in ``docs/data_schema.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Convention shared across the project.
LABEL_NORMAL = 0
LABEL_ANOMALOUS = 1


@dataclass
class SyntheticDataset:
    """A synthetic dataset split into development and evaluation parts.

    ``X_*`` are 2D float arrays of shape (n_samples, n_features).
    ``y_*`` are 1D int arrays of ground-truth labels (0 normal, 1 anomalous).
    ``X_dev`` contains only normal observations (unsupervised fit on normal
    behaviour); ``X_eval`` mixes normal and anomalous observations.
    """

    X_dev: np.ndarray
    y_dev: np.ndarray
    X_eval: np.ndarray
    y_eval: np.ndarray
    feature_names: list[str]
    is_synthetic: bool = True


def generate_synthetic_dataset(
    n_features: int = 6,
    n_dev: int = 800,
    n_eval_normal: int = 400,
    n_eval_anomalous: int = 100,
    noise_std: float = 1.0,
    deviation_offset: float = 6.0,
    spike_magnitude: float = 12.0,
    random_state: int = 42,
) -> SyntheticDataset:
    """Generate a reproducible synthetic dataset.

    Parameters
    ----------
    n_features:
        Number of features per observation.
    n_dev:
        Number of normal observations for development (fit).
    n_eval_normal, n_eval_anomalous:
        Number of normal / anomalous observations in the evaluation split.
    noise_std:
        Standard deviation of the Gaussian noise around the normal mean.
    deviation_offset:
        Offset (in units of the normal signal) applied to produce
        trajectory/value-deviation anomalies.
    spike_magnitude:
        Magnitude of the abrupt spike applied to produce sensor-spike
        anomalies.
    random_state:
        Seed for reproducibility.
    """
    rng = np.random.default_rng(random_state)

    # A fixed "nominal" mean per feature so the signal is not centred at zero.
    nominal_mean = rng.uniform(-2.0, 2.0, size=n_features)

    def _normal(n: int) -> np.ndarray:
        return nominal_mean + rng.normal(0.0, noise_std, size=(n, n_features))

    # Development data: normal behaviour only.
    X_dev = _normal(n_dev)
    y_dev = np.full(n_dev, LABEL_NORMAL, dtype=int)

    # Evaluation normal observations.
    X_eval_normal = _normal(n_eval_normal)

    # Evaluation anomalies: split between deviation and spike types.
    n_dev_anom = n_eval_anomalous // 2
    n_spike_anom = n_eval_anomalous - n_dev_anom

    # Type 1 - deviation: a sustained offset on a subset of features.
    dev_anom = _normal(n_dev_anom)
    for row in dev_anom:
        k = rng.integers(1, n_features + 1)
        idx = rng.choice(n_features, size=k, replace=False)
        sign = rng.choice([-1.0, 1.0], size=k)
        row[idx] += sign * deviation_offset * noise_std

    # Type 2 - spike: one abrupt large jump in a single feature.
    spike_anom = _normal(n_spike_anom)
    for row in spike_anom:
        idx = rng.integers(0, n_features)
        sign = rng.choice([-1.0, 1.0])
        row[idx] += sign * spike_magnitude * noise_std

    X_eval = np.vstack([X_eval_normal, dev_anom, spike_anom])
    y_eval = np.concatenate(
        [
            np.full(n_eval_normal, LABEL_NORMAL, dtype=int),
            np.full(n_dev_anom + n_spike_anom, LABEL_ANOMALOUS, dtype=int),
        ]
    )

    # Shuffle the evaluation split so ordering carries no label information.
    perm = rng.permutation(X_eval.shape[0])
    X_eval = X_eval[perm]
    y_eval = y_eval[perm]

    feature_names = [f"feature_{i}" for i in range(n_features)]

    return SyntheticDataset(
        X_dev=X_dev,
        y_dev=y_dev,
        X_eval=X_eval,
        y_eval=y_eval,
        feature_names=feature_names,
        is_synthetic=True,
    )
