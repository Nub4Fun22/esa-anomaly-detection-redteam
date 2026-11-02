"""Adversarial / evasion testing for the classical anomaly detector (WP5).

Security rationale
------------------
An anomaly detector is a *security control*. To understand where it can be
trusted, we probe it the way a red-team / penetration tester would probe a
defence: we try to make malicious-looking behaviour slip past it. The goal is
strictly defensive - we characterise the detector's blind spots so WP5 can
document them, WP3 can refine the threat scenarios, and WP7 can report them.

These tests do NOT touch any real robot, platform, or network. They operate on
the numeric feature representation only. "Attacks" here mean crafted inputs
that are genuinely anomalous (ground truth = 1) but designed to stay under the
detector's decision threshold (evasion).

Terminology:
* evasion / stealthy anomaly: a real deviation kept just below the threshold.
* evasion rate: fraction of crafted anomalies the detector fails to flag.
  A high evasion rate on a given attack = a documented limitation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from interface import AnomalyDetector


@dataclass
class EvasionResult:
    """Outcome of one evasion experiment."""

    attack_name: str
    n_samples: int
    n_evaded: int  # crafted anomalies predicted as normal (0)
    evasion_rate: float
    description: str
    details: dict = field(default_factory=dict)


def _count_evaded(detector: AnomalyDetector, X_attack: np.ndarray) -> int:
    """Count crafted anomalies that the detector labels as normal."""
    preds = detector.predict(X_attack)
    # Every row of X_attack is intended to be anomalous (ground truth 1),
    # so a prediction of 0 is a successful evasion.
    return int(np.sum(preds == 0))


def slow_drift_attack(
    detector: AnomalyDetector,
    baseline: np.ndarray,
    n_steps: int = 50,
    max_offset: float = 6.0,
    feature_index: int | None = None,
    random_state: int = 0,
) -> EvasionResult:
    """Gradually drift a feature away from its nominal value.

    A sudden large deviation is easy to catch; a slow drift mimics an attacker
    nudging a trajectory a little at a time. We sweep the offset from 0 up to
    ``max_offset`` and report how much of that drift goes undetected.
    """
    rng = np.random.default_rng(random_state)
    baseline = np.asarray(baseline, dtype=float)
    center = baseline.mean(axis=0)
    n_features = center.shape[0]
    idx = rng.integers(0, n_features) if feature_index is None else feature_index

    offsets = np.linspace(0.0, max_offset, n_steps)
    X_attack = np.tile(center, (n_steps, 1))
    X_attack[:, idx] += offsets

    evaded = _count_evaded(detector, X_attack)
    scores = detector.score_samples(X_attack)

    # Offset at which the detector first flags the drift, if ever.
    preds = detector.predict(X_attack)
    flagged = np.where(preds == 1)[0]
    first_detected_offset = float(offsets[flagged[0]]) if flagged.size else None

    return EvasionResult(
        attack_name="slow_drift",
        n_samples=n_steps,
        n_evaded=evaded,
        evasion_rate=evaded / n_steps,
        description=(
            "Gradual single-feature drift. Measures how far a slow deviation "
            "can grow before the detector reacts."
        ),
        details={
            "feature_index": int(idx),
            "max_offset": max_offset,
            "first_detected_offset": first_detected_offset,
            "max_score": float(scores.max()),
        },
    )


def boundary_probe_attack(
    detector: AnomalyDetector,
    baseline: np.ndarray,
    n_samples: int = 200,
    low: float = 0.5,
    high: float = 1.5,
    random_state: int = 0,
) -> EvasionResult:
    """Craft anomalies scaled to sit *near* the detection threshold.

    We locate the threshold in score space, then search for perturbations that
    produce scores just below it. This finds the thinnest margin an attacker
    needs to stay hidden.
    """
    rng = np.random.default_rng(random_state)
    baseline = np.asarray(baseline, dtype=float)
    center = baseline.mean(axis=0)
    n_features = center.shape[0]

    # Random direction perturbations scaled into a band around the baseline
    # spread, aiming to produce borderline-anomalous observations.
    spread = baseline.std(axis=0) + 1e-9
    directions = rng.normal(size=(n_samples, n_features))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True) + 1e-12
    scales = rng.uniform(low, high, size=(n_samples, 1))
    X_attack = center + directions * scales * spread * 5.0

    evaded = _count_evaded(detector, X_attack)
    scores = detector.score_samples(X_attack)

    return EvasionResult(
        attack_name="boundary_probe",
        n_samples=n_samples,
        n_evaded=evaded,
        evasion_rate=evaded / n_samples,
        description=(
            "Randomised perturbations scaled around the detection boundary. "
            "Estimates how easily borderline anomalies evade detection."
        ),
        details={
            "score_mean": float(scores.mean()),
            "score_max": float(scores.max()),
        },
    )


def single_feature_evasion_attack(
    detector: AnomalyDetector,
    baseline: np.ndarray,
    anomaly_magnitude: float = 10.0,
    masking: float = 0.0,
    random_state: int = 0,
) -> EvasionResult:
    """Hide a large single-feature anomaly, optionally masking it.

    A `max`-aggregation detector reacts to the single worst feature. An
    attacker who knows this might spread the deviation across features or add
    compensating noise. ``masking`` adds correlated noise on other features to
    test whether aggregation can be fooled.
    """
    rng = np.random.default_rng(random_state)
    baseline = np.asarray(baseline, dtype=float)
    center = baseline.mean(axis=0)
    n_features = center.shape[0]

    n_samples = 100
    X_attack = np.tile(center, (n_samples, 1))
    for row in X_attack:
        idx = rng.integers(0, n_features)
        row[idx] += anomaly_magnitude
        if masking > 0.0:
            others = [j for j in range(n_features) if j != idx]
            row[others] += rng.normal(0.0, masking, size=len(others))

    evaded = _count_evaded(detector, X_attack)

    return EvasionResult(
        attack_name="single_feature_evasion",
        n_samples=n_samples,
        n_evaded=evaded,
        evasion_rate=evaded / n_samples,
        description=(
            "Large single-feature anomaly with optional masking noise on the "
            "remaining features. Tests the robustness of score aggregation."
        ),
        details={"anomaly_magnitude": anomaly_magnitude, "masking": masking},
    )


def distributed_deviation_attack(
    detector: AnomalyDetector,
    baseline: np.ndarray,
    total_budget: float = 15.0,
    n_features_used: int | None = None,
    n_samples: int = 100,
    random_state: int = 0,
) -> EvasionResult:
    """Spread one anomaly budget across many features to beat `max` aggregation.

    A detector that aggregates per-feature deviations with `max` only reacts to
    the single worst feature. An attacker who understands this can split a large
    total deviation (the ``total_budget``) across several features, keeping each
    individual feature's deviation small enough to stay under the threshold,
    while the overall observation is clearly abnormal.

    This is the key blind spot of per-feature `max` scoring. A multivariate
    method (e.g. Mahalanobis) should resist it better, so this attack is useful
    for justifying the choice of scoring strategy.
    """
    rng = np.random.default_rng(random_state)
    baseline = np.asarray(baseline, dtype=float)
    center = baseline.mean(axis=0)
    spread = baseline.std(axis=0) + 1e-9
    n_features = center.shape[0]

    k = n_features if n_features_used is None else min(n_features_used, n_features)

    X_attack = np.tile(center, (n_samples, 1))
    for row in X_attack:
        idx = rng.choice(n_features, size=k, replace=False)
        # Split the total budget evenly, then add small jitter so the rows are
        # not identical. Budget is expressed in units of each feature's spread.
        per_feature = (total_budget / k) * spread[idx]
        signs = rng.choice([-1.0, 1.0], size=k)
        jitter = rng.uniform(0.85, 1.15, size=k)
        row[idx] += signs * per_feature * jitter

    evaded = _count_evaded(detector, X_attack)
    scores = detector.score_samples(X_attack)

    return EvasionResult(
        attack_name="distributed_deviation",
        n_samples=n_samples,
        n_evaded=evaded,
        evasion_rate=evaded / n_samples,
        description=(
            "Splits a large total deviation across many features so each stays "
            "small individually. Targets the blind spot of `max` aggregation."
        ),
        details={
            "total_budget": total_budget,
            "n_features_used": k,
            "score_mean": float(scores.mean()),
            "score_max": float(scores.max()),
        },
    )


def run_adversarial_suite(
    detector: AnomalyDetector, baseline: np.ndarray
) -> list[EvasionResult]:
    """Run the full evasion suite against a fitted detector."""
    return [
        slow_drift_attack(detector, baseline),
        boundary_probe_attack(detector, baseline),
        single_feature_evasion_attack(detector, baseline, masking=0.0),
        single_feature_evasion_attack(detector, baseline, masking=3.0),
        distributed_deviation_attack(detector, baseline),
    ]
