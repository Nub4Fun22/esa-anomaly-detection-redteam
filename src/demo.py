"""Runnable demonstration of the WP5 classical anomaly detector.

Run from the repository root:

    python -m src.detectors.classical.demo

This script:
  1. generates a SYNTHETIC dataset (not real robot data);
  2. fits the statistical detector on normal development data;
  3. evaluates it on the mixed evaluation split and reports detection rate
     and false alarm rate (the metrics WP7 cares about);
  4. runs the adversarial / evasion suite to surface the detector's blind
     spots.

The numbers below describe behaviour on synthetic signals only.
"""

from __future__ import annotations

import numpy as np

from adversarial import run_adversarial_suite
from statistical_detector import StatisticalAnomalyDetector
from synthetic_data import generate_synthetic_dataset


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute basic detection metrics (convention 0=normal, 1=anomaly)."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    n_anom = tp + fn
    n_norm = tn + fp

    detection_rate = tp / n_anom if n_anom else float("nan")  # recall
    false_alarm_rate = fp / n_norm if n_norm else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    f1 = (
        2 * precision * detection_rate / (precision + detection_rate)
        if precision and detection_rate and not np.isnan(precision)
        else float("nan")
    )
    return {
        "detection_rate": detection_rate,
        "false_alarm_rate": false_alarm_rate,
        "precision": precision,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def main() -> None:
    data = generate_synthetic_dataset()
    print("=" * 70)
    print("WP5 Classical Anomaly Detector - demo on SYNTHETIC data")
    print("=" * 70)
    print(f"Synthetic data: {data.is_synthetic}")
    print(f"Dev (normal only): {data.X_dev.shape}")
    print(f"Eval (mixed):      {data.X_eval.shape}, anomalies={int(data.y_eval.sum())}")
    print()

    for scoring in ("robust_zscore", "zscore", "mahalanobis"):
        detector = StatisticalAnomalyDetector(
            scoring=scoring,
            threshold_method="percentile",
            percentile=99.0,
        )
        detector.fit(data.X_dev)
        preds = detector.predict(data.X_eval)
        metrics = evaluate(data.y_eval, preds)

        print(f"[{scoring}] threshold={detector.threshold_:.3f}")
        print(
            f"    detection_rate={metrics['detection_rate']:.3f}  "
            f"false_alarm_rate={metrics['false_alarm_rate']:.3f}  "
            f"precision={metrics['precision']:.3f}  f1={metrics['f1']:.3f}"
        )

    print()
    print("-" * 70)
    print("Adversarial / evasion suite (red-team view)")
    print("Comparing a per-feature 'max' detector vs a multivariate one.")
    print("-" * 70)
    for scoring in ("robust_zscore", "mahalanobis"):
        detector = StatisticalAnomalyDetector(scoring=scoring, percentile=99.0)
        detector.fit(data.X_dev)
        print(f"\n### scoring = {scoring}")
        for result in run_adversarial_suite(detector, data.X_dev):
            print(f"[{result.attack_name}] evasion_rate={result.evasion_rate:.3f} "
                  f"({result.n_evaded}/{result.n_samples} evaded)")
            print(f"    {result.description}")
            if result.details:
                print(f"    details: {result.details}")


if __name__ == "__main__":
    main()
