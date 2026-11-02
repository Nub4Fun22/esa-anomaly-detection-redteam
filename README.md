# ESA Anomaly Detection — Classical Detector & Adversarial (Red-Team) Testing

A classical statistical anomaly detector for autonomous space robotics, built as
my contribution to an **ESA Academy Experiments Programme — Orbital Robotics Lab**
project, together with a **red-team / evasion testing suite** that attacks the
detector to find and document its blind spots.

This repository is a personal portfolio extract of my work. It focuses on the
part I own: the classical detection baseline (WP5) and the security evaluation
of that detector against an adaptive adversary.

---

## Context

**Project:** Cybersecurity for Autonomous Space Robotics — AI-Based Anomaly
Detection. A team project that builds a system to detect anomalous behaviour in
autonomous robotic operations, comparing a **classical baseline** against an
**AI/ML detector** (Isolation Forest).

**My role (WP5 — Classical Anomaly Detection):** design, implement, and evaluate
the classical, non-learning detection baseline — and, because an anomaly
detector is a security control, probe it the way a penetration tester would.

> **Data note.** All results here are produced on a **synthetic** dataset. No
> real robot, simulator, or sensor data is used. The synthetic generator and the
> numbers it produces are a development/testing baseline, clearly labelled as
> such, pending the real dataset.

---

## What I built

### 1. A scalable classical detector (`src/statistical_detector.py`)

A multivariate statistical detector rather than a single hard-coded rule,
designed to be extensible:

- **Pluggable scoring strategies:** `robust_zscore` (median + MAD),
  `zscore`, and `mahalanobis` (uses the inverse covariance matrix, accounts for
  feature correlations). Adding a new strategy means registering one function.
- **Data-driven thresholding:** percentile of training scores, a sigma
  multiplier, an expected contamination rate, or a fixed value — never an
  arbitrary hand-picked number.
- Implements a shared `AnomalyDetector` interface (`fit` / `predict` /
  `score_samples`) with the convention `0 = normal`, `1 = anomalous`, higher
  score = more anomalous.

### 2. A reproducible synthetic dataset (`src/synthetic_data.py`)

Generates labelled normal and anomalous observations (value deviations and
sensor spikes), split into development (normal only, for fitting) and evaluation
(mixed) sets, with no data leakage.

### 3. A red-team / evasion testing suite (`src/adversarial.py`)

This is the security core. Each attack crafts inputs that are genuinely
anomalous but engineered to **evade** the detector, and measures the
**evasion rate**:

| Attack | Idea |
|---|---|
| `slow_drift` | Nudge one feature away gradually — mimics an attacker drifting a trajectory slowly. |
| `boundary_probe` | Probe perturbations scaled around the decision boundary. |
| `single_feature_evasion` | One large single-feature anomaly, optionally masked with noise. |
| `distributed_deviation` | Split one deviation budget across all features so each stays small individually. |

### 4. Evaluation report (`docs/evaluation_report.md`)

Full write-up: standard detection metrics plus the adversarial results, the key
finding, documented limitations, and recommendations.

---

## Key result — why the "best" detector was not the safe one

On benign data, the simple `robust_zscore` baseline looked best (highest
precision/F1). The adversarial evaluation reversed that conclusion:

| Attack | `robust_zscore` (max aggregation) | `mahalanobis` (multivariate) |
|---|---|---|
| single-feature (large) | 0% evasion | 0% evasion |
| **distributed deviation** | **100% evasion** | **0% evasion** |

**Why.** `robust_zscore` reacts to the single worst feature (max aggregation).
An attacker who understands this can spread the same total deviation across all
features so each one stays under the threshold — the observation is clearly
abnormal, yet **all 100 crafted anomalies passed as normal**. `mahalanobis`
evaluates the joint distribution, so a coordinated multi-feature deviation is
still caught.

**Takeaway.** Against an adaptive adversary, benign metrics alone are
misleading. Evaluating only non-adaptive anomalies hid a total-evasion
vulnerability. This is the core lesson I wanted to demonstrate: evaluate
security controls the way an attacker would.

---

## Run it

```bash
pip install -r requirements.txt
cd src
python demo.py
```

The demo prints the detection metrics and the adversarial results for both
scoring strategies. Results are deterministic (`random_state=42`).

> On Windows, if `python` opens the Microsoft Store, use `py demo.py`.

---

## Skills demonstrated

- **Security mindset applied to ML:** treating an anomaly detector as a defence
  and attacking it (evasion testing) rather than only measuring benign accuracy.
- **Adversarial thinking:** reasoning about what an attacker who understands the
  system would do, and turning that into concrete, measurable attacks.
- **Statistical / multivariate methods:** robust statistics (MAD), Mahalanobis
  distance, data-driven thresholding.
- **Clean, extensible engineering:** pluggable strategies, a shared interface,
  input validation, reproducibility, and documentation.

---

## Repository layout

```
src/
  interface.py              # shared AnomalyDetector interface
  statistical_detector.py   # the classical detector (pluggable scoring)
  synthetic_data.py         # synthetic dataset generator
  adversarial.py            # red-team / evasion testing suite
  demo.py                   # runnable end-to-end demo
docs/
  evaluation_report.md      # full evaluation + adversarial report
requirements.txt
```

---

*Developed as part of an ESA Academy Experiments Programme team project. This
repository contains my individual contribution (the classical detector and its
adversarial evaluation) extracted for portfolio purposes. Results are on
synthetic data.*
