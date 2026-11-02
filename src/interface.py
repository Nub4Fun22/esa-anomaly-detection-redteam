"""Common interface for anomaly detection methods."""

# Fix (WP5): removed a stray Markdown code fence ("```") left at the end of
# this file, which caused a SyntaxError and broke every import of the detector
# interface (and therefore both detectors). Logic/signatures unchanged.

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class AnomalyDetector(ABC):
    """Base interface for classical and AI/ML anomaly detectors."""

    @abstractmethod
    def fit(self, X: np.ndarray) -> "AnomalyDetector":
        """Prepare or train the detector using development data."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return one prediction per input sample.

        Convention:
            0: normal observation
            1: anomalous observation
        """
        raise NotImplementedError

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Return anomaly scores when supported by the implementation.

        Higher scores should indicate more anomalous observations.
        Detectors that do not support scoring must override this method
        or leave scoring unused by the evaluation workflow.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement anomaly scoring."
        )

    def get_metadata(self) -> dict[str, Any]:
        """Return basic information about the detector."""
        return {
            "name": self.__class__.__name__,
            "type": "anomaly_detector",
        }
