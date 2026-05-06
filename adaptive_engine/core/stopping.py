"""
Domain-agnostic stopping rules for the adaptive engine.
"""

import math
from typing import Optional, Protocol

import numpy as np


class StoppingRule(Protocol):
    def should_stop(
        self,
        mu: np.ndarray,
        Sigma: np.ndarray,
        num_asked: int,
        max_questions: int,
    ) -> bool:
        ...


class VarianceThresholdStopping:
    """Stop when max(diag(Sigma)) < variance_threshold, or max_questions reached."""

    def __init__(
        self,
        variance_threshold: float = 0.1,
        max_questions: Optional[int] = None,
    ) -> None:
        self.variance_threshold = variance_threshold
        self.max_questions = max_questions

    def should_stop(
        self,
        mu: np.ndarray,
        Sigma: np.ndarray,
        num_asked: int,
        max_questions: int,
    ) -> bool:
        if self.max_questions is not None and num_asked >= self.max_questions:
            return True
        if max_questions is not None and num_asked >= max_questions:
            return True
        return float(np.max(np.diag(Sigma))) < self.variance_threshold


class SignConfidenceStopping:
    """
    Stop when posterior sign confidence >= threshold for ALL dimensions.

    For each dimension i: confidence_i = max(Φ(μ_i/σ_i), 1 - Φ(μ_i/σ_i)).
    """

    def __init__(
        self,
        confidence_threshold: float = 0.95,
        max_questions: Optional[int] = None,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.max_questions = max_questions

    def _normal_cdf(self, z: float) -> float:
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    def should_stop(
        self,
        mu: np.ndarray,
        Sigma: np.ndarray,
        num_asked: int,
        max_questions: int,
    ) -> bool:
        if self.max_questions is not None and num_asked >= self.max_questions:
            return True
        if max_questions is not None and num_asked >= max_questions:
            return True

        for i in range(len(mu)):
            var_i = Sigma[i, i]
            if var_i <= 0.0:
                continue
            z_i = mu[i] / math.sqrt(var_i)
            p_pos = self._normal_cdf(z_i)
            if max(p_pos, 1.0 - p_pos) < self.confidence_threshold:
                return False
        return True
