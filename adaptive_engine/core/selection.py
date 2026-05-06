"""
Domain-agnostic question selection strategies (variance heuristic, information gain).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Protocol, Set, Union

import numpy as np


class SelectionStrategy(Protocol):
    """
    Strategy interface for choosing the next question.
    """

    def select(
        self,
        mu: np.ndarray,
        Sigma: np.ndarray,
        question_weights: Dict[str, np.ndarray],
        asked_ids: Set[str],
    ) -> str:
        ...


def expected_information_gain(
    Sigma_prior: np.ndarray,
    w: np.ndarray,
    sigma2_t: float,
) -> float:
    """
    Expected information gain = 0.5 * log(1 + w^T Sigma w / sigma2_t).

    Note: In this Gaussian linear model, InformationGainSelection and
    VarianceSelection produce identical rankings (same argmax). The meaningful
    comparison is adaptive vs. random, not between these two strategies.
    """
    Sigma_inv_prior = np.linalg.inv(Sigma_prior)
    Sigma_post = np.linalg.inv(Sigma_inv_prior + np.outer(w, w) / sigma2_t)
    det_prior = np.linalg.det(Sigma_prior)
    det_post = np.linalg.det(Sigma_post)
    if det_prior <= 0 or det_post <= 0:
        return 0.0
    return 0.5 * np.log(det_prior / det_post)


def select_next_question(
    mu: np.ndarray,
    Sigma: np.ndarray,
    question_pool: List[np.ndarray],
    asked_indices: Optional[Union[Set[int], List[int]]] = None,
    mode: str = "variance",
    sigma2_t: float = 0.2,
) -> tuple[int, np.ndarray]:
    """Legacy function-based API. New code should prefer VarianceSelection."""
    if asked_indices is None:
        asked_indices = set()
    asked_set = set(asked_indices)

    scores: List[float] = []
    for idx, w in enumerate(question_pool):
        if idx in asked_set:
            scores.append(-np.inf)
            continue
        if mode == "info_gain":
            score = expected_information_gain(Sigma, w, sigma2_t)
        else:
            score = float(w.T @ Sigma @ w)
        scores.append(score)

    next_index = int(np.argmax(scores))
    return next_index, question_pool[next_index]


class VarianceSelection:
    """Selection strategy that maximizes projected variance w^T Σ w."""

    def select(
        self,
        mu: np.ndarray,
        Sigma: np.ndarray,
        question_weights: Dict[str, np.ndarray],
        asked_ids: Set[str],
    ) -> str:
        scores: Dict[str, float] = {}
        for qid, w in question_weights.items():
            if qid in asked_ids:
                continue
            scores[qid] = float(w.T @ Sigma @ w)
        if not scores:
            raise ValueError("No remaining questions to select from.")
        return max(scores, key=scores.get)


class InformationGainSelection:
    """Selection strategy that maximizes expected information gain."""

    def __init__(self, sigma2_t: float = 0.2) -> None:
        self.sigma2_t = sigma2_t

    def select(
        self,
        mu: np.ndarray,
        Sigma: np.ndarray,
        question_weights: Dict[str, np.ndarray],
        asked_ids: Set[str],
    ) -> str:
        scores: Dict[str, float] = {}
        for qid, w in question_weights.items():
            if qid in asked_ids:
                continue
            scores[qid] = expected_information_gain(Sigma, w, self.sigma2_t)
        if not scores:
            raise ValueError("No remaining questions to select from.")
        return max(scores, key=scores.get)


def generate_question_weights(
    d: int,
    T: int,
    random_state: Optional[np.random.RandomState] = None,
) -> List[np.ndarray]:
    """Generate synthetic weight vectors probing 1 or 2 latent traits. For experiments."""
    if random_state is None:
        random_state = np.random
    w_list: List[np.ndarray] = []
    for _ in range(T):
        w = np.zeros(d)
        num_axes = random_state.randint(1, 3)
        axes = random_state.choice(d, num_axes, replace=False)
        w[axes] = 1.0
        w_list.append(w)
    return w_list
