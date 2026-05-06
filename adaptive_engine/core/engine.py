"""
Domain-agnostic adaptive engine. Operates in latent vector space only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

import numpy as np

from adaptive_engine.core.bayesian import bayesian_update
from adaptive_engine.core.selection import (
    InformationGainSelection,
    SelectionStrategy,
    VarianceSelection,
    select_next_question,
)
from adaptive_engine.core.stopping import StoppingRule


@dataclass
class EngineState(dict):
    """
    Snapshot of engine state.
    Inherits from dict for mapping-style access (state["mu"], ...).
    """

    mu: np.ndarray
    Sigma: np.ndarray
    num_questions: int
    asked_question_ids: List[str]

    def __post_init__(self) -> None:
        self["mu"] = self.mu
        self["Sigma"] = self.Sigma
        self["num_questions"] = self.num_questions
        self["asked_question_ids"] = self.asked_question_ids


class NoiseModel(Protocol):
    def compute_variance(self, response: float, response_time: Optional[float]) -> float:
        ...


class DefaultLikertNoiseModel:
    """sigma2_t = 0.1 + 0.1 * response_time + 0.1 * (1 - |response|)"""

    def compute_variance(self, response: float, response_time: Optional[float]) -> float:
        rt = 0.0 if response_time is None else response_time
        return 0.1 + 0.1 * rt + 0.1 * (1.0 - abs(response))


class _CallableNoiseAdapter(DefaultLikertNoiseModel):
    def __init__(self, fn: Callable[[float, float], float]) -> None:
        self._fn = fn

    def compute_variance(self, response: float, response_time: Optional[float]) -> float:
        rt = 0.0 if response_time is None else response_time
        return self._fn(response, rt)


class AdaptiveEngine:
    """
    Maintains posterior (mu, Sigma), selects questions, updates posterior, checks stopping.

    Schema format expected by the constructor:
        {
            "questions": [
                {"id": str, "weights": list[float], ...},
                ...
            ]
        }
    The "weights" list must have length equal to the number of latent dimensions.
    """

    def __init__(
        self,
        schema: Dict[str, Any],
        selection_strategy: Optional[SelectionStrategy | Callable[..., Tuple[int, np.ndarray]]] = None,
        stopping_rule: Optional[StoppingRule] = None,
        prior_mean: Optional[np.ndarray] = None,
        prior_cov: Optional[np.ndarray] = None,
        selection_mode: str = "variance",
        noise_model: Optional[NoiseModel] = None,
        noise_variance_fn: Optional[Callable[[float, float], float]] = None,
        on_update: Optional[Callable[[EngineState], None]] = None,
    ) -> None:
        questions = schema["questions"]
        self._questions = questions
        self._weight_vectors = [np.asarray(q["weights"], dtype=float) for q in questions]
        self._id_to_index = {q["id"]: i for i, q in enumerate(questions)}
        self._id_to_weight: Dict[str, np.ndarray] = {
            q["id"]: self._weight_vectors[i] for i, q in enumerate(questions)
        }
        self._d = len(self._weight_vectors[0])
        self._selection_strategy = selection_strategy
        self._stopping_rule = stopping_rule
        self._selection_mode = selection_mode

        if noise_model is not None:
            self._noise_model: NoiseModel = noise_model
        elif noise_variance_fn is not None:
            self._noise_model = _CallableNoiseAdapter(noise_variance_fn)
        else:
            self._noise_model = DefaultLikertNoiseModel()

        self._on_update = on_update
        self._mu = prior_mean if prior_mean is not None else np.zeros(self._d)
        self._Sigma = prior_cov if prior_cov is not None else np.eye(self._d)
        self._asked_indices: set = set()
        self._max_questions = len(questions)

    def get_next_question(self) -> Optional[Tuple[str, np.ndarray]]:
        """Returns (question_id, weight_vector) or None if complete."""
        if self.is_complete():
            return None
        if self._selection_strategy is not None:
            if hasattr(self._selection_strategy, "select"):
                asked_ids = {self._questions[i]["id"] for i in self._asked_indices}
                question_id = self._selection_strategy.select(
                    self._mu, self._Sigma, self._id_to_weight, asked_ids,
                )
                idx = self._id_to_index[question_id]
                w = self._weight_vectors[idx]
            else:
                idx, w = self._selection_strategy(
                    self._mu, self._Sigma, self._weight_vectors, self._asked_indices,
                )
                question_id = self._questions[idx]["id"]
        else:
            if self._selection_mode == "info_gain":
                strategy = InformationGainSelection()
                asked_ids = {self._questions[i]["id"] for i in self._asked_indices}
                question_id = strategy.select(self._mu, self._Sigma, self._id_to_weight, asked_ids)
                idx = self._id_to_index[question_id]
                w = self._weight_vectors[idx]
            else:
                idx, w = select_next_question(
                    self._mu, self._Sigma, self._weight_vectors, self._asked_indices, mode="variance",
                )
                question_id = self._questions[idx]["id"]
        return question_id, w.copy()

    def submit_answer(self, question_id: str, response: float, response_time: float) -> None:
        if question_id not in self._id_to_index:
            raise ValueError(f"Unknown question_id: {question_id}")
        idx = self._id_to_index[question_id]
        w = self._weight_vectors[idx]
        sigma2_t = self._noise_model.compute_variance(response, response_time)
        self._mu, self._Sigma = bayesian_update(self._mu, self._Sigma, w, response, sigma2_t)
        self._asked_indices.add(idx)
        if self._on_update is not None:
            self._on_update(self.get_state())

    def is_complete(self) -> bool:
        if self._stopping_rule is None:
            return len(self._asked_indices) >= self._max_questions
        return self._stopping_rule.should_stop(
            self._mu, self._Sigma,
            num_asked=len(self._asked_indices),
            max_questions=self._max_questions,
        )

    def get_state(self) -> EngineState:
        asked_ids = [self._questions[i]["id"] for i in sorted(self._asked_indices)]
        return EngineState(
            mu=self._mu.copy(),
            Sigma=self._Sigma.copy(),
            num_questions=len(self._asked_indices),
            asked_question_ids=asked_ids,
        )
