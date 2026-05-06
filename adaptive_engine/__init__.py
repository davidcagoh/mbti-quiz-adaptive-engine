"""
adaptive-engine: domain-agnostic Bayesian adaptive assessment engine.

Quick start
-----------
    from adaptive_engine import AdaptiveEngine, SignConfidenceStopping

    engine = AdaptiveEngine(
        schema=my_domain.build_schema(),
        stopping_rule=SignConfidenceStopping(confidence_threshold=0.90, max_questions=20),
    )
    while not engine.is_complete():
        qid, w = engine.get_next_question()
        response = ask_user(qid)          # domain-specific
        engine.submit_answer(qid, response, response_time=1.0)

    state = engine.get_state()
    result = my_domain.score(state.mu, state.Sigma)
"""

from adaptive_engine.core.bayesian import bayesian_update
from adaptive_engine.core.engine import (
    AdaptiveEngine,
    DefaultLikertNoiseModel,
    EngineState,
    NoiseModel,
)
from adaptive_engine.core.selection import (
    InformationGainSelection,
    SelectionStrategy,
    VarianceSelection,
    generate_question_weights,
    select_next_question,
)
from adaptive_engine.core.stopping import (
    SignConfidenceStopping,
    StoppingRule,
    VarianceThresholdStopping,
)
from adaptive_engine.schema import DomainAdapter, validate_schema

__all__ = [
    "AdaptiveEngine",
    "EngineState",
    "NoiseModel",
    "DefaultLikertNoiseModel",
    "SelectionStrategy",
    "VarianceSelection",
    "InformationGainSelection",
    "select_next_question",
    "generate_question_weights",
    "StoppingRule",
    "SignConfidenceStopping",
    "VarianceThresholdStopping",
    "bayesian_update",
    "DomainAdapter",
    "validate_schema",
]
