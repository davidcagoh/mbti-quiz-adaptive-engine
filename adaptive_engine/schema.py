"""
Schema contract and DomainAdapter protocol for adaptive-engine.

A domain is anything that can be expressed as:
  - A list of questions, each with a weight vector in R^d
  - A scoring function that maps (mu, Sigma) -> a domain-specific result

Schema format (dict passed to AdaptiveEngine):
    {
        "dimensions": ["dim0", "dim1", ...],   # optional: names for latent dims
        "questions": [
            {
                "id": str,           # unique across the question bank
                "weights": [float],  # length == len(dimensions)
                ...                  # domain-specific fields passed through unchanged
            },
            ...
        ]
    }

Response convention:
    Responses must be mapped to floats before passing to submit_answer().
    Typical conventions:
        Likert 1-5  →  [-1, -0.5, 0, 0.5, 1]
        Binary correct/incorrect  →  [1, -1]
        Binary yes/no  →  [1, -1]

Weight vector convention:
    A positive weight on dimension i means a positive response increases mu[i].
    Domain adapters are responsible for sign conventions (e.g. MBTI E/I polarity,
    mastery know/don't-know polarity).
"""

from __future__ import annotations

from typing import Any, Dict, List, Protocol

import numpy as np


class DomainAdapter(Protocol):
    """
    Protocol implemented by domain-specific layers (MBTI, mastery, political compass, …).

    The engine is completely unaware of the domain; the adapter translates
    posterior state into a human-readable result.
    """

    def build_schema(self) -> Dict[str, Any]:
        """Return a schema dict compatible with AdaptiveEngine."""
        ...

    def map_response(self, raw_response: Any) -> float:
        """Map a domain-specific raw response to a float in the engine's observation space."""
        ...

    def score(self, mu: np.ndarray, Sigma: np.ndarray) -> Dict[str, Any]:
        """
        Translate posterior (mu, Sigma) into a domain-specific result dict.
        The result structure is domain-defined; callers should not assume shape.
        """
        ...


def validate_schema(schema: Dict[str, Any]) -> List[str]:
    """
    Validate a schema dict. Returns a list of error strings (empty = valid).
    """
    errors: List[str] = []
    if "questions" not in schema:
        errors.append("schema must have a 'questions' key")
        return errors

    questions = schema["questions"]
    if not isinstance(questions, list) or len(questions) == 0:
        errors.append("'questions' must be a non-empty list")
        return errors

    ids_seen: set = set()
    d = len(questions[0].get("weights", []))

    for i, q in enumerate(questions):
        if "id" not in q:
            errors.append(f"question[{i}] missing 'id'")
        elif q["id"] in ids_seen:
            errors.append(f"duplicate question id: {q['id']}")
        else:
            ids_seen.add(q["id"])

        w = q.get("weights")
        if w is None:
            errors.append(f"question[{i}] missing 'weights'")
        elif len(w) != d:
            errors.append(
                f"question[{i}] weight length {len(w)} != expected {d}"
            )

    return errors
