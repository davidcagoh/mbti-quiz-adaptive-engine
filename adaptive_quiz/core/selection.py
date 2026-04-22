# Re-export from adaptive_engine for backward compatibility.
from adaptive_engine.core.selection import (  # noqa: F401
    InformationGainSelection,
    SelectionStrategy,
    VarianceSelection,
    expected_information_gain,
    generate_question_weights,
    select_next_question,
)
