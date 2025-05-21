from enum import Enum

class ErrorType(Enum):
    """
    Defines the type of error encountered during agent execution.
    """
    PLANNING_ERROR = "PLANNING_ERROR"
    EXECUTION_ERROR = "EXECUTION_ERROR"
    AMBIGUOUS_ERROR = "AMBIGUOUS_ERROR"
    UNKNOWN_ERROR = "UNKNOWN_ERROR" # Default or fallback

class AgentType(Enum):
    """
    Defines the type of agent in the system.
    """
    PLANNER = "PLANNER"
    EXECUTOR = "EXECUTOR"
    CLASSIFIER = "CLASSIFIER" # Added for completeness if needed later
    CRITIC = "CRITIC"         # Added for completeness if needed later
    UNKNOWN_AGENT = "UNKNOWN_AGENT" # Default or fallback

# --- Confidence Thresholds ---

# High confidence: Strong indication of the error type.
HIGH_CONFIDENCE_THRESHOLD: float = 0.85
"""Threshold for high confidence in error classification.
If confidence is above this, the classification is considered strong."""

# Medium confidence: Reasonable indication, but warrants caution.
MEDIUM_CONFIDENCE_THRESHOLD: float = 0.60
"""Threshold for medium confidence.
May suggest a primary agent but could involve others."""

# Low confidence: Ambiguous or unclear error source.
# Below this, the error might be classified as AMBIGUOUS_ERROR or UNKNOWN_ERROR
# or default to a specific handling strategy.
LOW_CONFIDENCE_THRESHOLD: float = 0.30
"""Threshold for low confidence.
Suggests ambiguity; fallback strategies may be needed."""


# --- Helper Functions ---

def get_responsible_agent_for_error_type(error_type: ErrorType) -> AgentType:
    """
    Maps an error type to the primarily responsible agent type.

    This provides a default mapping. The actual responsible agent might be
    determined with more nuance by the ErrorClassificationAgent based on context.

    Args:
        error_type: The classified type of the error.

    Returns:
        The agent type primarily responsible for this error category.
    """
    if error_type == ErrorType.PLANNING_ERROR:
        return AgentType.PLANNER
    elif error_type == ErrorType.EXECUTION_ERROR:
        return AgentType.EXECUTOR
    # For AMBIGUOUS or UNKNOWN errors, the responsibility is less clear.
    # A default (e.g., Executor) or a more complex decision logic might be needed.
    # For now, let's return UNKNOWN_AGENT, to be handled by the controller.
    elif error_type in [ErrorType.AMBIGUOUS_ERROR, ErrorType.UNKNOWN_ERROR]:
        return AgentType.UNKNOWN_AGENT
    else:
        # Should not happen if all ErrorType members are handled
        raise ValueError(f"Unhandled error type: {error_type}")

def is_high_confidence(confidence_score: float) -> bool:
    """Checks if a confidence score is considered high."""
    return confidence_score >= HIGH_CONFIDENCE_THRESHOLD

def is_medium_confidence(confidence_score: float) -> bool:
    """Checks if a confidence score is considered medium (but not high)."""
    return MEDIUM_CONFIDENCE_THRESHOLD <= confidence_score < HIGH_CONFIDENCE_THRESHOLD

def is_low_confidence(confidence_score: float) -> bool:
    """Checks if a confidence score is considered low."""
    return confidence_score < MEDIUM_CONFIDENCE_THRESHOLD

def classify_confidence_level(confidence_score: float) -> str:
    """Classifies a confidence score into 'HIGH', 'MEDIUM', or 'LOW'."""
    if is_high_confidence(confidence_score):
        return "HIGH"
    elif is_medium_confidence(confidence_score):
        return "MEDIUM"
    elif confidence_score >= LOW_CONFIDENCE_THRESHOLD : # Catch cases between LOW and MEDIUM explicitly
        return "MEDIUM" # Or define a separate category if needed
    else: # Handles scores below LOW_CONFIDENCE_THRESHOLD
        return "LOW" 