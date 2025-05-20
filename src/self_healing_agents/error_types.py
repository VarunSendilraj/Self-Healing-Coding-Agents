from enum import Enum

class ErrorType(Enum):
    """
    Defines the taxonomy of error types encountered during agent execution.

    Attributes:
        PLANNING_ERROR: Indicates flaws in the overall approach, strategy,
                        or logical structure of the solution.
        EXECUTION_ERROR: Signifies issues in the concrete implementation or
                         code, such as syntax errors or runtime exceptions.
        AMBIGUOUS_ERROR: Represents cases where the error classification confidence
                         is below a defined threshold, making it unclear whether
                         the error is due to planning or execution.
    """
    PLANNING_ERROR = "PLANNING_ERROR"
    EXECUTION_ERROR = "EXECUTION_ERROR"
    AMBIGUOUS_ERROR = "AMBIGUOUS_ERROR"

class AgentType(Enum):
    """
    Defines the types of agents within the system, primarily for routing
    correction tasks.

    Attributes:
        PLANNER: The agent responsible for high-level strategy,
                 problem decomposition, and algorithm selection.
        EXECUTOR: The agent responsible for code implementation,
                  syntax correctness, and adhering to the planner's directives.
    """
    PLANNER = "PLANNER"
    EXECUTOR = "EXECUTOR"

# Confidence Thresholds for Classification
# These constants define the minimum confidence score required for a
# classification to be considered reliable.
HIGH_CONFIDENCE_THRESHOLD: float = 0.85
"""Minimum confidence for a classification to be considered high."""

MEDIUM_CONFIDENCE_THRESHOLD: float = 0.60
"""Minimum confidence for a classification to be considered medium;
   errors below this but above LOW_CONFIDENCE_THRESHOLD might be
   treated as AMBIGUOUS_ERROR or default to a specific handling strategy.
"""

# Helper functions (initial placeholders, to be expanded based on classification logic)

def classify_error_type_by_confidence(
    confidence_score: float,
    predicted_error_type: ErrorType
) -> ErrorType:
    """
    Determines the final error type based on a confidence score.

    If the confidence score is below the MEDIUM_CONFIDENCE_THRESHOLD,
    the error is classified as AMBIGUOUS_ERROR. Otherwise, the
    predicted_error_type is returned.

    Args:
        confidence_score: The confidence score (0.0 to 1.0) of the
                          initial error classification.
        predicted_error_type: The initially predicted error type.

    Returns:
        The final ErrorType based on the confidence score.
    """
    if confidence_score < MEDIUM_CONFIDENCE_THRESHOLD:
        return ErrorType.AMBIGUOUS_ERROR
    return predicted_error_type

def suggest_agent_for_error(error_type: ErrorType) -> AgentType:
    """
    Suggests which agent should handle the correction based on the error type.

    Args:
        error_type: The classified error type.

    Returns:
        The AgentType best suited to handle the error.
        Defaults to EXECUTOR for AMBIGUOUS_ERRORs as a conservative approach.
    """
    if error_type == ErrorType.PLANNING_ERROR:
        return AgentType.PLANNER
    elif error_type == ErrorType.EXECUTION_ERROR:
        return AgentType.EXECUTOR
    elif error_type == ErrorType.AMBIGUOUS_ERROR:
        # Default to Executor for ambiguous errors as a conservative approach
        return AgentType.EXECUTOR
    else:
        # Should not happen with Enum types, but good for robustness
        raise ValueError(f"Unknown error type: {error_type}")

def get_error_type_from_string(error_type_str: str) -> ErrorType:
    """
    Converts a string to an ErrorType enum member.

    Args:
        error_type_str: The string representation of the error type.

    Returns:
        The corresponding ErrorType enum member.

    Raises:
        ValueError: If the string does not match any ErrorType member.
    """
    try:
        return ErrorType[error_type_str.upper()]
    except KeyError:
        raise ValueError(f"Invalid error type string: '{error_type_str}'")

def get_agent_type_from_string(agent_type_str: str) -> AgentType:
    """
    Converts a string to an AgentType enum member.

    Args:
        agent_type_str: The string representation of the agent type.

    Returns:
        The corresponding AgentType enum member.

    Raises:
        ValueError: If the string does not match any AgentType member.
    """
    try:
        return AgentType[agent_type_str.upper()]
    except KeyError:
        raise ValueError(f"Invalid agent type string: '{agent_type_str}'") 