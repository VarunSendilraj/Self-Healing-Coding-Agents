import pytest
from self_healing_agents.error_types import (
    ErrorType,
    AgentType,
    HIGH_CONFIDENCE_THRESHOLD,
    MEDIUM_CONFIDENCE_THRESHOLD,
    classify_error_type_by_confidence,
    suggest_agent_for_error,
    get_error_type_from_string,
    get_agent_type_from_string
)

# Test Enum Definitions and Values

def test_error_type_enum():
    """Tests the definition and members of the ErrorType enum."""
    assert ErrorType.PLANNING_ERROR.value == "PLANNING_ERROR"
    assert ErrorType.EXECUTION_ERROR.value == "EXECUTION_ERROR"
    assert ErrorType.AMBIGUOUS_ERROR.value == "AMBIGUOUS_ERROR"
    assert len(ErrorType) == 3

def test_agent_type_enum():
    """Tests the definition and members of the AgentType enum."""
    assert AgentType.PLANNER.value == "PLANNER"
    assert AgentType.EXECUTOR.value == "EXECUTOR"
    assert len(AgentType) == 2

# Test Confidence Threshold Constants

def test_confidence_thresholds():
    """Tests the values of confidence threshold constants."""
    assert isinstance(HIGH_CONFIDENCE_THRESHOLD, float)
    assert isinstance(MEDIUM_CONFIDENCE_THRESHOLD, float)
    assert 0.0 <= HIGH_CONFIDENCE_THRESHOLD <= 1.0
    assert 0.0 <= MEDIUM_CONFIDENCE_THRESHOLD <= 1.0
    assert HIGH_CONFIDENCE_THRESHOLD > MEDIUM_CONFIDENCE_THRESHOLD

# Test classify_error_type_by_confidence function

@pytest.mark.parametrize(
    "confidence, predicted_type, expected_type",
    [
        (0.9, ErrorType.PLANNING_ERROR, ErrorType.PLANNING_ERROR),
        (HIGH_CONFIDENCE_THRESHOLD, ErrorType.EXECUTION_ERROR, ErrorType.EXECUTION_ERROR),
        (MEDIUM_CONFIDENCE_THRESHOLD, ErrorType.PLANNING_ERROR, ErrorType.PLANNING_ERROR),
        (MEDIUM_CONFIDENCE_THRESHOLD - 0.01, ErrorType.EXECUTION_ERROR, ErrorType.AMBIGUOUS_ERROR),
        (0.5, ErrorType.PLANNING_ERROR, ErrorType.AMBIGUOUS_ERROR), # Below MEDIUM_CONFIDENCE_THRESHOLD
        (0.0, ErrorType.EXECUTION_ERROR, ErrorType.AMBIGUOUS_ERROR),
    ]
)
def test_classify_error_type_by_confidence(confidence, predicted_type, expected_type):
    """Tests the logic of classify_error_type_by_confidence."""
    assert classify_error_type_by_confidence(confidence, predicted_type) == expected_type

# Test suggest_agent_for_error function

@pytest.mark.parametrize(
    "error_type, expected_agent_type",
    [
        (ErrorType.PLANNING_ERROR, AgentType.PLANNER),
        (ErrorType.EXECUTION_ERROR, AgentType.EXECUTOR),
        (ErrorType.AMBIGUOUS_ERROR, AgentType.EXECUTOR), # Default case for ambiguous
    ]
)
def test_suggest_agent_for_error(error_type, expected_agent_type):
    """Tests the logic of suggest_agent_for_error."""
    assert suggest_agent_for_error(error_type) == expected_agent_type

# Test get_error_type_from_string function

@pytest.mark.parametrize(
    "input_str, expected_error_type",
    [
        ("PLANNING_ERROR", ErrorType.PLANNING_ERROR),
        ("planning_error", ErrorType.PLANNING_ERROR),
        ("EXECUTION_ERROR", ErrorType.EXECUTION_ERROR),
        ("execution_error", ErrorType.EXECUTION_ERROR),
        ("AMBIGUOUS_ERROR", ErrorType.AMBIGUOUS_ERROR),
        ("ambiguous_error", ErrorType.AMBIGUOUS_ERROR),
    ]
)
def test_get_error_type_from_string_valid(input_str, expected_error_type):
    """Tests get_error_type_from_string with valid string inputs."""
    assert get_error_type_from_string(input_str) == expected_error_type

@pytest.mark.parametrize(
    "invalid_str",
    [
        "INVALID_ERROR",
        "plan_error",
        "",
        "PLANNINGERROR"
    ]
)
def test_get_error_type_from_string_invalid(invalid_str):
    """Tests get_error_type_from_string with invalid string inputs."""
    with pytest.raises(ValueError, match=f"Invalid error type string: '{invalid_str}'"):
        get_error_type_from_string(invalid_str)

# Test get_agent_type_from_string function

@pytest.mark.parametrize(
    "input_str, expected_agent_type",
    [
        ("PLANNER", AgentType.PLANNER),
        ("planner", AgentType.PLANNER),
        ("EXECUTOR", AgentType.EXECUTOR),
        ("executor", AgentType.EXECUTOR),
    ]
)
def test_get_agent_type_from_string_valid(input_str, expected_agent_type):
    """Tests get_agent_type_from_string with valid string inputs."""
    assert get_agent_type_from_string(input_str) == expected_agent_type

@pytest.mark.parametrize(
    "invalid_str",
    [
        "INVALID_AGENT",
        "planer", # Typo
        "",
        "EXECUTORAGENT"
    ]
)
def test_get_agent_type_from_string_invalid(invalid_str):
    """Tests get_agent_type_from_string with invalid string inputs."""
    with pytest.raises(ValueError, match=f"Invalid agent type string: '{invalid_str}'"):
        get_agent_type_from_string(invalid_str) 