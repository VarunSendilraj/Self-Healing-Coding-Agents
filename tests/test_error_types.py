import pytest
from enum import Enum
from self_healing_agents.error_types import (
    ErrorType, 
    AgentType, 
    get_responsible_agent_for_error_type,
    is_high_confidence,
    is_medium_confidence,
    is_low_confidence,
    classify_confidence_level,
    HIGH_CONFIDENCE_THRESHOLD,
    MEDIUM_CONFIDENCE_THRESHOLD,
    LOW_CONFIDENCE_THRESHOLD
)

# Test Enum Definitions
def test_error_type_enum_values():
    assert ErrorType.PLANNING_ERROR.value == "PLANNING_ERROR"
    assert ErrorType.EXECUTION_ERROR.value == "EXECUTION_ERROR"
    assert ErrorType.AMBIGUOUS_ERROR.value == "AMBIGUOUS_ERROR"
    assert ErrorType.UNKNOWN_ERROR.value == "UNKNOWN_ERROR"
    # Check that all expected members are present
    expected_members = ["PLANNING_ERROR", "EXECUTION_ERROR", "AMBIGUOUS_ERROR", "UNKNOWN_ERROR"]
    assert all(member.name in expected_members for member in ErrorType)
    assert len(ErrorType) == len(expected_members)

def test_agent_type_enum_values():
    assert AgentType.PLANNER.value == "PLANNER"
    assert AgentType.EXECUTOR.value == "EXECUTOR"
    assert AgentType.CLASSIFIER.value == "CLASSIFIER"
    assert AgentType.CRITIC.value == "CRITIC"
    assert AgentType.UNKNOWN_AGENT.value == "UNKNOWN_AGENT"
    # Check that all expected members are present
    expected_members = ["PLANNER", "EXECUTOR", "CLASSIFIER", "CRITIC", "UNKNOWN_AGENT"]
    assert all(member.name in expected_members for member in AgentType)
    assert len(AgentType) == len(expected_members)

# Test Confidence Threshold Constants (ensure they are floats and ordered logically)
def test_confidence_thresholds():
    assert isinstance(HIGH_CONFIDENCE_THRESHOLD, float)
    assert isinstance(MEDIUM_CONFIDENCE_THRESHOLD, float)
    assert isinstance(LOW_CONFIDENCE_THRESHOLD, float)
    assert HIGH_CONFIDENCE_THRESHOLD > MEDIUM_CONFIDENCE_THRESHOLD
    assert MEDIUM_CONFIDENCE_THRESHOLD > LOW_CONFIDENCE_THRESHOLD
    assert 0.0 < LOW_CONFIDENCE_THRESHOLD < 1.0
    assert 0.0 < MEDIUM_CONFIDENCE_THRESHOLD < 1.0
    assert 0.0 < HIGH_CONFIDENCE_THRESHOLD < 1.0

# Test get_responsible_agent_for_error_type function
@pytest.mark.parametrize(
    "error_type, expected_agent_type",
    [
        (ErrorType.PLANNING_ERROR, AgentType.PLANNER),
        (ErrorType.EXECUTION_ERROR, AgentType.EXECUTOR),
        (ErrorType.AMBIGUOUS_ERROR, AgentType.UNKNOWN_AGENT),
        (ErrorType.UNKNOWN_ERROR, AgentType.UNKNOWN_AGENT),
    ]
)
def test_get_responsible_agent_for_error_type(error_type, expected_agent_type):
    assert get_responsible_agent_for_error_type(error_type) == expected_agent_type

def test_get_responsible_agent_for_error_type_invalid():
    class MockInvalidErrorType(Enum):
        SOME_OTHER_ERROR = "SOME_OTHER_ERROR"
    with pytest.raises(ValueError):
        get_responsible_agent_for_error_type(MockInvalidErrorType.SOME_OTHER_ERROR)

# Test confidence checking helper functions
@pytest.mark.parametrize(
    "score, expected_high, expected_medium, expected_low",
    [
        (0.9, True, False, False), # High
        (HIGH_CONFIDENCE_THRESHOLD, True, False, False), # Exactly High
        (0.7, False, True, False), # Medium (between medium and high)
        (MEDIUM_CONFIDENCE_THRESHOLD, False, True, False), # Exactly Medium
        (0.4, False, False, True), # Low (between low and medium)
        (LOW_CONFIDENCE_THRESHOLD, False, False, True), # Exactly Low, but classify_confidence_level makes it Medium
        (0.2, False, False, True), # Very Low
        (0.0, False, False, True), # Zero
        (1.0, True, False, False) # Max
    ]
)
def test_confidence_boolean_helpers(score, expected_high, expected_medium, expected_low):
    assert is_high_confidence(score) == expected_high
    # Note: is_medium_confidence is exclusive of high
    assert is_medium_confidence(score) == expected_medium 
    # Note: is_low_confidence is based on being less than MEDIUM_CONFIDENCE_THRESHOLD
    assert is_low_confidence(score) == expected_low

# Test classify_confidence_level function
@pytest.mark.parametrize(
    "score, expected_level_str",
    [
        (0.9, "HIGH"),
        (HIGH_CONFIDENCE_THRESHOLD, "HIGH"),
        (0.75, "MEDIUM"), # Clearly medium
        (MEDIUM_CONFIDENCE_THRESHOLD, "MEDIUM"),
        (0.5, "MEDIUM"), # Medium, above LOW_CONFIDENCE_THRESHOLD
        (LOW_CONFIDENCE_THRESHOLD, "MEDIUM"), # At LOW_CONFIDENCE_THRESHOLD, should be MEDIUM by current logic
        (0.25, "LOW"), # Clearly low
        (0.1, "LOW"),
        (0.0, "LOW"),
        (1.0, "HIGH")
    ]
)
def test_classify_confidence_level(score, expected_level_str):
    assert classify_confidence_level(score) == expected_level_str 