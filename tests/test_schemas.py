import pytest
from pydantic import ValidationError

from self_healing_agents.schemas import (
    CriticReport,
    CRITIC_STATUS_SUCCESS,
    CRITIC_STATUS_FAILURE_RUNTIME,
    CRITIC_STATUS_FAILURE_SYNTAX
)
from self_healing_agents.error_types import ErrorType, AgentType


# --- Test CriticReport Basic Instantiation (Backward Compatibility) ---
def test_critic_report_backward_compatible():
    """Test that CriticReport can be instantiated without new Phase 2 fields."""
    report = CriticReport(
        status=CRITIC_STATUS_SUCCESS,
        score=1.0,
        summary="All good!"
    )
    assert report.status == CRITIC_STATUS_SUCCESS
    assert report.score == 1.0
    assert report.summary == "All good!"
    assert report.error_type is None
    assert report.confidence is None
    assert report.suggested_agent is None
    assert report.reasoning is None
    assert not report.is_failure()

def test_critic_report_failure_backward_compatible():
    """Test failure report without new Phase 2 fields."""
    report = CriticReport(
        status=CRITIC_STATUS_FAILURE_SYNTAX,
        score=0.0,
        error_details={"type": "SyntaxError", "message": "EOL"},
        summary="Syntax broke"
    )
    assert report.status == CRITIC_STATUS_FAILURE_SYNTAX
    assert report.score == 0.0
    assert report.error_details["type"] == "SyntaxError"
    assert report.error_type is None # Expected to be None for backward compatibility
    assert report.is_failure()

# --- Test CriticReport with New Phase 2 Fields ---
def test_critic_report_with_all_classification_fields():
    """Test instantiation with all new classification fields correctly populated."""
    report_data = {
        "status": CRITIC_STATUS_FAILURE_RUNTIME,
        "score": 0.1,
        "summary": "It crashed.",
        "error_details": {"type": "RuntimeError", "message": "Something broke"},
        "error_type": ErrorType.EXECUTION_ERROR,
        "confidence": 0.95,
        "suggested_agent": AgentType.EXECUTOR,
        "reasoning": "The code raised a RuntimeError during execution."
    }
    report = CriticReport(**report_data)
    assert report.status == CRITIC_STATUS_FAILURE_RUNTIME
    assert report.score == 0.1
    assert report.error_type == ErrorType.EXECUTION_ERROR
    assert report.confidence == 0.95
    assert report.suggested_agent == AgentType.EXECUTOR
    assert report.reasoning == "The code raised a RuntimeError during execution."
    assert report.is_failure()

def test_critic_report_with_string_enums():
    """Test instantiation with string values for enums."""
    report = CriticReport(
        status=CRITIC_STATUS_FAILURE_RUNTIME,
        score=0.1,
        error_type="EXECUTION_ERROR", # String value
        confidence=0.95,
        suggested_agent="EXECUTOR",   # String value
        reasoning="String enums test"
    )
    assert report.error_type == ErrorType.EXECUTION_ERROR
    assert report.suggested_agent == AgentType.EXECUTOR

# --- Test Validation Logic for New Fields ---

def test_critic_report_invalid_confidence_score():
    """Test validation for confidence score out of range."""
    with pytest.raises(ValidationError, match="Input should be less than or equal to 1"): # Pydantic v2 ge/le message
        CriticReport(
            status=CRITIC_STATUS_FAILURE_RUNTIME, score=0.0,
            error_type=ErrorType.EXECUTION_ERROR, confidence=1.5, # Invalid
            suggested_agent=AgentType.EXECUTOR, reasoning="High confidence"
        )
    with pytest.raises(ValidationError, match="Input should be greater than or equal to 0"): # Pydantic v2 ge/le message
        CriticReport(
            status=CRITIC_STATUS_FAILURE_RUNTIME, score=0.0,
            error_type=ErrorType.EXECUTION_ERROR, confidence=-0.5, # Invalid
            suggested_agent=AgentType.EXECUTOR, reasoning="Low confidence"
        )

def test_critic_report_invalid_error_type_string():
    """Test validation for invalid string for ErrorType."""
    with pytest.raises(ValidationError, match="Invalid error_type: INVALID_ERROR_TYPE_STRING"):
        CriticReport(
            status=CRITIC_STATUS_FAILURE_RUNTIME, score=0.0,
            error_type="INVALID_ERROR_TYPE_STRING", # Invalid
            confidence=0.5, suggested_agent=AgentType.EXECUTOR, reasoning="test"
        )

def test_critic_report_invalid_suggested_agent_string():
    """Test validation for invalid string for AgentType."""
    with pytest.raises(ValidationError, match="Invalid suggested_agent: INVALID_AGENT_TYPE_STRING"):
        CriticReport(
            status=CRITIC_STATUS_FAILURE_RUNTIME, score=0.0,
            error_type=ErrorType.EXECUTION_ERROR, confidence=0.5, 
            suggested_agent="INVALID_AGENT_TYPE_STRING", # Invalid
            reasoning="test"
        )

def test_critic_report_partial_classification_fields_for_failure():
    """Test that providing only some classification fields for a failure raises an error."""
    with pytest.raises(ValidationError, match="If any of error_type, confidence, suggested_agent, or reasoning are provided for a failure, all must be provided."):
        CriticReport(
            status=CRITIC_STATUS_FAILURE_RUNTIME,
            score=0.2,
            error_type=ErrorType.PLANNING_ERROR, # Provided
            confidence=0.9,                     # Provided
            suggested_agent=AgentType.PLANNER,  # Provided
            reasoning=None                      # Missing
        )
    
    with pytest.raises(ValidationError, match="If any of error_type, confidence, suggested_agent, or reasoning are provided for a failure, all must be provided."):
        CriticReport(
            status=CRITIC_STATUS_FAILURE_SYNTAX,
            score=0.0,
            error_type=ErrorType.EXECUTION_ERROR, # Provided
            # confidence, suggested_agent, reasoning are missing
        )

def test_critic_report_classification_fields_for_success():
    """Test that providing classification fields for a success report raises an error."""
    with pytest.raises(ValidationError, match="Error classification fields .* must not be present for success reports."):
        CriticReport(
            status=CRITIC_STATUS_SUCCESS,
            score=1.0,
            error_type=ErrorType.EXECUTION_ERROR, # Should not be here for success
            confidence=0.9,
            suggested_agent=AgentType.EXECUTOR,
            reasoning="This should not happen"
        )

def test_critic_report_success_with_no_classification_fields():
    """Test that a success report is valid with no classification fields."""
    try:
        CriticReport(
            status=CRITIC_STATUS_SUCCESS,
            score=1.0,
            summary="Everything is fine."
        )
    except ValidationError as e:
        pytest.fail(f"Instantiation of success report with no classification fields failed: {e}")

def test_critic_report_failure_with_no_classification_fields():
    """Test that a failure report is valid with no classification fields (for backward compatibility or pre-classification)."""
    try:
        CriticReport(
            status=CRITIC_STATUS_FAILURE_RUNTIME,
            score=0.1,
            summary="Crashed before classification.",
            error_details={"type": "RuntimeError", "message": "Something broke"}
        )
    except ValidationError as e:
        pytest.fail(f"Instantiation of failure report with no classification fields failed: {e}")

# Test is_failure() method
def test_is_failure_method():
    report_success = CriticReport(status=CRITIC_STATUS_SUCCESS, score=1.0)
    report_failure = CriticReport(
        status=CRITIC_STATUS_FAILURE_RUNTIME, score=0.0,
        error_type="EXECUTION_ERROR", confidence=0.9, 
        suggested_agent="EXECUTOR", reasoning="test"
    )
    assert not report_success.is_failure()
    assert report_failure.is_failure() 