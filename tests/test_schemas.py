import pytest
from dataclasses import asdict
from self_healing_agents.schemas import (
    CriticReport,
    CRITIC_STATUS_SUCCESS,
    CRITIC_STATUS_FAILURE_LOGIC,
    CRITIC_STATUS_FAILURE_RUNTIME,
    CRITIC_STATUS_FAILURE_SYNTAX
)
from self_healing_agents.error_types import ErrorType, AgentType

# Minimal valid data for old CriticReport (backward compatibility)
MINIMAL_OLD_REPORT_DATA = {
    "status": CRITIC_STATUS_SUCCESS,
    "score": 1.0,
}

# Valid data for new fields
VALID_NEW_FIELDS_DATA = {
    "error_type": ErrorType.PLANNING_ERROR,
    "confidence": 0.95,
    "suggested_agent": AgentType.PLANNER,
    "reasoning": "The overall plan missed a critical step."
}

class TestCriticReport:

    def test_critic_report_minimal_initialization(self):
        """Test basic initialization with only required fields (backward compatibility)."""
        report = CriticReport(status=CRITIC_STATUS_SUCCESS, score=1.0)
        assert report.status == CRITIC_STATUS_SUCCESS
        assert report.score == 1.0
        assert report.error_details is None
        assert report.test_results == []
        assert report.summary is None
        # New fields should default to None
        assert report.error_type is None
        assert report.confidence is None
        assert report.suggested_agent is None
        assert report.reasoning is None
        assert not report.is_failure()

    def test_critic_report_full_initialization_new_fields(self):
        """Test initialization with all new fields populated."""
        report_data = {**MINIMAL_OLD_REPORT_DATA, **VALID_NEW_FIELDS_DATA}
        report = CriticReport(**report_data)
        assert report.status == CRITIC_STATUS_SUCCESS
        assert report.score == 1.0
        assert report.error_type == ErrorType.PLANNING_ERROR
        assert report.confidence == 0.95
        assert report.suggested_agent == AgentType.PLANNER
        assert report.reasoning == "The overall plan missed a critical step."

    def test_critic_report_with_optional_old_fields(self):
        """Test initialization with optional old fields populated."""
        report = CriticReport(
            status=CRITIC_STATUS_FAILURE_LOGIC,
            score=0.2,
            error_details={"type": "LogicError", "message": "Algorithm incorrect"},
            test_results=[{"input": 1, "expected": 2, "actual": 1, "pass": False}],
            summary="Failed due to logical error in algorithm.",
            execution_stdout="Output: 1",
            execution_stderr="",
            generated_test_specifications=[{"name": "test_case_1"}],
            function_to_test="my_func",
            generated_code_for_report="def my_func(): return 1"
        )
        assert report.status == CRITIC_STATUS_FAILURE_LOGIC
        assert report.score == 0.2
        assert report.error_details == {"type": "LogicError", "message": "Algorithm incorrect"}
        assert len(report.test_results) == 1
        assert report.summary == "Failed due to logical error in algorithm."
        assert report.is_failure()

    # Validation Tests for new fields
    def test_confidence_valid_values(self):
        """Test valid confidence scores."""
        CriticReport(status="SUCCESS", score=1.0, confidence=0.0)
        CriticReport(status="SUCCESS", score=1.0, confidence=1.0)
        CriticReport(status="SUCCESS", score=1.0, confidence=0.75)
        # No exception should be raised

    def test_confidence_invalid_low(self):
        """Test confidence score below 0.0."""
        with pytest.raises(ValueError, match="Confidence score must be between 0.0 and 1.0."):
            CriticReport(status="SUCCESS", score=1.0, confidence=-0.1)

    def test_confidence_invalid_high(self):
        """Test confidence score above 1.0."""
        with pytest.raises(ValueError, match="Confidence score must be between 0.0 and 1.0."):
            CriticReport(status="SUCCESS", score=1.0, confidence=1.1)

    def test_error_type_invalid_type(self):
        """Test providing an invalid type for error_type."""
        with pytest.raises(TypeError, match="error_type must be an instance of ErrorType"):
            CriticReport(status="SUCCESS", score=1.0, error_type="NotAnErrorType")

    def test_suggested_agent_invalid_type(self):
        """Test providing an invalid type for suggested_agent."""
        with pytest.raises(TypeError, match="suggested_agent must be an instance of AgentType"):
            CriticReport(status="SUCCESS", score=1.0, suggested_agent="NotAnAgentType")

    def test_error_type_valid_enum(self):
        """Test valid ErrorType enum for error_type."""
        report = CriticReport(status="SUCCESS", score=1.0, error_type=ErrorType.EXECUTION_ERROR)
        assert report.error_type == ErrorType.EXECUTION_ERROR

    def test_suggested_agent_valid_enum(self):
        """Test valid AgentType enum for suggested_agent."""
        report = CriticReport(status="SUCCESS", score=1.0, suggested_agent=AgentType.EXECUTOR)
        assert report.suggested_agent == AgentType.EXECUTOR

    def test_is_failure_method(self):
        """Test the is_failure() method."""
        report_success = CriticReport(status=CRITIC_STATUS_SUCCESS, score=1.0)
        assert not report_success.is_failure()

        report_failure = CriticReport(status=CRITIC_STATUS_FAILURE_LOGIC, score=0.1)
        assert report_failure.is_failure()

    def test_serialization_basic_asdict(self):
        """Test basic serialization using asdict for dataclasses (if no custom methods)."""
        report_data = {
            "status": CRITIC_STATUS_FAILURE_RUNTIME,
            "score": 0.1,
            "error_details": {"type": "RuntimeError", "message": "Oops"},
            "test_results": [],
            "summary": "Crashed.",
            "error_type": ErrorType.EXECUTION_ERROR,
            "confidence": 0.99,
            "suggested_agent": AgentType.EXECUTOR,
            "reasoning": "Runtime exception clearly indicates execution phase fault."
        }
        report = CriticReport(**report_data)
        serialized_report = asdict(report)

        assert serialized_report["status"] == CRITIC_STATUS_FAILURE_RUNTIME
        assert serialized_report["score"] == 0.1
        assert serialized_report["error_type"] == ErrorType.EXECUTION_ERROR # Enums will be themselves
        assert serialized_report["confidence"] == 0.99
        assert serialized_report["suggested_agent"] == AgentType.EXECUTOR
        assert serialized_report["reasoning"] == "Runtime exception clearly indicates execution phase fault."

    def test_backward_compatibility_instantiation(self):
        """Ensure that a report can be instantiated with only old fields."""
        try:
            report = CriticReport(
                status=CRITIC_STATUS_FAILURE_SYNTAX,
                score=0.0,
                error_details={"type": "SyntaxError"},
                summary="Syntax issue."
            )
            assert report.status == CRITIC_STATUS_FAILURE_SYNTAX
            assert report.score == 0.0
            assert report.error_type is None
            assert report.confidence is None
            assert report.suggested_agent is None
            assert report.reasoning is None
        except Exception as e:
            pytest.fail(f"Backward compatibility instantiation failed: {e}") 