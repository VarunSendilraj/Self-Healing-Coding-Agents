import pytest
from pydantic import ValidationError

from self_healing_agents.classifiers.base import (
    ErrorClassifier,
    ClassificationResult
)
from self_healing_agents.error_types import ErrorType, AgentType
from typing import Any, Dict, List, Optional

# --- Test ClassificationResult Model ---
def test_classification_result_instantiation():
    """Test successful instantiation of ClassificationResult."""
    result = ClassificationResult(
        error_type=ErrorType.EXECUTION_ERROR,
        confidence=0.85,
        suggested_agent=AgentType.EXECUTOR,
        reasoning="A runtime error occurred due to incorrect API usage.",
        raw_classifier_output={"detail": "some raw log"}
    )
    assert result.error_type == ErrorType.EXECUTION_ERROR
    assert result.confidence == 0.85
    assert result.suggested_agent == AgentType.EXECUTOR
    assert result.reasoning == "A runtime error occurred due to incorrect API usage."
    assert result.raw_classifier_output == {"detail": "some raw log"}

def test_classification_result_instantiation_minimal():
    """Test successful instantiation with minimal required fields."""
    result = ClassificationResult(
        error_type=ErrorType.PLANNING_ERROR,
        confidence=0.60,
        suggested_agent=AgentType.PLANNER,
        reasoning="Logic flaw in plan."
    )
    assert result.error_type == ErrorType.PLANNING_ERROR
    assert result.confidence == 0.60
    assert result.suggested_agent == AgentType.PLANNER
    assert result.reasoning == "Logic flaw in plan."
    assert result.raw_classifier_output is None

@pytest.mark.parametrize(
    "invalid_confidence", [-0.1, 1.1, "not_a_float"]
)
def test_classification_result_invalid_confidence(invalid_confidence):
    """Test ClassificationResult raises ValidationError for out-of-range confidence."""
    with pytest.raises(ValidationError):
        ClassificationResult(
            error_type=ErrorType.AMBIGUOUS_ERROR,
            confidence=invalid_confidence,
            suggested_agent=AgentType.UNKNOWN_AGENT,
            reasoning="Uncertain."
        )

def test_classification_result_invalid_enum_types():
    """Test ClassificationResult raises ValidationError for invalid enum string values."""
    with pytest.raises(ValidationError):
        ClassificationResult(
            error_type="NOT_AN_ERROR_TYPE",
            confidence=0.5,
            suggested_agent=AgentType.EXECUTOR,
            reasoning="Invalid error type string."
        )
    with pytest.raises(ValidationError):
        ClassificationResult(
            error_type=ErrorType.EXECUTION_ERROR,
            confidence=0.5,
            suggested_agent="NOT_AN_AGENT_TYPE",
            reasoning="Invalid agent type string."
        )

# --- Test ErrorClassifier ABC --- 

class ConcreteClassifier(ErrorClassifier):
    """A minimal concrete implementation for testing purposes."""
    def classify_error(
        self,
        error_details: Dict[str, Any],
        code_context: Optional[str] = None,
        execution_history: Optional[List[Dict[str, Any]]] = None,
        test_results: Optional[List[Dict[str, Any]]] = None,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> ClassificationResult:
        # Dummy implementation for the test
        return ClassificationResult(
            error_type=ErrorType.UNKNOWN_ERROR,
            confidence=0.1,
            suggested_agent=AgentType.UNKNOWN_AGENT,
            reasoning="Dummy classification from ConcreteClassifier."
        )

def test_error_classifier_abc_cannot_be_instantiated():
    """Test that the ErrorClassifier ABC cannot be instantiated directly."""
    with pytest.raises(TypeError, match="Can't instantiate abstract class ErrorClassifier with abstract method classify_error"):
        ErrorClassifier() # type: ignore

def test_concrete_classifier_can_be_instantiated():
    """Test that a concrete subclass of ErrorClassifier can be instantiated."""
    try:
        classifier = ConcreteClassifier()
        assert isinstance(classifier, ErrorClassifier)
    except Exception as e:
        pytest.fail(f"ConcreteClassifier instantiation failed: {e}")

def test_concrete_classifier_must_implement_classify_error():
    """Test that a concrete classifier instance calls its classify_error method."""
    classifier = ConcreteClassifier()
    dummy_error_details = {"type": "TestError", "message": "This is a test."}
    result = classifier.classify_error(error_details=dummy_error_details)
    assert isinstance(result, ClassificationResult)
    assert result.reasoning == "Dummy classification from ConcreteClassifier."

# --- Test Utility Methods of ErrorClassifier (via ConcreteClassifier instance) ---
def test_extract_patterns_from_trace_utility():
    """Test the basic functionality of the extract_patterns_from_trace utility."""
    classifier = ConcreteClassifier() # Can use the concrete one to test base utilities
    
    assert classifier.extract_patterns_from_trace(None) == {"trace_analysis_status": "No traceback provided"}
    
    syntax_trace = "Traceback (most recent call last):\n  File \"<stdin>\", line 1, in <module>\nSyntaxError: invalid syntax"
    assert classifier.extract_patterns_from_trace(syntax_trace) == {"contains_syntax_error": True}
    
    runtime_trace = "Traceback (most recent call last):\n  File \"<stdin>\", line 1, in <module>\nRuntimeError: something went wrong"
    # Current basic impl only checks SyntaxError and RuntimeError literally
    expected_runtime = {"contains_runtime_error": True}
    if "SyntaxError" in runtime_trace: # our basic impl is a bit naive
        expected_runtime["contains_syntax_error"] = True
    assert classifier.extract_patterns_from_trace(runtime_trace) == expected_runtime
    
    other_trace = "Some other error message without keywords"
    assert classifier.extract_patterns_from_trace(other_trace) == {}

def test_analyze_test_failures_utility():
    """Test the basic functionality of the analyze_test_failures utility."""
    classifier = ConcreteClassifier()
    
    assert classifier.analyze_test_failures(None) == {"test_analysis_status": "No test results provided"}
    
    all_passed = [
        {"test_name": "test1", "passed": True},
        {"test_name": "test2", "passed": True}
    ]
    assert classifier.analyze_test_failures(all_passed) == {
        "total_tests": 2, "passed_count": 2, "failed_count": 0, "failure_types": []
    }
    
    some_failed = [
        {"test_name": "test1", "passed": True},
        {"test_name": "test2", "passed": False, "error_type": "AssertionError"},
        {"test_name": "test3", "passed": False, "error_type": "ValueError"}
    ]
    assert classifier.analyze_test_failures(some_failed) == {
        "total_tests": 3, "passed_count": 1, "failed_count": 2, "failure_types": ["AssertionError", "ValueError"]
    }
    
    all_failed_no_type = [
        {"test_name": "test1", "passed": False},
        {"test_name": "test2", "passed": False}
    ]
    assert classifier.analyze_test_failures(all_failed_no_type) == {
        "total_tests": 2, "passed_count": 0, "failed_count": 2, "failure_types": []
    } 