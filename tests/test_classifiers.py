import pytest
from self_healing_agents.classifiers import BaseErrorClassifier, ErrorDataType
from self_healing_agents.error_types import ErrorType, HIGH_CONFIDENCE_THRESHOLD, MEDIUM_CONFIDENCE_THRESHOLD


class TestBaseErrorClassifier:
    classifier = BaseErrorClassifier()

    @pytest.mark.parametrize(
        "error_data, expected_error_type, expected_confidence",
        [
            # Heuristic 1: SyntaxError
            ({"exception_type": "SyntaxError", "error_message": "invalid syntax"},
             ErrorType.EXECUTION_ERROR, HIGH_CONFIDENCE_THRESHOLD),
            
            # Heuristic 2: Common Runtime Errors (from exception_type)
            ({"exception_type": "NameError", "error_message": "name 'x' is not defined"},
             ErrorType.EXECUTION_ERROR, (HIGH_CONFIDENCE_THRESHOLD + MEDIUM_CONFIDENCE_THRESHOLD) / 2),
            ({"exception_type": "TypeError", "error_message": "unsupported operand type(s)"},
             ErrorType.EXECUTION_ERROR, (HIGH_CONFIDENCE_THRESHOLD + MEDIUM_CONFIDENCE_THRESHOLD) / 2),
            
            # Heuristic 2: Common Runtime Errors (from error_message keyword)
            ({"error_message": "some traceback with zerodivisionerror details"},
             ErrorType.EXECUTION_ERROR, (HIGH_CONFIDENCE_THRESHOLD + MEDIUM_CONFIDENCE_THRESHOLD) / 2),
            ({"error_message": "[Errno 2] No such file or directory: 'nonexistent.txt'", "exception_type": "FileNotFoundError"},
             ErrorType.EXECUTION_ERROR, (HIGH_CONFIDENCE_THRESHOLD + MEDIUM_CONFIDENCE_THRESHOLD) / 2),
            ({"error_message": "keyerror: 'missing_key'"},
             ErrorType.EXECUTION_ERROR, (HIGH_CONFIDENCE_THRESHOLD + MEDIUM_CONFIDENCE_THRESHOLD) / 2),
            ({"error_message": "ValueError: invalid literal for int() with base 10"},
                ErrorType.EXECUTION_ERROR, (HIGH_CONFIDENCE_THRESHOLD + MEDIUM_CONFIDENCE_THRESHOLD) / 2),
            ({"error_message": "ImportError: cannot import name 'SpecificClass'"},
                ErrorType.EXECUTION_ERROR, (HIGH_CONFIDENCE_THRESHOLD + MEDIUM_CONFIDENCE_THRESHOLD) / 2),
            ({"error_message": "ModuleNotFoundError: No module named 'non_existent_module'"},
                ErrorType.EXECUTION_ERROR, (HIGH_CONFIDENCE_THRESHOLD + MEDIUM_CONFIDENCE_THRESHOLD) / 2),

            # Heuristic 3: Test Failures
            ({"test_failures": [{"name": "test_addition", "status": "failed"}]},
             ErrorType.EXECUTION_ERROR, MEDIUM_CONFIDENCE_THRESHOLD),
            ({"error_message": "assertion failed", "test_failures": [{"name": "test_logic", "status": "failed"}]},
             ErrorType.EXECUTION_ERROR, MEDIUM_CONFIDENCE_THRESHOLD), # error_message here is generic, test_failures dominate

            # Ambiguous / Default cases
            ({"error_message": "Something unexpected happened but not a known runtime keyword."},
             ErrorType.AMBIGUOUS_ERROR, MEDIUM_CONFIDENCE_THRESHOLD / 2),
            ({"exception_type": "CustomAppException", "error_message": "Application specific issue"},
             ErrorType.AMBIGUOUS_ERROR, MEDIUM_CONFIDENCE_THRESHOLD / 2),
            
            # Very low confidence if no information
            ({},
             ErrorType.AMBIGUOUS_ERROR, 0.1),
            ({"some_other_key": "some_value"}, # No relevant keys
             ErrorType.AMBIGUOUS_ERROR, 0.1),

            # Edge case: Syntax error message but not exception type (should still be caught by message if keyword present)
            # This scenario is less likely as SyntaxError exception type is usually present.
            # The current heuristic prioritizes exception_type for SyntaxError.
            # If exception_type is NOT SyntaxError, then message check applies.
            ({"error_message": "syntaxerror: invalid syntax"}, 
             ErrorType.EXECUTION_ERROR, (HIGH_CONFIDENCE_THRESHOLD + MEDIUM_CONFIDENCE_THRESHOLD) / 2), # Caught by message keyword

            # What if exception_type is present but not a runtime keyword, and message is generic, but test fails?
            ({"exception_type": "GenericException", "error_message": "failed", "test_failures": [{"name": "test_1"}]},
             ErrorType.EXECUTION_ERROR, MEDIUM_CONFIDENCE_THRESHOLD), # test_failures rule applies

            # Empty error message, but specific exception type
            ({"exception_type": "IndexError", "error_message": ""},
             ErrorType.EXECUTION_ERROR, (HIGH_CONFIDENCE_THRESHOLD + MEDIUM_CONFIDENCE_THRESHOLD) / 2),
        ]
    )
    def test_classify_error_scenarios(self, error_data: ErrorDataType, expected_error_type: ErrorType, expected_confidence: float):
        """Test various error scenarios against BaseErrorClassifier."""
        error_type, confidence = self.classifier.classify_error(error_data)
        assert error_type == expected_error_type
        assert confidence == pytest.approx(expected_confidence)

    def test_classify_error_empty_input(self):
        """Test with completely empty error_data."""
        error_type, confidence = self.classifier.classify_error({})
        assert error_type == ErrorType.AMBIGUOUS_ERROR
        assert confidence == pytest.approx(0.1)

    def test_no_relevant_keys(self):
        """Test with error_data containing no relevant keys."""
        error_data = {"foo": "bar", "details": "some details"}
        error_type, confidence = self.classifier.classify_error(error_data)
        assert error_type == ErrorType.AMBIGUOUS_ERROR
        assert confidence == pytest.approx(0.1)

    def test_error_message_case_insensitivity(self):
        """Test that error message keyword matching is case-insensitive."""
        error_data = {"error_message": "encountered a TypeError during processing"}
        error_type, confidence = self.classifier.classify_error(error_data)
        assert error_type == ErrorType.EXECUTION_ERROR
        assert confidence == pytest.approx((HIGH_CONFIDENCE_THRESHOLD + MEDIUM_CONFIDENCE_THRESHOLD) / 2)

    def test_exception_type_takes_precedence_for_syntax_error(self):
        """If exception_type is SyntaxError, it should be EXECUTION_ERROR with high confidence, 
           even if other fields might suggest something else at lower confidence."""
        error_data = {"exception_type": "SyntaxError", "error_message": "Assertion failed", "test_failures": [{}]}
        error_type, confidence = self.classifier.classify_error(error_data)
        assert error_type == ErrorType.EXECUTION_ERROR
        assert confidence == pytest.approx(HIGH_CONFIDENCE_THRESHOLD)

    def test_runtime_error_type_takes_precedence_over_test_failure_heuristic(self):
        """If a known runtime exception_type is present, it should classify as EXECUTION_ERROR
           with its confidence, even if test_failures are also present."""
        error_data = {"exception_type": "NameError", "error_message": "x not found", "test_failures": [{}]}
        error_type, confidence = self.classifier.classify_error(error_data)
        assert error_type == ErrorType.EXECUTION_ERROR
        assert confidence == pytest.approx((HIGH_CONFIDENCE_THRESHOLD + MEDIUM_CONFIDENCE_THRESHOLD) / 2)

    def test_runtime_error_message_takes_precedence_over_test_failure_heuristic(self):
        """If a known runtime error keyword is in message, it should classify as EXECUTION_ERROR
           with its confidence, even if test_failures are also present (and no overriding exception_type)."""
        error_data = {"error_message": "got a valueerror here", "test_failures": [{}]}
        error_type, confidence = self.classifier.classify_error(error_data)
        assert error_type == ErrorType.EXECUTION_ERROR
        assert confidence == pytest.approx((HIGH_CONFIDENCE_THRESHOLD + MEDIUM_CONFIDENCE_THRESHOLD) / 2) 