import unittest
from unittest.mock import patch, MagicMock
import os
import sys

# Ensure the test can find the src directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.self_healing_agents.evaluation.fix_evaluator import ErrorFixEvaluator, FixAssessment
from src.self_healing_agents.schemas import CriticReport


class TestErrorFixEvaluator(unittest.TestCase):
    
    def setUp(self):
        self.evaluator = ErrorFixEvaluator()
        
        # Sample code with a division by zero error
        self.original_code = """def divide(a, b):
    return a / b

result = divide(10, 0)
print(f"Result: {result}")
"""
        
        # Fixed code that handles division by zero
        self.fixed_code = """def divide(a, b):
    if b == 0:
        return "Cannot divide by zero"
    return a / b

result = divide(10, 0)
print(f"Result: {result}")
"""
        
        # Sample code with a syntax error
        self.syntax_error_code = """def calculate_sum(numbers)
    total = 0
    for num in numbers:
        total += num
    return total
"""

        # Fixed syntax error code
        self.fixed_syntax_code = """def calculate_sum(numbers):
    total = 0
    for num in numbers:
        total += num
    return total
"""
        
        # Original error details
        self.division_error_details = {
            "error_type": "ZeroDivisionError",
            "error_message": "division by zero",
            "traceback": "Traceback (most recent call last):\n  File \"<string>\", line 4, in <module>\n  File \"<string>\", line 2, in divide\nZeroDivisionError: division by zero"
        }
        
        # No error in fixed code
        self.no_error_details = None
        
        # Different error in fixed code
        self.new_error_details = {
            "error_type": "TypeError",
            "error_message": "unsupported operand type(s) for +: 'int' and 'str'",
            "traceback": "Traceback (most recent call last):\n  File \"<string>\", line 5, in <module>\nTypeError: unsupported operand type(s) for +: 'int' and 'str'"
        }
        
        # Syntax error details
        self.syntax_error_details = {
            "error_type": "SyntaxError",
            "error_message": "invalid syntax",
            "traceback": "  File \"<string>\", line 1\n    def calculate_sum(numbers)\n                           ^\nSyntaxError: invalid syntax"
        }
        
        # Sample test results
        self.original_test_results = [
            {"name": "test_divide_normal", "status": "passed", "inputs": {"a": 10, "b": 2}, "expected_output": 5, "actual_output": 5},
            {"name": "test_divide_zero", "status": "failed", "inputs": {"a": 10, "b": 0}, "expected_output": "Error", "actual_output": None},
        ]
        
        self.fixed_test_results = [
            {"name": "test_divide_normal", "status": "passed", "inputs": {"a": 10, "b": 2}, "expected_output": 5, "actual_output": 5},
            {"name": "test_divide_zero", "status": "passed", "inputs": {"a": 10, "b": 0}, "expected_output": "Cannot divide by zero", "actual_output": "Cannot divide by zero"},
        ]

    def test_evaluate_fix_successful_fix(self):
        """Test evaluating a successful fix that resolves the error and improves test pass rate."""
        assessment = self.evaluator.evaluate_fix(
            original_code=self.original_code,
            fixed_code=self.fixed_code,
            original_error_details=self.division_error_details,
            fixed_error_details=self.no_error_details,
            original_test_results=self.original_test_results,
            fixed_test_results=self.fixed_test_results
        )
        
        # Verify the assessment
        self.assertTrue(assessment.original_error_resolved)
        self.assertFalse(assessment.error_still_present)
        self.assertFalse(assessment.new_errors_introduced)
        self.assertFalse(assessment.error_type_changed)
        self.assertGreater(assessment.test_improvement, 0)
        self.assertGreater(assessment.code_change_magnitude, 0)
        self.assertGreater(assessment.fix_quality_score, 0.7)  # High quality fix
        
        # Verify details
        self.assertIn("diff", assessment.details)
        self.assertIn("if b == 0", assessment.details["diff"])
        self.assertEqual(len(assessment.details["test_changes"]["newly_passing"]), 1)
        self.assertEqual(assessment.details["test_changes"]["newly_passing"][0], "test_divide_zero")

    def test_evaluate_fix_unsuccessful_fix(self):
        """Test evaluating an unsuccessful fix that changes the error type."""
        assessment = self.evaluator.evaluate_fix(
            original_code=self.original_code,
            fixed_code=self.fixed_code,
            original_error_details=self.division_error_details,
            fixed_error_details=self.new_error_details,
            original_test_results=self.original_test_results,
            fixed_test_results=self.original_test_results  # Same tests, still failing
        )
        
        # Verify the assessment
        self.assertFalse(assessment.original_error_resolved)
        self.assertFalse(assessment.error_still_present)
        self.assertTrue(assessment.new_errors_introduced)
        self.assertTrue(assessment.error_type_changed)
        self.assertEqual(assessment.test_improvement, 0)
        self.assertGreater(assessment.code_change_magnitude, 0)
        self.assertLess(assessment.fix_quality_score, 0.5)  # Low quality fix

    def test_evaluate_fix_syntax_error(self):
        """Test evaluating a fix for a syntax error."""
        assessment = self.evaluator.evaluate_fix(
            original_code=self.syntax_error_code,
            fixed_code=self.fixed_syntax_code,
            original_error_details=self.syntax_error_details,
            fixed_error_details=self.no_error_details,
            original_test_results=None,  # No tests for syntax error
            fixed_test_results=None
        )
        
        # Verify the assessment
        self.assertTrue(assessment.original_error_resolved)
        self.assertFalse(assessment.error_still_present)
        self.assertFalse(assessment.new_errors_introduced)
        self.assertFalse(assessment.error_type_changed)
        self.assertEqual(assessment.test_improvement, 0)  # No tests to improve
        self.assertLessEqual(assessment.code_change_magnitude, 0.5)  # Small to moderate change
        self.assertGreater(assessment.fix_quality_score, 0.7)  # High quality fix (error resolved with minimal changes)

    def test_evaluate_fix_from_reports(self):
        """Test evaluating fix with CriticReport objects."""
        # Create mock CriticReport objects
        original_report = CriticReport(
            status="FAILURE_RUNTIME", 
            score=0.2,
            error_details=self.division_error_details,
            test_results=self.original_test_results,
            summary="Code failed with division by zero"
        )
        
        fixed_report = CriticReport(
            status="SUCCESS", 
            score=1.0,
            error_details=None,
            test_results=self.fixed_test_results,
            summary="All tests passed"
        )
        
        assessment = self.evaluator.evaluate_fix_from_reports(
            original_report=original_report,
            fixed_report=fixed_report,
            original_code=self.original_code,
            fixed_code=self.fixed_code
        )
        
        # Verify the assessment
        self.assertTrue(assessment.original_error_resolved)
        self.assertFalse(assessment.error_still_present)
        self.assertFalse(assessment.new_errors_introduced)
        self.assertGreater(assessment.test_improvement, 0)

    def test_similar_error_messages(self):
        """Test the similar error messages function."""
        # Same error with slight differences in wording
        msg1 = "division by zero at line 5"
        msg2 = "division by zero at line 10"
        self.assertTrue(self.evaluator._similar_error_messages(msg1, msg2))
        
        # Different errors
        msg3 = "name 'x' is not defined"
        self.assertFalse(self.evaluator._similar_error_messages(msg1, msg3))
        
        # Empty messages
        self.assertFalse(self.evaluator._similar_error_messages("", ""))

    def test_calculate_test_improvement(self):
        """Test calculation of test improvement."""
        # All tests pass in fixed version
        improvement = self.evaluator._calculate_test_improvement(
            self.original_test_results,  # 1/2 pass
            self.fixed_test_results      # 2/2 pass
        )
        self.assertEqual(improvement, 0.5)
        
        # Regression (tests start failing)
        regression = self.evaluator._calculate_test_improvement(
            self.fixed_test_results,     # 2/2 pass
            self.original_test_results   # 1/2 pass
        )
        self.assertEqual(regression, -0.5)
        
        # No tests
        no_tests = self.evaluator._calculate_test_improvement(None, None)
        self.assertEqual(no_tests, 0.0)

    def test_calculate_code_change_magnitude(self):
        """Test calculation of code change magnitude."""
        # Small change
        small_change = self.evaluator._calculate_code_change_magnitude(
            self.syntax_error_code,
            self.fixed_syntax_code
        )
        self.assertLessEqual(small_change, 0.5)  # Allowing for a bit more change
        
        # Larger change
        larger_change = self.evaluator._calculate_code_change_magnitude(
            self.original_code,
            self.fixed_code
        )
        self.assertGreater(larger_change, 0)
        
        # No change
        no_change = self.evaluator._calculate_code_change_magnitude(
            self.original_code,
            self.original_code
        )
        self.assertEqual(no_change, 0.0)
        
        # Complete change
        complete_change = self.evaluator._calculate_code_change_magnitude(
            self.original_code,
            "# Completely different code\ndef new_function(): pass"
        )
        self.assertGreater(complete_change, 0.5)

    def test_is_error_resolved(self):
        """Test error resolution check."""
        self.assertTrue(self.evaluator.is_error_resolved(None))
        self.assertFalse(self.evaluator.is_error_resolved({"error_type": "SyntaxError"}))

    def test_fix_quality_calculation(self):
        """Test fix quality score calculation."""
        # Successful fix, good test improvement, small changes
        high_quality = self.evaluator._calculate_fix_quality(
            error_resolved=True, 
            new_errors_introduced=False, 
            test_improvement=0.5, 
            change_magnitude=0.1
        )
        self.assertGreaterEqual(high_quality, 0.8)
        
        # Error not resolved
        low_quality = self.evaluator._calculate_fix_quality(
            error_resolved=False, 
            new_errors_introduced=False, 
            test_improvement=0, 
            change_magnitude=0.1
        )
        self.assertLessEqual(low_quality, 0.2)
        
        # New errors introduced
        new_errors = self.evaluator._calculate_fix_quality(
            error_resolved=True, 
            new_errors_introduced=True, 
            test_improvement=0, 
            change_magnitude=0.1
        )
        self.assertLess(new_errors, high_quality)


if __name__ == '__main__':
    unittest.main() 