import unittest
from unittest.mock import patch, MagicMock
import time
import os
import sys

# Ensure the test can find the src directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.self_healing_agents.error_resolution import SimpleErrorResolver, SimpleFixAttempt


class TestSimpleErrorResolver(unittest.TestCase):
    
    def setUp(self):
        self.resolver = SimpleErrorResolver()
        self.sample_code = """def divide_numbers(a, b):
    return a / b

result = divide_numbers(10, 0)
print(f"Result: {result}")
"""
        self.division_error_details = {
            "error_type": "ZeroDivisionError",
            "error_message": "division by zero",
            "traceback": "Traceback (most recent call last):\n  File \"<string>\", line 4, in <module>\n  File \"<string>\", line 2, in divide_numbers\nZeroDivisionError: division by zero"
        }

    def test_format_error_for_prompt(self):
        # Test with minimal error details
        minimal_error = {"error_type": "TypeError", "error_message": "unsupported operand type(s)"}
        minimal_formatted = self.resolver.format_error_for_prompt(minimal_error)
        
        self.assertIn("### ERROR DETAILS ###", minimal_formatted)
        self.assertIn("Error Type: TypeError", minimal_formatted)
        self.assertIn("Error Message: unsupported operand type(s)", minimal_formatted)
        
        # Test with full error details and code context
        code_context = "4:     result = divide_numbers(10, 0)"
        full_formatted = self.resolver.format_error_for_prompt(
            self.division_error_details, 
            code_context
        )
        
        self.assertIn("Error Type: ZeroDivisionError", full_formatted)
        self.assertIn("Error Message: division by zero", full_formatted)
        self.assertIn("Traceback:", full_formatted)
        self.assertIn("Error Location:", full_formatted)
        self.assertIn("```python", full_formatted)
        self.assertIn("4:     result = divide_numbers(10, 0)", full_formatted)
        self.assertIn("```", full_formatted)
        self.assertIn("Please fix the error above", full_formatted)

    def test_extract_error_context(self):
        # Test with valid line number in traceback
        context = self.resolver.extract_error_context(self.sample_code, self.division_error_details)
        
        self.assertIsNotNone(context)
        self.assertIn("2: >>>     return a / b", context)  # The error occurs on line 2
        
        # Test with no line number information
        no_line_error = {
            "error_type": "SyntaxError",
            "error_message": "invalid syntax"
        }
        context = self.resolver.extract_error_context(self.sample_code, no_line_error)
        self.assertIsNone(context)
        
        # Test with empty code
        context = self.resolver.extract_error_context("", self.division_error_details)
        self.assertIsNone(context)
        
        # Test with stderr instead of traceback
        stderr_error = {
            "error_type": "SyntaxError",
            "error_message": "invalid syntax",
            "stderr": "  File \"<string>\", line 3\n    if x == 5\n            ^\nSyntaxError: invalid syntax"
        }
        context = self.resolver.extract_error_context("x = 10\ny = 20\nif x == 5\n    print(x)", stderr_error)
        self.assertIsNotNone(context)
        self.assertIn("3: >>> if x == 5", context)

    def test_append_error_to_prompt(self):
        original_prompt = "You are a Python expert. Write efficient and correct code."
        
        modified_prompt, fix_attempt = self.resolver.append_error_to_prompt(
            original_prompt, 
            self.division_error_details, 
            self.sample_code
        )
        
        # Check the prompt was modified correctly
        self.assertIn(original_prompt, modified_prompt)
        self.assertIn("### ERROR DETAILS ###", modified_prompt)
        self.assertIn("Error Type: ZeroDivisionError", modified_prompt)
        
        # Check the fix attempt was created correctly
        self.assertEqual(fix_attempt.error_details, self.division_error_details)
        self.assertEqual(fix_attempt.original_prompt, original_prompt)
        self.assertEqual(fix_attempt.modified_prompt, modified_prompt)
        self.assertFalse(fix_attempt.successful)
        
        # Check the attempt was tracked
        self.assertEqual(len(self.resolver.fix_attempts), 1)
        self.assertEqual(self.resolver.fix_attempts[0], fix_attempt)

    def test_record_fix_success(self):
        # Create a fix attempt
        fix_attempt = SimpleFixAttempt(
            error_details={"error_type": "ValueError"},
            original_prompt="Original prompt",
            modified_prompt="Modified prompt"
        )
        
        # Record it as successful
        self.resolver.fix_attempts.append(fix_attempt)
        self.resolver.record_fix_success(fix_attempt, True)
        
        # Check it was updated
        self.assertTrue(fix_attempt.successful)
        
        # Create another attempt and mark it as failed
        another_attempt = SimpleFixAttempt(
            error_details={"error_type": "TypeError"},
            original_prompt="Original prompt 2",
            modified_prompt="Modified prompt 2"
        )
        
        self.resolver.fix_attempts.append(another_attempt)
        self.resolver.record_fix_success(another_attempt, False)
        
        # Check it was updated
        self.assertFalse(another_attempt.successful)

    def test_get_success_rate(self):
        # Test with no attempts
        self.assertEqual(self.resolver.get_success_rate(), 0.0)
        
        # Test with one successful attempt
        attempt1 = SimpleFixAttempt(
            error_details={"error_type": "ValueError"},
            original_prompt="Original prompt",
            modified_prompt="Modified prompt",
            successful=True
        )
        self.resolver.fix_attempts.append(attempt1)
        self.assertEqual(self.resolver.get_success_rate(), 1.0)
        
        # Test with mixed results
        attempt2 = SimpleFixAttempt(
            error_details={"error_type": "TypeError"},
            original_prompt="Original prompt 2",
            modified_prompt="Modified prompt 2",
            successful=False
        )
        self.resolver.fix_attempts.append(attempt2)
        self.assertEqual(self.resolver.get_success_rate(), 0.5)
        
        # Test with all failures
        attempt3 = SimpleFixAttempt(
            error_details={"error_type": "SyntaxError"},
            original_prompt="Original prompt 3",
            modified_prompt="Modified prompt 3",
            successful=False
        )
        self.resolver.fix_attempts.append(attempt3)
        self.assertEqual(self.resolver.get_success_rate(), 1/3)

    def test_get_fix_history(self):
        # Create some attempts with different timestamps
        attempt1 = SimpleFixAttempt(
            error_details={"error_type": "ValueError"},
            original_prompt="Original prompt",
            modified_prompt="Modified prompt",
            successful=True,
            timestamp=1000.0
        )
        
        attempt2 = SimpleFixAttempt(
            error_details={"error_type": "TypeError"},
            original_prompt="Same prompt",
            modified_prompt="Same prompt",  # No modification
            successful=False,
            timestamp=2000.0
        )
        
        self.resolver.fix_attempts.extend([attempt1, attempt2])
        
        history = self.resolver.get_fix_history()
        
        # Check the history format
        self.assertEqual(len(history), 2)
        
        # Check the first attempt
        self.assertEqual(history[0]["error_type"], "ValueError")
        self.assertEqual(history[0]["timestamp"], 1000.0)
        self.assertTrue(history[0]["successful"])
        self.assertTrue(history[0]["prompt_modified"])
        
        # Check the second attempt
        self.assertEqual(history[1]["error_type"], "TypeError")
        self.assertEqual(history[1]["timestamp"], 2000.0)
        self.assertFalse(history[1]["successful"])
        self.assertFalse(history[1]["prompt_modified"])


if __name__ == '__main__':
    unittest.main() 