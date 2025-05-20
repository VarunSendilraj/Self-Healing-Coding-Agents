import unittest
import sys
import os
import re

# Ensure the src directory is in the Python path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from self_healing_agents.error_types import ErrorType, HIGH_CONFIDENCE_THRESHOLD, MEDIUM_CONFIDENCE_THRESHOLD
from self_healing_agents.classifiers.rule_based import RuleBasedErrorClassifier

class TestRuleBasedErrorClassifier(unittest.TestCase):
    """
    Comprehensive test suite for the RuleBasedErrorClassifier.
    Tests various error scenarios to validate classification accuracy.
    """
    
    def setUp(self):
        """Initialize the classifier before each test."""
        self.classifier = RuleBasedErrorClassifier()
    
    # Syntax Error Tests
    
    def test_syntax_error_missing_colon(self):
        """Test classification of syntax error due to missing colon."""
        error_data = {
            'exception_type': 'SyntaxError',
            'error_message': 'SyntaxError: invalid syntax',
            'traceback': 'File "<string>", line 1\n def add(a, b) return a + b\n                ^\nSyntaxError: invalid syntax',
            'code_snippet': 'def add(a, b) return a + b'
        }
        error_type, confidence = self.classifier.classify_error(error_data)
        self.assertEqual(error_type, ErrorType.EXECUTION_ERROR)
        self.assertGreaterEqual(confidence, HIGH_CONFIDENCE_THRESHOLD)
        
    def test_syntax_error_missing_parenthesis(self):
        """Test classification of syntax error due to missing parenthesis."""
        error_data = {
            'exception_type': 'SyntaxError',
            'error_message': 'SyntaxError: unexpected EOF while parsing',
            'traceback': 'File "<string>", line 1\n print("Hello World"\nSyntaxError: unexpected EOF while parsing',
            'code_snippet': 'print("Hello World"'
        }
        error_type, confidence = self.classifier.classify_error(error_data)
        self.assertEqual(error_type, ErrorType.EXECUTION_ERROR)
        self.assertGreaterEqual(confidence, HIGH_CONFIDENCE_THRESHOLD)
    
    def test_syntax_error_invalid_indentation(self):
        """Test classification of syntax error due to invalid indentation."""
        error_data = {
            'exception_type': 'IndentationError',
            'error_message': 'IndentationError: unexpected indent',
            'traceback': 'File "<string>", line 2\n  print("Hello")\n  ^\nIndentationError: unexpected indent',
            'code_snippet': 'def hello():\nprint("Hello")'
        }
        error_type, confidence = self.classifier.classify_error(error_data)
        self.assertEqual(error_type, ErrorType.EXECUTION_ERROR)
        self.assertGreaterEqual(confidence, HIGH_CONFIDENCE_THRESHOLD)
    
    # Runtime Error Tests
    
    def test_name_error(self):
        """Test classification of name error due to undefined variable."""
        error_data = {
            'exception_type': 'NameError',
            'error_message': "NameError: name 'x' is not defined",
            'traceback': 'File "<string>", line 1, in <module>\nNameError: name "x" is not defined',
            'code_snippet': 'print(x)'
        }
        error_type, confidence = self.classifier.classify_error(error_data)
        self.assertEqual(error_type, ErrorType.EXECUTION_ERROR)
        self.assertGreaterEqual(confidence, MEDIUM_CONFIDENCE_THRESHOLD)
    
    def test_type_error(self):
        """Test classification of type error."""
        error_data = {
            'exception_type': 'TypeError',
            'error_message': "TypeError: can only concatenate str (not 'int') to str",
            'traceback': 'File "<string>", line 1, in <module>\nTypeError: can only concatenate str (not "int") to str',
            'code_snippet': 'print("2" + 2)'
        }
        error_type, confidence = self.classifier.classify_error(error_data)
        self.assertEqual(error_type, ErrorType.EXECUTION_ERROR)
        self.assertGreaterEqual(confidence, MEDIUM_CONFIDENCE_THRESHOLD)
    
    def test_index_error(self):
        """Test classification of index error."""
        error_data = {
            'exception_type': 'IndexError',
            'error_message': 'IndexError: list index out of range',
            'traceback': 'File "<string>", line 1, in <module>\nIndexError: list index out of range',
            'code_snippet': 'a = [1, 2, 3]\nprint(a[10])'
        }
        error_type, confidence = self.classifier.classify_error(error_data)
        self.assertEqual(error_type, ErrorType.EXECUTION_ERROR)
        self.assertGreaterEqual(confidence, MEDIUM_CONFIDENCE_THRESHOLD)
    
    def test_zero_division_error(self):
        """Test classification of zero division error."""
        error_data = {
            'exception_type': 'ZeroDivisionError',
            'error_message': 'ZeroDivisionError: division by zero',
            'traceback': 'File "<string>", line 1, in <module>\nZeroDivisionError: division by zero',
            'code_snippet': 'print(1/0)'
        }
        error_type, confidence = self.classifier.classify_error(error_data)
        self.assertEqual(error_type, ErrorType.EXECUTION_ERROR)
        self.assertGreaterEqual(confidence, MEDIUM_CONFIDENCE_THRESHOLD)
    
    # Logical Error Tests
    
    def test_logical_error_wrong_operation(self):
        """Test classification of logical error due to using wrong operation."""
        error_data = {
            'exception_type': None,
            'error_message': '',
            'code_snippet': 'def add(a, b): return a - b',
            'test_failures': [
                {
                    'name': 'test_add',
                    'inputs': {'a': 1, 'b': 2},
                    'expected_output': 3,
                    'actual_output': -1,
                    'status': 'failed',
                    'error_message': 'Expected: 3, Got: -1'
                }
            ]
        }
        error_type, confidence = self.classifier.classify_error(error_data)
        self.assertEqual(error_type, ErrorType.PLANNING_ERROR)
        self.assertGreaterEqual(confidence, MEDIUM_CONFIDENCE_THRESHOLD)
    
    def test_logical_error_boundary_condition(self):
        """Test classification of logical error due to missed boundary condition."""
        error_data = {
            'exception_type': None,
            'error_message': '',
            'code_snippet': 'def is_positive(n): return n > 0',
            'test_failures': [
                {
                    'name': 'test_is_positive_zero',
                    'inputs': {'n': 0},
                    'expected_output': True,
                    'actual_output': False,
                    'status': 'failed',
                    'error_message': 'Expected: True, Got: False'
                }
            ]
        }
        error_type, confidence = self.classifier.classify_error(error_data)
        self.assertEqual(error_type, ErrorType.PLANNING_ERROR)
        self.assertGreaterEqual(confidence, MEDIUM_CONFIDENCE_THRESHOLD)
    
    def test_logical_error_algorithm_issue(self):
        """Test classification of logical error due to algorithmic issue."""
        error_data = {
            'exception_type': None,
            'error_message': '',
            'code_snippet': 'def is_prime(n):\n    return n % 2 != 0',
            'test_failures': [
                {
                    'name': 'test_is_prime_9',
                    'inputs': {'n': 9},
                    'expected_output': False,
                    'actual_output': True,
                    'status': 'failed',
                    'error_message': 'Expected: False, Got: True'
                }
            ]
        }
        error_type, confidence = self.classifier.classify_error(error_data)
        self.assertEqual(error_type, ErrorType.PLANNING_ERROR)
        self.assertGreaterEqual(confidence, MEDIUM_CONFIDENCE_THRESHOLD)
    
    # Mixed Error Type Tests
    
    def test_mixed_error_syntax_and_logical(self):
        """Test classification when both syntax and logical errors are present."""
        error_data = {
            'exception_type': 'SyntaxError',
            'error_message': 'SyntaxError: invalid syntax',
            'traceback': 'File "<string>", line 1\n def add(a, b) return a - b\n                ^\nSyntaxError: invalid syntax',
            'code_snippet': 'def add(a, b) return a - b',
            'test_failures': [
                {
                    'name': 'test_add',
                    'inputs': {'a': 1, 'b': 2},
                    'expected_output': 3,
                    'actual_output': None,
                    'status': 'error',
                    'error_message': 'SyntaxError'
                }
            ]
        }
        error_type, confidence = self.classifier.classify_error(error_data)
        self.assertEqual(error_type, ErrorType.EXECUTION_ERROR)
        self.assertGreaterEqual(confidence, HIGH_CONFIDENCE_THRESHOLD)
    
    # Edge Cases
    
    def test_ambiguous_error(self):
        """Test classification of ambiguous error with minimal information."""
        error_data = {
            'exception_type': None,
            'error_message': 'Something went wrong',
            'code_snippet': '',
            'test_failures': []
        }
        error_type, confidence = self.classifier.classify_error(error_data)
        self.assertEqual(error_type, ErrorType.AMBIGUOUS_ERROR)
        self.assertLess(confidence, MEDIUM_CONFIDENCE_THRESHOLD)
    
    def test_no_error_data(self):
        """Test classification when no error data is provided."""
        error_data = {}
        error_type, confidence = self.classifier.classify_error(error_data)
        self.assertEqual(error_type, ErrorType.AMBIGUOUS_ERROR)
        self.assertLess(confidence, MEDIUM_CONFIDENCE_THRESHOLD)
    
    # Pattern Recognition Test
    
    def test_pattern_recognition_infinite_recursion(self):
        """Test classification of error pattern indicating infinite recursion."""
        error_data = {
            'exception_type': 'RecursionError',
            'error_message': 'RecursionError: maximum recursion depth exceeded',
            'traceback': 'File "<string>", line 2, in factorial\nFile "<string>", line 2, in factorial\nFile "<string>", line 2, in factorial\nRecursionError: maximum recursion depth exceeded',
            'code_snippet': 'def factorial(n):\n    return n * factorial(n-1)'
        }
        error_type, confidence = self.classifier.classify_error(error_data)
        self.assertEqual(error_type, ErrorType.EXECUTION_ERROR)
        self.assertGreaterEqual(confidence, MEDIUM_CONFIDENCE_THRESHOLD)
    
    def test_pattern_recognition_off_by_one(self):
        """Test classification of off-by-one error pattern."""
        error_data = {
            'exception_type': None,
            'error_message': '',
            'code_snippet': 'def sum_to_n(n):\n    total = 0\n    for i in range(n):\n        total += i\n    return total',
            'test_failures': [
                {
                    'name': 'test_sum_to_5',
                    'inputs': {'n': 5},
                    'expected_output': 15,  # 1+2+3+4+5
                    'actual_output': 10,    # 1+2+3+4
                    'status': 'failed',
                    'error_message': 'Expected: 15, Got: 10'
                }
            ]
        }
        error_type, confidence = self.classifier.classify_error(error_data)
        self.assertEqual(error_type, ErrorType.PLANNING_ERROR)
        self.assertGreaterEqual(confidence, MEDIUM_CONFIDENCE_THRESHOLD)
        
    # Advanced Pattern Tests
    
    def test_advanced_pattern_import_path_issue(self):
        """Test classification of complex import path issue."""
        error_data = {
            'exception_type': 'ModuleNotFoundError',
            'error_message': "ModuleNotFoundError: No module named 'self_healing_agents'",
            'traceback': 'File "<string>", line 1, in <module>\nModuleNotFoundError: No module named "self_healing_agents"',
            'code_snippet': 'from self_healing_agents import Agent'
        }
        error_type, confidence = self.classifier.classify_error(error_data)
        self.assertEqual(error_type, ErrorType.EXECUTION_ERROR)
        self.assertGreaterEqual(confidence, HIGH_CONFIDENCE_THRESHOLD)
    
    def test_advanced_pattern_mismatched_arguments(self):
        """Test classification of function with mismatched arguments."""
        error_data = {
            'exception_type': 'TypeError',
            'error_message': "TypeError: process() takes 2 positional arguments but 3 were given",
            'traceback': 'File "<string>", line 3, in <module>\nTypeError: process() takes 2 positional arguments but 3 were given',
            'code_snippet': 'def process(a, b):\n    return a + b\n\nprocess(1, 2, 3)'
        }
        error_type, confidence = self.classifier.classify_error(error_data)
        self.assertEqual(error_type, ErrorType.EXECUTION_ERROR)
        self.assertGreaterEqual(confidence, HIGH_CONFIDENCE_THRESHOLD)
    
    def test_advanced_pattern_wrong_data_structure(self):
        """Test classification of error due to wrong data structure choice."""
        error_data = {
            'exception_type': None,
            'error_message': '',
            'code_snippet': 'def find_duplicates(nums):\n    return list(set([x for x in nums if nums.count(x) > 1]))',
            'test_failures': [
                {
                    'name': 'test_find_duplicates_performance',
                    'inputs': {'nums': list(range(10000)) + [5]},
                    'expected_output': [],
                    'actual_output': None,
                    'status': 'timeout',
                    'error_message': 'Test timed out - inefficient algorithm detected'
                }
            ]
        }
        error_type, confidence = self.classifier.classify_error(error_data)
        self.assertEqual(error_type, ErrorType.PLANNING_ERROR)
        self.assertGreaterEqual(confidence, MEDIUM_CONFIDENCE_THRESHOLD)

if __name__ == '__main__':
    unittest.main() 