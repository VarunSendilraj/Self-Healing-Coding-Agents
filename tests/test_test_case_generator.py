import sys
import os
import unittest
from unittest.mock import MagicMock, patch
import json

# Ensure the src directory is in the Python path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from self_healing_agents.agents import Critic
from self_healing_agents.llm_service import LLMService, LLMServiceError
from self_healing_agents.classifiers import ErrorClassifier


class TestTestCaseGenerator(unittest.TestCase):
    def setUp(self):
        self.llm_mock = MagicMock(spec=LLMService)
        self.critic = Critic(name="TestCritic", llm_service=self.llm_mock)
        
        # Sample Python code for testing
        self.sample_code = """
def add(a, b):
    return a + b
"""
        
        self.sample_task = "Write a function that adds two numbers and returns the result."
        
        # Expected response from LLM
        self.expected_test_cases_dict = {
            "function_to_test": "add",
            "test_cases": [
                {
                    "test_case_name": "test_positive_numbers",
                    "inputs": {"a": 1, "b": 2},
                    "expected_output": 3
                },
                {
                    "test_case_name": "test_negative_numbers",
                    "inputs": {"a": -1, "b": -2},
                    "expected_output": -3
                },
                {
                    "test_case_name": "test_zero_sum",
                    "inputs": {"a": 1, "b": -1},
                    "expected_output": 0
                }
            ]
        }

    def _create_critic_with_mocked_llm(self) -> Critic:
        # Helper to create Critic with the instance's llm_mock
        # This allows llm_mock to be configured before Critic instantiation if needed.
        return Critic(name="TestCritic", llm_service=self.llm_mock)

    def test_generate_test_cases_success(self):
        """Test that _generate_test_cases correctly processes LLM response for a valid function"""
        # Configure mock to return the dictionary directly, as if expect_json=True succeeded
        self.llm_mock.invoke.return_value = self.expected_test_cases_dict
        critic = self._create_critic_with_mocked_llm()
        
        function_name, test_cases = critic._generate_test_cases(self.sample_task, self.sample_code)
        
        self.assertEqual(function_name, "add")
        self.assertEqual(len(test_cases), 3)
        self.assertEqual(test_cases[0]["test_case_name"], "test_positive_numbers")
        self.assertEqual(test_cases[0]["inputs"], {"a": 1, "b": 2})
        self.assertEqual(test_cases[0]["expected_output"], 3)
        
        self.llm_mock.invoke.assert_called_once()
        call_args_list = self.llm_mock.invoke.call_args_list
        self.assertEqual(len(call_args_list), 1)
        args, kwargs = call_args_list[0]
        self.assertTrue(kwargs.get('expect_json'), "expect_json should be True")
        messages = args[0]
        system_message = next((msg for msg in messages if msg["role"] == "system"), None)
        self.assertIsNotNone(system_message)
        self.assertIn(self.sample_task, system_message["content"])
        self.assertIn(self.sample_code, system_message["content"])

    def test_generate_test_cases_empty_code(self):
        """Test that _generate_test_cases handles empty code gracefully"""
        critic = self._create_critic_with_mocked_llm()
        function_name, test_cases = critic._generate_test_cases(self.sample_task, "")
        
        self.assertIsNone(function_name)
        self.assertEqual(test_cases, [])
        self.llm_mock.invoke.assert_not_called()

    def test_generate_test_cases_llm_fails_json_parsing(self):
        """Test that _generate_test_cases handles LLMServiceError from failed JSON parsing"""
        # Simulate LLMService raising LLMServiceError due to bad JSON from LLM
        self.llm_mock.invoke.side_effect = LLMServiceError("Failed to parse LLM output as JSON")
        critic = self._create_critic_with_mocked_llm()
        
        function_name, test_cases = critic._generate_test_cases(self.sample_task, self.sample_code)
        
        self.assertIsNone(function_name)
        self.assertEqual(test_cases, [])
        self.llm_mock.invoke.assert_called_once() # LLM was called, but parsing failed

    def test_generate_test_cases_llm_returns_non_dict(self):
        """Test graceful handling if LLM invoke (with expect_json=True) somehow returns non-dict/non-str"""
        # This case simulates an unexpected return type from LLMService.invoke despite expect_json=True
        # (e.g. if LLMService had a bug or different behavior)
        self.llm_mock.invoke.return_value = 123 # An integer, neither dict nor str
        critic = self._create_critic_with_mocked_llm()

        function_name, test_cases = critic._generate_test_cases(self.sample_task, self.sample_code)
        self.assertIsNone(function_name)
        self.assertEqual(test_cases, [])
        self.llm_mock.invoke.assert_called_once()

    def test_generate_test_cases_missing_fields(self):
        """Test that _generate_test_cases handles LLM response with missing required fields"""
        incomplete_response_dict = {
            "test_cases": [
                {
                    "test_case_name": "test_positive_numbers",
                    "inputs": {"a": 1, "b": 2},
                    "expected_output": 3
                }
            ]
        }
        self.llm_mock.invoke.return_value = incomplete_response_dict
        critic = self._create_critic_with_mocked_llm()
        
        function_name, test_cases = critic._generate_test_cases(self.sample_task, self.sample_code)
        
        self.assertIsNone(function_name) # function_to_test is missing
        self.assertEqual(len(test_cases), 1)
        self.assertEqual(test_cases[0]['test_case_name'], 'test_positive_numbers')
        self.llm_mock.invoke.assert_called_once()

    def test_llm_call_format(self):
        """Test that the LLM is called with the correct format for test generation"""
        self.llm_mock.invoke.return_value = self.expected_test_cases_dict
        critic = self._create_critic_with_mocked_llm()
        
        critic._generate_test_cases(self.sample_task, self.sample_code)
        
        self.llm_mock.invoke.assert_called_once()
        args, kwargs = self.llm_mock.invoke.call_args
        
        messages = args[0]
        self.assertTrue(len(messages) > 0)
        system_message = next((msg for msg in messages if msg["role"] == "system"), None)
        self.assertIsNotNone(system_message)
        self.assertIn(self.sample_task, system_message["content"])
        self.assertIn(self.sample_code, system_message["content"])
        
        self.assertTrue(kwargs.get('expect_json', False), "expect_json should be True")

if __name__ == '__main__':
    unittest.main() 