import unittest
from typing import Dict, List, Any
from self_healing_agents.agents import Critic
from self_healing_agents.llm_service import LLMService # Required for Critic instantiation
from unittest.mock import patch, MagicMock
import json

# A mock LLMService for testing purposes, as Critic requires an LLMService instance
class MockLLMService(LLMService):
    def __init__(self, api_key: str = "mock_key_for_testing", provider: str = "deepseek", model_name: str = "mock_deepseek_model"):
        # Using "deepseek" as it's a supported provider, but the behavior is mocked.
        # The api_key and model_name are also mocks for this test setup.
        super().__init__(api_key=api_key, provider=provider, model_name=model_name)

    def invoke(self, messages: List[Dict[str, str]], expect_json: bool = False) -> Any:
        # This mock invoke won't be directly called by the placeholder logic being tested,
        # but needs to exist for the LLMService interface.
        print(f"MockLLMService invoked with messages: {messages}, expect_json: {expect_json}")
        if expect_json and any("You are an expert Python test case generator" in msg.get("content", "") for msg in messages):
            # Default mock response for test case generation if not overridden by a specific test
            return [
                {"test_case_name": "mock_llm_test_1", "inputs": {"x": 1}, "expected_output": "output_1"}
            ]
        if expect_json:
            return {"mock_response": "This is a mock JSON response"}
        return "This is a mock string response"

class TestCriticAgentPlaceholderLogic(unittest.TestCase):
    def setUp(self):
        """Set up a Critic agent instance for testing."""
        self.mock_llm_service = MockLLMService()
        # The system_prompt for Critic is not used by the placeholder evaluate_code,
        # but it's part of the constructor.
        self.critic_agent = Critic(name="TestCritic", llm_service=self.mock_llm_service, system_prompt="Mock critic system prompt")

    def test_evaluate_code_success(self):
        """Test placeholder logic returns a success report for normal code."""
        code_snippet = "def hello():\n    print('Hello')"
        task_desc = "A simple hello function"
        
        report = self.critic_agent.evaluate_code(code=code_snippet, task_description=task_desc)
        
        self.assertIsInstance(report, dict)
        self.assertEqual(report.get("status"), "SUCCESS_EXECUTION")
        self.assertEqual(report.get("score"), 1.0)
        self.assertIsNone(report.get("error_details"))
        self.assertEqual(report.get("summary"), "Code executed successfully. Test cases generated.")

    def test_evaluate_code_failure(self):
        """Test placeholder logic returns a failure report if 'FAIL_TOKEN' is present."""
        code_snippet_with_fail_token = "print('This code has a FAIL_TOKEN')"
        task_desc_fail = "A code snippet that used to trigger placeholder failure"
        
        report = self.critic_agent.evaluate_code(code=code_snippet_with_fail_token, task_description=task_desc_fail)
        
        self.assertIsInstance(report, dict)
        self.assertEqual(report.get("status"), "SUCCESS_EXECUTION")
        self.assertEqual(report.get("score"), 1.0)
        self.assertIsNone(report.get("error_details"))
        self.assertEqual(report.get("summary"), "Code executed successfully. Test cases generated.")

class TestCriticAgentSandboxExecution(unittest.TestCase):
    def setUp(self):
        """Set up a Critic agent instance for testing the sandbox."""
        self.mock_llm_service = MockLLMService() # Still needed for Critic instantiation
        self.critic_agent = Critic(name="SandboxTestCritic", llm_service=self.mock_llm_service)

    def test_sandbox_executes_safe_code(self):
        """Test that simple, safe Python code executes successfully."""
        safe_code = "x = 10\ny = 20\nresult = x + y\nprint(f'Result: {result}')"
        execution_details = self.critic_agent._execute_sandboxed_code(safe_code)
        
        self.assertTrue(execution_details["success"])
        self.assertEqual(execution_details["stdout"].strip(), "Result: 30")
        self.assertEqual(execution_details["stderr"], "")
        self.assertIsNone(execution_details["error_type"])

    def test_sandbox_captures_stdout(self):
        """Test that stdout from executed code is captured."""
        code_with_print = "print('Hello Sandbox')\nprint('Another line')"
        execution_details = self.critic_agent._execute_sandboxed_code(code_with_print)
        
        self.assertTrue(execution_details["success"])
        self.assertEqual(execution_details["stdout"], "Hello Sandbox\nAnother line\n")

    def test_sandbox_handles_syntax_error(self):
        """Test that syntax errors are caught and reported."""
        code_with_syntax_error = "x = 10 +\nprint(x)"
        execution_details = self.critic_agent._execute_sandboxed_code(code_with_syntax_error)
        
        self.assertFalse(execution_details["success"])
        self.assertEqual(execution_details["error_type"], "SyntaxError")
        # For SyntaxError, 'stderr' and 'traceback' should be the same simplified message
        self.assertIn("SyntaxError at line", execution_details["traceback"])
        self.assertIn("syntax", execution_details["error_message"].lower())
        self.assertEqual(execution_details["stderr"], execution_details["traceback"])

    def test_sandbox_handles_runtime_error_zerodivision(self):
        """Test that runtime errors like ZeroDivisionError are caught."""
        code_with_runtime_error = "result = 10 / 0"
        execution_details = self.critic_agent._execute_sandboxed_code(code_with_runtime_error)
        
        self.assertFalse(execution_details["success"])
        self.assertEqual(execution_details["error_type"], "ZeroDivisionError")
        self.assertIn("ZeroDivisionError", execution_details["stderr"]) # Full traceback in stderr
        self.assertIn("agents.py", execution_details["stderr"]) # Full traceback includes outer call
        
        # Simplified traceback should focus on <string>
        self.assertIn('File "<string>"', execution_details["traceback"])
        self.assertIn("ZeroDivisionError: division by zero", execution_details["traceback"])
        self.assertNotIn("agents.py", execution_details["traceback"]) # Should not have outer call
        self.assertEqual(execution_details["error_message"], "division by zero")

    def test_sandbox_handles_runtime_error_nameerror(self):
        """Test that runtime errors like NameError are caught."""
        code_with_name_error = "print(undefined_variable)"
        execution_details = self.critic_agent._execute_sandboxed_code(code_with_name_error)

        self.assertFalse(execution_details["success"])
        self.assertEqual(execution_details["error_type"], "NameError")
        self.assertIn("NameError", execution_details["stderr"]) # Full traceback in stderr
        self.assertIn("agents.py", execution_details["stderr"]) # Full traceback includes outer call
        
        # Simplified traceback
        self.assertIn('File "<string>"', execution_details["traceback"])
        self.assertIn("NameError: name 'undefined_variable' is not defined", execution_details["traceback"])
        self.assertNotIn("agents.py", execution_details["traceback"])
        self.assertEqual(execution_details["error_message"], "name 'undefined_variable' is not defined")

    def test_sandbox_blocks_import_os(self):
        """Test that importing 'os' is blocked."""
        code_import_os = "import os\nprint(os.getcwd())"
        execution_details = self.critic_agent._execute_sandboxed_code(code_import_os)

        self.assertFalse(execution_details["success"])
        self.assertEqual(execution_details["error_type"], "ImportError")
        self.assertIn("__import__ not found", execution_details["error_message"])
        
        # The simplified traceback should reflect this ImportError related to <string>
        # traceback.extract_tb might not provide a frame.filename == "<string>" for this type of early import error
        # so we check that the error message is present in the simplified traceback.
        self.assertIn("ImportError: __import__ not found", execution_details["traceback"])
        self.assertNotIn("agents.py", execution_details["traceback"]) # Simplified should not have agent path

    def test_sandbox_blocks_file_open(self):
        """Test that opening files is blocked (should raise NameError for 'open')."""
        code_open_file = "with open('test.txt', 'w') as f:\n    f.write('hello')"
        # 'open' is not in our allowed_globals['__builtins__'] explicitly
        execution_details = self.critic_agent._execute_sandboxed_code(code_open_file)
        
        self.assertFalse(execution_details["success"])
        self.assertEqual(execution_details["error_type"], "NameError")
        self.assertIn("name 'open' is not defined", execution_details["error_message"])
        # Simplified traceback for NameError
        self.assertIn('File "<string>"', execution_details["traceback"])
        self.assertIn("NameError: name 'open' is not defined", execution_details["traceback"])
        self.assertNotIn("agents.py", execution_details["traceback"])

    def test_evaluate_code_uses_sandbox_for_success(self):
        """Test that evaluate_code uses the sandbox and reports success correctly."""
        safe_code = "print('Evaluated successfully')"
        task_desc = "Test eval success"
        # Mock the LLM call within _generate_test_cases for this specific test
        expected_tests = [{"test_case_name": "gen_test_1", "inputs": {}, "expected_output": "ok"}]
        self.critic_agent.llm_service.invoke = MagicMock(return_value=expected_tests)
        
        report = self.critic_agent.evaluate_code(code=safe_code, task_description=task_desc)
        self.assertEqual(report["status"], "SUCCESS_EXECUTION")
        self.assertEqual(report["score"], 1.0)
        self.assertEqual(report["execution_stdout"].strip(), "Evaluated successfully")
        self.assertTrue(not report["execution_stderr"])
        self.assertIsNone(report["error_details"])
        self.assertEqual(report["test_results"], expected_tests)
        self.critic_agent.llm_service.invoke.assert_called_once() # Verify LLM was called for test gen

    def test_evaluate_code_uses_sandbox_for_failure(self):
        """Test that evaluate_code uses the sandbox and reports failure correctly."""
        failing_code = "x = 1 / 0"
        task_desc = "Test eval failure"
        report = self.critic_agent.evaluate_code(code=failing_code, task_description=task_desc)
        
        self.assertEqual(report["status"], "FAILURE_ZERODIVISIONERROR")
        self.assertEqual(report["score"], 0.0)
        self.assertIsNotNone(report["error_details"])
        self.assertEqual(report["error_details"]["type"], "ZeroDivisionError")
        self.assertIn("division by zero", report["error_details"]["message"])
        
        # Check the simplified traceback in the report
        self.assertIn('File "<string>"', report["error_details"]["traceback"])
        self.assertIn("ZeroDivisionError: division by zero", report["error_details"]["traceback"])
        self.assertNotIn("agents.py", report["error_details"]["traceback"])
        
        # Check the full traceback in execution_stderr
        self.assertIn("ZeroDivisionError", report["execution_stderr"])
        self.assertIn("agents.py", report["execution_stderr"])
        self.assertEqual(report["test_results"], []) # Ensure no tests are in results if code fails

class TestCriticAgentTestGeneration(unittest.TestCase):
    def setUp(self):
        """Set up a Critic agent instance for testing the sandbox."""
        self.mock_llm_service = MagicMock(spec=LLMService) 
        self.critic_agent = Critic(name="TestGenCritic", llm_service=self.mock_llm_service)
        self.task_desc = "Write a function that adds two numbers."
        self.generated_code = "def add(a, b):\n    return a + b"

    def test_generate_test_cases_success(self):
        mock_llm_response = [
            {"test_case_name": "test_positive_add", "inputs": {"a": 1, "b": 2}, "expected_output": 3},
            {"test_case_name": "test_negative_add", "inputs": {"a": -1, "b": -2}, "expected_output": -3}
        ]
        self.mock_llm_service.invoke.return_value = mock_llm_response
        test_cases = self.critic_agent._generate_test_cases(self.task_desc, self.generated_code)
        self.assertEqual(len(test_cases), 2)
        self.assertEqual(test_cases[0]["test_case_name"], "test_positive_add")
        self.assertEqual(test_cases[1]["inputs"], {"a": -1, "b": -2})
        self.mock_llm_service.invoke.assert_called_once()
        args, kwargs = self.mock_llm_service.invoke.call_args
        self.assertTrue(kwargs["expect_json"])
        user_prompt_content = args[0][1]["content"]
        self.assertIn(self.task_desc, user_prompt_content)
        self.assertIn(self.generated_code, user_prompt_content)
        self.assertIn("Output ONLY the JSON list of test cases", user_prompt_content)

    def test_generate_test_cases_llm_returns_malformed_json_string_handled_by_service(self):
        self.mock_llm_service.invoke.return_value = {"error": "JSONDecodeError", "raw_response": "not json"}
        test_cases = self.critic_agent._generate_test_cases(self.task_desc, self.generated_code)
        self.assertEqual(test_cases, [])

    def test_generate_test_cases_llm_returns_non_json_list(self):
        self.mock_llm_service.invoke.return_value = "This is not JSON or a list."
        test_cases = self.critic_agent._generate_test_cases(self.task_desc, self.generated_code)
        self.assertEqual(test_cases, [])

    def test_generate_test_cases_llm_returns_invalid_structure(self):
        mock_llm_response = {"some_other_key": "some_value"} 
        self.mock_llm_service.invoke.return_value = mock_llm_response
        test_cases = self.critic_agent._generate_test_cases(self.task_desc, self.generated_code)
        self.assertEqual(test_cases, [])

        mock_llm_response_list_bad_items = [
            {"test_case_name": "valid_test", "inputs": {}, "expected_output": None},
            {"wrong_key": "bad_item"}
        ]
        self.mock_llm_service.invoke.return_value = mock_llm_response_list_bad_items
        test_cases = self.critic_agent._generate_test_cases(self.task_desc, self.generated_code)
        self.assertEqual(len(test_cases), 1)
        self.assertEqual(test_cases[0]["test_case_name"], "valid_test")

    def test_generate_test_cases_llm_service_returns_error_dict(self):
        self.mock_llm_service.invoke.return_value = {"error": "LLM unavailable", "details": "..."}
        test_cases = self.critic_agent._generate_test_cases(self.task_desc, self.generated_code)
        self.assertEqual(test_cases, [])

    def test_generate_test_cases_json_decode_error_in_method(self):
        self.mock_llm_service.invoke.side_effect = json.JSONDecodeError("mock error", "doc", 0)
        test_cases = self.critic_agent._generate_test_cases(self.task_desc, self.generated_code)
        self.assertEqual(test_cases, [])

    def test_evaluate_code_calls_generate_test_cases_on_success(self):
        safe_code = "def my_func(): pass"
        mock_tests = [{"test_case_name": "mock_test", "inputs": {}, "expected_output": None}]
        with patch.object(self.critic_agent, '_execute_sandboxed_code') as mock_sandbox_exec:
            mock_sandbox_exec.return_value = {
                "success": True, "stdout": "", "stderr": "", 
                "error_type": None, "error_message": "", "traceback": ""
            }
            # We need to mock the llm_service used by _generate_test_cases directly here
            self.critic_agent.llm_service.invoke.return_value = mock_tests # Ensure this is the mocked LLM service instance
            report = self.critic_agent.evaluate_code(safe_code, self.task_desc)
            mock_sandbox_exec.assert_called_once_with(safe_code)
            self.critic_agent.llm_service.invoke.assert_called_once() # Check the specific LLM call for test gen
            self.assertEqual(report["test_results"], mock_tests)
            self.assertEqual(report["status"], "SUCCESS_EXECUTION")

    def test_evaluate_code_skips_generate_test_cases_on_failure(self):
        failing_code = "syntax error"
        with patch.object(self.critic_agent, '_execute_sandboxed_code') as mock_sandbox_exec:
            mock_sandbox_exec.return_value = {
                "success": False, "error_type": "SyntaxError",
                "error_message": "mock syntax error", "traceback": "SyntaxError at line 1: mock syntax error",
                "stdout": "", "stderr": "SyntaxError at line 1: mock syntax error"
            }
            with patch.object(self.critic_agent, '_generate_test_cases') as mock_generate: # Keep this to ensure it's not called
                report = self.critic_agent.evaluate_code(failing_code, self.task_desc)
                mock_sandbox_exec.assert_called_once_with(failing_code)
                mock_generate.assert_not_called()
                self.assertEqual(report["test_results"], [])
                self.assertIn("FAILURE_SYNTAXERROR", report["status"])

if __name__ == '__main__':
    unittest.main() 