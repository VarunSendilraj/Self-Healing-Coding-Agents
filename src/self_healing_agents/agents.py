from typing import Dict, List, Any, Optional, Tuple
import json
from .llm_service import LLMService
from .prompts import PLANNER_SYSTEM_PROMPT, EXECUTOR_SYSTEM_PROMPT_V1, CRITIC_SYSTEM_PROMPT, CRITIC_TEST_GENERATION_SYSTEM_PROMPT
from abc import ABC, abstractmethod
import io # Added for capturing stdout
import sys # Added for capturing stderr (though exec captures it directly)
import traceback # Added for formatting exceptions
import subprocess
import tempfile
import os
import logging
import importlib.util
import inspect
import dataclasses # Add this import

from self_healing_agents.schemas import (
    PlannerOutput,
    ExecutorOutput,
    CriticReport, # Base report
    EnhancedCriticReport, # New enhanced report
    TaskDefinition
)
from self_healing_agents.error_types import ErrorType, AgentType # New enums
from self_healing_agents.classifiers import ErrorClassifier # Classifier interface and base
from self_healing_agents.classifiers.rule_based import RuleBasedErrorClassifier

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define constants for overall status
STATUS_CRITICAL_SYNTAX_ERROR = "CRITICAL_SYNTAX_ERROR"
STATUS_CRITICAL_RUNTIME_ERROR = "CRITICAL_RUNTIME_ERROR"
STATUS_LOGICAL_ERROR = "LOGICAL_ERROR"
STATUS_SUCCESS = "SUCCESS"
NO_TESTS_FOUND = "NO_TESTS_FOUND" # If no tests are run, but code executes.

class Agent(ABC):
    def __init__(self, name: str, llm_service: LLMService):
        self.name = name
        self.llm_service = llm_service

    @abstractmethod
    def run(self, *args, **kwargs) -> Any:
        pass

class Planner(Agent):
    def __init__(self, name: str, llm_service: LLMService, system_prompt: str = PLANNER_SYSTEM_PROMPT):
        super().__init__(name, llm_service)
        self.system_prompt = system_prompt

    def run(self, user_request: str) -> Dict:
        """
        Generates a plan based on the user request.
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_request}
        ]
        try:
            # Assuming the PLANNER_SYSTEM_PROMPT guides the LLM to produce a structured plan (e.g., JSON)
            # If not, expect_json might need to be False or the prompt refined.
            plan = self.llm_service.invoke(messages, expect_json=True)
            if not isinstance(plan, Dict):
                # Fallback if JSON parsing failed or LLM didn't comply
                print(f"Warning: PlannerAgent received non-dict plan: {plan}")
                return {"error": "Plan was not in the expected dictionary format.", "raw_response": str(plan)}
            return plan
        except Exception as e:
            print(f"Error in PlannerAgent: {e}")
            return {"error": f"Could not generate plan due to: {e}"}

class Executor(Agent):
    def __init__(self, name: str, llm_service: LLMService, system_prompt: str = EXECUTOR_SYSTEM_PROMPT_V1):
        super().__init__(name, llm_service)
        self.system_prompt = system_prompt

    def set_prompt(self, new_prompt: str):
        """
        Updates the system prompt for the Executor agent.
        """
        self.system_prompt = new_prompt
        # Optionally, log this change if your agents have their own logging
        print(f"Executor '{self.name}': System prompt updated to: '{new_prompt[:70]}...'")

    def run(self, plan: Dict, original_request: str) -> str:
        """
        Generates Python code based on the plan from the Planner and the original request.
        """
        # Construct a detailed prompt for the Executor
        # It's often helpful to include the original request for context, even if the plan is detailed.
        executor_prompt_content = (
            f"Original User Request:\\n{original_request}\\n\\n"
            f"Execution Plan:\\n{plan.get('steps', 'No specific steps provided in plan.')}\\n\\n"
            f"Please generate Python code to accomplish this. Ensure you only output the raw Python code, without any markdown formatting or explanations."
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": executor_prompt_content}
        ]
        try:
            # Code generation typically results in a string of code
            code_output = self.llm_service.invoke(messages, expect_json=False)
            return code_output
        except Exception as e:
            print(f"Error in ExecutorAgent: {e}")
            return f"# Error generating code: {e}"

class Critic(Agent):
    def __init__(self, name: str, llm_service: LLMService, system_prompt: str = CRITIC_SYSTEM_PROMPT):
        super().__init__(name, llm_service)
        self.system_prompt = system_prompt # system_prompt is now an instance variable
        self.test_generation_system_prompt = CRITIC_TEST_GENERATION_SYSTEM_PROMPT # Store the new prompt
        # Always use RuleBasedErrorClassifier for now, regardless of llm_service presence.
        # This can be updated if an LLM-based classifier is introduced.
        self.error_classifier = RuleBasedErrorClassifier()

    def _execute_sandboxed_code(self, code_string: str) -> Dict[str, Any]:
        """
        Executes the provided Python code string in a sandboxed environment (code env).
        Captures stdout, stderr, and any exceptions.
        
        Returns:
            Dict containing:
            - stdout: Captured standard output
            - stderr: Captured standard error or formatted exception
            - error_type: Type of error if any (None if successful)
            - error_message: Error message if any (empty if successful)
            - traceback: Formatted traceback if error (empty if successful)
            - success: Boolean indicating if execution succeeded
        """
        # --- Start of Addition: Strip markdown fences ---
        processed_code_string = code_string.strip()
        if processed_code_string.startswith("```python") and processed_code_string.endswith("```"):
            processed_code_string = processed_code_string[len("```python"):-(len("```"))].strip()
        elif processed_code_string.startswith("```") and processed_code_string.endswith("```"):
            # General case for markdown fence without language specifier
            processed_code_string = processed_code_string[len("```"):-(len("```"))].strip()
        # --- End of Addition ---

        # Prepare a dictionary of allowed globals.
        # This is crucial for sandboxing. Only explicitly allow what's needed.
        allowed_globals = {
            "__builtins__": {
                "print": print,
                "len": len,
                "range": range,
                "list": list,
                "dict": dict,
                "str": str,
                "int": int,
                "float": float,
                "bool": bool,
                "True": True,
                "False": False,
                "None": None,
                # Common and relatively safe exceptions:
                "Exception": Exception,
                "ValueError": ValueError,
                "TypeError": TypeError,
                "IndexError": IndexError,
                "KeyError": KeyError,
                "ZeroDivisionError": ZeroDivisionError,
                "AttributeError": AttributeError,
                "NameError": NameError,
                "__import__": __import__,
                "__build_class__": __build_class__,
                "repr": repr,
                "isinstance": isinstance,
            },
            "__build_class__": __build_class__,
            # Add current package modules to the sandbox if needed
            "self_healing_agents": self_healing_agents if 'self_healing_agents' in sys.modules else None,
        }
        
        # Add module-level attributes
        allowed_globals["__name__"] = "__main__"
        
        # Configure Python module import path 
        original_path = list(sys.path)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        src_dir = os.path.dirname(current_dir)
        
        # Add src directory to path to resolve imports
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        
        captured_stdout = io.StringIO()
        
        execution_result = {
            "stdout": "",
            "stderr": "",
            "error_type": None,
            "error_message": "",
            "traceback": "",
            "success": False,
            "raw_exception": None  # Store the actual exception object
        }

        # Redirect stdout
        original_stdout = sys.stdout
        sys.stdout = captured_stdout
        
        try:
            # Execute the code with restricted globals and locals
            exec(processed_code_string, allowed_globals, {})
            execution_result["success"] = True
        except SyntaxError as e:
            execution_result["error_type"] = "SyntaxError"
            execution_result["error_message"] = str(e)
            execution_result["raw_exception"] = e
            # This format is already quite simplified and directly useful.
            simplified_traceback = f"SyntaxError at line {e.lineno}, offset {e.offset}: {e.msg}"
            execution_result["traceback"] = simplified_traceback
        except Exception as e:
            # Handle any other exception
            execution_result["error_type"] = e.__class__.__name__
            execution_result["error_message"] = str(e)
            execution_result["raw_exception"] = e
            
            # Get traceback but skip the exec call frame from the traceback
            # as it's not relevant to the executed code
            # This provides a cleaner traceback focused on the executed code
            tb = traceback.format_exc()
            execution_result["traceback"] = tb
        finally:
            # Restore stdout
            sys.stdout = original_stdout
            execution_result["stdout"] = captured_stdout.getvalue()
            
            # Reset the sys.path to avoid side effects
            sys.path = original_path
        
        return execution_result

    def _execute_test_cases(self, code_string: str, test_case_strings: List[str]) -> List[Dict[str, Any]]:
        """
        Executes a list of test case strings against the provided code string.

        Args:
            code_string: The Python code (e.g., function definitions) to test.
            test_case_strings: A list of strings, where each string is an executable
                               Python statement, typically an assertion (e.g., "assert func(1) == 2").

        Returns:
            A list of dictionaries, each containing:
            - test_case_str: The original test case string.
            - passed: Boolean indicating if the test passed.
            - stdout: Captured standard output during the test.
            - stderr: Captured standard error during the test (empty if passed and no other output).
        """
        results = []
        if not test_case_strings:
            logger.info("No test case strings provided to _execute_test_cases.")
            return results

        for test_str in test_case_strings:
            # Combine the original code with the current test case string
            # This ensures any functions defined in code_string are available to the test_str
            full_script_to_run = f"{code_string}\n{test_str}"
            
            logger.debug(f"Executing test case: {test_str} with combined script.")
            execution_details = self._execute_sandboxed_code(full_script_to_run)
            
            passed = execution_details["success"] and not execution_details["stderr"] # An assertion error would put info in stderr
            
            # If an assertion fails, it raises an AssertionError, which _execute_sandboxed_code captures.
            # So, 'success' might be False if an assertion fails.
            # A clearer check for assertion-based tests: if stderr is empty after exec, it implies assertion passed or no error.
            # However, _execute_sandboxed_code sets success=False on any exception.
            # If it's an AssertionError, it's a test failure. Any other error is also a test failure.
            # So, execution_details["success"] being True is a good indicator of a pass here,
            # assuming test_str is an assertion. If stderr is present despite success=True,
            # it might be print statements, which are fine.
            # Let's refine 'passed': A test fails if an error occurs (stderr is populated by _execute_sandboxed_code for exceptions).
            
            passed = not execution_details["stderr"] # Simpler: if stderr is not empty, it's a fail (either runtime error or assertion error)

            results.append({
                "test_case_str": test_str,
                "passed": passed,
                "stdout": execution_details["stdout"],
                "stderr": execution_details["stderr"],
                "raw_result": execution_details # For more detailed analysis later if needed
            })
            logger.debug(f"Test case result: {'Passed' if passed else 'Failed'}. Stderr: {execution_details['stderr']}")
        
        return results

    def _generate_test_cases(self, task_description: str, generated_code: str) -> Tuple[Optional[str], List[str]]:
        """
        Generates test cases for the given code and task description using an LLM call.
        
        Args:
            task_description: The original task description
            generated_code: The Python code to generate tests for
            
        Returns:
            A tuple containing:
            - function_name: The name of the primary function to test (or None if not identifiable)
            - test_cases: A list of test case strings
        """
        if not generated_code or not generated_code.strip():
            logging.warning("Cannot generate test cases for empty code")
            return None, []
        
        # Construct the prompt for the LLM
        # Assuming CRITIC_TEST_GENERATION_SYSTEM_PROMPT guides the LLM
        # to produce a list of assertion strings.
        prompt_content = (
            f"Original Task Description:\n{task_description}\n\n"
            f"Generated Python Code:\n```python\n{generated_code}\n```\n\n"
            f"Please generate a list of Python assertion strings that can be used to test the correctness "
            f"of the 'solution' function (or the primary function if named differently) in the generated code. "
            f"Focus on validating key aspects of the task description. "
            f"Output ONLY a Python list of strings, where each string is a valid assertion. "
            f"Example: ['assert solution(1, 2) == 3', 'assert solution(\"test\") == \"TEST_PASSED\"']"
        )
        
        messages = [
            {"role": "system", "content": self.test_generation_system_prompt}, # Use the dedicated prompt
            {"role": "user", "content": prompt_content}
        ]
        
        try:
            # We expect the LLM to return a string representation of a list of strings.
            # E.g., "['assert func(1)==1', 'assert func(2)==2']"
            raw_llm_output = self.llm_service.invoke(messages, expect_json=False) # Get as string
            
            if not raw_llm_output or not raw_llm_output.strip().startswith('[') or not raw_llm_output.strip().endswith(']'):
                logger.warning(f"LLM test case generation did not return a valid list-like string: {raw_llm_output}")
                # Attempt to wrap in list if it's a single assertion string not in a list
                if isinstance(raw_llm_output, str) and raw_llm_output.strip().startswith("assert"):
                    logger.info("Attempting to wrap single assertion string in a list.")
                    raw_llm_output = f"[{repr(raw_llm_output.strip())}]" # repr to handle quotes correctly
                else:
                    return None, []


            # Attempt to parse the string into a Python list of strings
            # Using ast.literal_eval for safety over eval()
            import ast
            try:
                parsed_test_cases = ast.literal_eval(raw_llm_output.strip())
                if not isinstance(parsed_test_cases, list) or not all(isinstance(tc, str) for tc in parsed_test_cases):
                    logger.warning(f"LLM test case generation parsed, but not a list of strings: {parsed_test_cases}")
                    return None, []
                
                # For now, we assume the LLM correctly identifies/uses the function name like 'solution'
                # or the test assertions are self-contained.
                # A more advanced version might try to parse the function name from `generated_code`.
                # For this MVP, we'll assume the generated assertions are directly executable with the code.
                logger.info(f"Successfully generated and parsed {len(parsed_test_cases)} test cases.")
                return "solution", parsed_test_cases # Assuming 'solution' or handled by assertion string

            except (ValueError, SyntaxError) as e:
                logger.error(f"Error parsing LLM-generated test cases string '{raw_llm_output}': {e}")
                return None, []

        except Exception as e:
            logger.error(f"Error during LLM call for test case generation: {e}")
            return None, []

    def evaluate_code(self, code: str, task_description: str) -> Dict[str, Any]:
        """
        Evaluates the given Python code.
        1. Executes the code to check for syntax and basic runtime errors.
        2. Generates test cases based on the task description and code.
        3. Executes the generated test cases against the code.
        4. Classifies errors and provides a structured report.
        """
        logger.info(f"Critic evaluating code for task: {task_description[:100]}...")
        
        # Initialize parts of the report
        raw_execution_details: Optional[Dict[str, Any]] = None
        test_generation_results: Tuple[Optional[str], List[str]] = (None, [])
        test_execution_results: List[Dict[str, Any]] = [] # Initialize here
        error_classification: Optional[Dict[str, Any]] = None
        overall_status: str = STATUS_CRITICAL_RUNTIME_ERROR # Default to an error state
        score: float = 0.0
        feedback: str = "Evaluation incomplete."

        # Step 1: Initial code execution for syntax and immediate runtime errors
        logger.debug("Step 1: Initial code execution...")
        raw_execution_details = self._execute_sandboxed_code(code)
        
        if not raw_execution_details["success"]:
            logger.warning(f"Initial code execution failed. Error type: {raw_execution_details.get('error_type')}")
            # Error already captured, proceed to classification
            error_classification = self.error_classifier.classify(
                raw_execution_details.get("raw_exception"), # Pass the actual exception object
                raw_execution_details.get("traceback", ""),
                raw_execution_details.get("error_type", "UnknownError"),
                AgentType.EXECUTOR # Error occurred in Executor's code
            )
            overall_status = STATUS_CRITICAL_RUNTIME_ERROR if raw_execution_details.get('error_type') != "SyntaxError" else STATUS_CRITICAL_SYNTAX_ERROR
            feedback = f"Code execution failed: {raw_execution_details.get('error_message', 'Unknown error')}"
            # Score will be calculated later based on this status
        else:
            logger.info("Initial code execution successful.")
            # Step 2: Generate test cases only if initial execution is successful
            logger.debug("Step 2: Generating test cases...")
            # Assuming 'code' is the string of the generated Python code
            function_name, test_case_strings = self._generate_test_cases(task_description, code)
            test_generation_results = (function_name, test_case_strings)

            if not test_case_strings:
                logger.warning("No test cases were generated by the LLM.")
                overall_status = NO_TESTS_FOUND # Or could be SUCCESS if code ran but no tests made
                feedback = "Code executed successfully, but no test cases were generated to verify logic."
                # Score might be moderate in this case
            else:
                logger.info(f"Successfully generated {len(test_case_strings)} test cases for function '{function_name or 'unknown'}'.")
                # Step 3: Execute generated test cases
                logger.debug("Step 3: Executing generated test cases...")
                test_execution_results = self._execute_test_cases(code, test_case_strings)
                
                num_tests_total = len(test_execution_results)
                num_tests_passed = sum(1 for res in test_execution_results if res["passed"])
                
                if num_tests_passed == num_tests_total:
                    overall_status = STATUS_SUCCESS
                    feedback = f"All {num_tests_total} generated test(s) passed."
                    logger.info(feedback)
                else:
                    overall_status = STATUS_LOGICAL_ERROR
                    failed_tests_summary = [
                        f"Test: '{res['test_case_str']}' Failed. Stderr: {res['stderr']}" 
                        for res in test_execution_results if not res["passed"]
                    ]
                    feedback = f"{num_tests_passed}/{num_tests_total} test(s) passed. Failures:\n" + "\n".join(failed_tests_summary)
                    logger.warning(feedback)
                    # Find the first failed test's execution details for error classification
                    # This assumes the error in the test is representative.
                    first_failed_test_raw_result = next((res["raw_result"] for res in test_execution_results if not res["passed"]), None)
                    if first_failed_test_raw_result and not first_failed_test_raw_result.get("success"):
                         error_classification = self.error_classifier.classify(
                            first_failed_test_raw_result.get("raw_exception"),
                            first_failed_test_raw_result.get("traceback", ""),
                            first_failed_test_raw_result.get("error_type", "AssertionError"), # Default to AssertionError if it's a logical fail
                            AgentType.EXECUTOR # Error is in executor's code logic
                        )


        # Calculate score (placeholder for now, will be refined in Task 2.6)
        # For now, a simple score based on status and test pass rate
        # This will be replaced by _calculate_score method call later
        # Placeholder score calculation:
        num_total_from_exec = len(test_execution_results) if test_execution_results else 0
        num_passed_from_exec = sum(1 for res in test_execution_results if res["passed"]) if test_execution_results else 0

        if overall_status == STATUS_SUCCESS:
            score = 1.0
        elif overall_status == STATUS_LOGICAL_ERROR:
            score = 0.2 + 0.5 * (num_passed_from_exec / num_total_from_exec if num_total_from_exec > 0 else 0)
        elif overall_status == NO_TESTS_FOUND and raw_execution_details and raw_execution_details["success"]:
            score = 0.4 # Code ran, but no tests to confirm logic
        elif overall_status == STATUS_CRITICAL_RUNTIME_ERROR:
            score = 0.1
        elif overall_status == STATUS_CRITICAL_SYNTAX_ERROR:
            score = 0.0
        else:
            score = 0.0 # Default for unhandled status


        # Construct the final report
        # Using EnhancedCriticReport structure
        report = EnhancedCriticReport(
            evaluator_name=self.name,
            task_description=task_description,
            generated_code=code,
            raw_execution_details=raw_execution_details if raw_execution_details else {},
            # test_generation_details needs to be structured. For now, a summary.
            test_generation_details={
                "function_name": test_generation_results[0] if test_generation_results else None,
                "generated_test_case_strings": test_generation_results[1] if test_generation_results else [],
                "status": "Generated" if test_generation_results and test_generation_results[1] else "Not Generated/Failed"
            },
            test_execution_results=test_execution_results if test_execution_results else [],
            error_classification=error_classification if error_classification else {},
            overall_status=overall_status,
            feedback=feedback,
            score=score # Will be refined by Task 2.6
        )
        
        logger.info(f"Critic evaluation complete. Overall Status: {overall_status}, Score: {score:.2f}")
        # Return as dict as per original method signature
        if hasattr(report, 'model_dump'): # Pydantic models
            return report.model_dump()
        elif dataclasses.is_dataclass(report): # Python dataclasses
            return dataclasses.asdict(report)
        else: # Fallback, though might not be ideal for custom objects
            return dict(report)

    def run(self, generated_code: str, task_description: str, plan: Dict) -> Dict:
        """
        Evaluates the generated Python code using placeholder logic.
        The 'plan' parameter is added for context, though not used by the current placeholder.
        The 'task_description' is the original user request or task goal.
        """
        print(f"Critic '{self.name}': Evaluating code for task: '{task_description[:50]}...'")
        # For Task 1.5, we directly call the placeholder evaluation logic.
        # The LLM call with CRITIC_SYSTEM_PROMPT is bypassed for now.
        evaluation_report = self.evaluate_code(generated_code, task_description)
        
        # Log the placeholder evaluation
        print(f"Critic '{self.name}': Placeholder evaluation complete. Status: {evaluation_report.get('status')}, Score: {evaluation_report.get('score')}")
        return evaluation_report

class CriticAgent:
    """
    The Critic Agent evaluates Python code generated by the Executor Agent.
    It runs the code, captures errors, generates/runs test cases, and provides
    structured feedback along with a quantifiable score.
    """

    def __init__(self, llm_service: Any = None): # llm_service can be used for test generation later
        self.llm_service = llm_service

    def _calculate_score(self, overall_status: str, num_tests_passed: int, num_tests_total: int) -> float:
        """
        Calculates a quantifiable score based on the analysis report.
        Score is between 0.0 (complete failure) and 1.0 (all tests passed, no errors).
        """
        score = 0.0
        if overall_status == STATUS_CRITICAL_SYNTAX_ERROR:
            score = 0.0
        elif overall_status == STATUS_CRITICAL_RUNTIME_ERROR:
            score = 0.1
        elif overall_status == NO_TESTS_FOUND:
            # Code ran without critical errors, but no tests to verify correctness.
            score = 0.3 
        elif overall_status == STATUS_LOGICAL_ERROR:
            if num_tests_total > 0:
                base_score_logical = 0.2
                max_additional_score_logical = 0.7
                score = base_score_logical + (max_additional_score_logical * (num_tests_passed / num_tests_total))
            else:
                # This case implies a logical error was declared without tests, which is unusual.
                # Default to a low score, similar to runtime error if this state is reached.
                score = 0.15 
        elif overall_status == STATUS_SUCCESS:
            score = 1.0
        
        return max(0.0, min(1.0, round(score, 4)))

    def analyze_results(
        self,
        execution_details: Optional[Dict[str, str]],
        test_results: Optional[List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """
        Consolidates findings from direct code execution and test case execution,
        differentiates between critical errors and logical errors, calculates a score,
        and generates a structured feedback report.

        Args:
            execution_details: A dictionary containing error details if direct execution failed.
                               Example: {'error_type': 'SyntaxError', 'error_message': 'invalid syntax', 'traceback': '...'}
            test_results: A list of dictionaries, where each dictionary represents a test case outcome.
                          Example: [{'name': 'test_one', 'status': 'passed', ...}]

        Returns:
            A dictionary containing the comprehensive analysis report.
            Example:
            {
                'overall_status': str,
                'quantitative_score': float,
                'execution_error_type': str | None,
                'execution_error_message': str | None,
                'execution_traceback': str | None,
                'num_tests_total': int,
                'num_tests_passed': int,
                'num_tests_failed': int,
                'failed_test_details': list[dict],
                'all_test_details': list[dict] | None,
                'concise_summary': str
            }
        """
        report: Dict[str, Any] = {
            'overall_status': "",
            'quantitative_score': 0.0,
            'execution_error_type': None,
            'execution_error_message': None,
            'execution_traceback': None,
            'num_tests_total': 0,
            'num_tests_passed': 0,
            'num_tests_failed': 0,
            'failed_test_details': [],
            'all_test_details': test_results,
            'concise_summary': ""
        }

        if execution_details and execution_details.get('error_type'):
            report['execution_error_type'] = execution_details['error_type']
            report['execution_error_message'] = execution_details.get('error_message', '')
            report['execution_traceback'] = execution_details.get('traceback')
            if execution_details['error_type'] == 'SyntaxError':
                report['overall_status'] = STATUS_CRITICAL_SYNTAX_ERROR
            else:
                report['overall_status'] = STATUS_CRITICAL_RUNTIME_ERROR

        num_tests_total = 0
        num_tests_passed = 0
        failed_test_details: List[Dict[str, Any]] = []

        if test_results:
            num_tests_total = len(test_results)
            for test in test_results:
                if test.get('status') == 'passed':
                    num_tests_passed += 1
                else: 
                    failed_test_details.append(test)
        
        report['num_tests_total'] = num_tests_total
        report['num_tests_passed'] = num_tests_passed
        report['num_tests_failed'] = num_tests_total - num_tests_passed
        report['failed_test_details'] = failed_test_details

        if not report['overall_status']: 
            if num_tests_total > 0:
                if num_tests_passed == num_tests_total:
                    report['overall_status'] = STATUS_SUCCESS
                else:
                    report['overall_status'] = STATUS_LOGICAL_ERROR
            else: 
                report['overall_status'] = NO_TESTS_FOUND
        
        report['quantitative_score'] = self._calculate_score(
            report['overall_status'], 
            report['num_tests_passed'], 
            report['num_tests_total']
        )

        # Generate concise summary
        summary_parts = []
        if report['overall_status'] == STATUS_CRITICAL_SYNTAX_ERROR:
            summary_parts.append(f"Critical syntax error: {report.get('execution_error_message', 'N/A')}.")
            if report.get('execution_traceback'):
                 summary_parts.append(f"Details: {report['execution_traceback']}")
        elif report['overall_status'] == STATUS_CRITICAL_RUNTIME_ERROR:
            summary_parts.append(f"Critical runtime error: {report.get('execution_error_type', 'N/A')} - {report.get('execution_error_message', 'N/A')}.")
            if report.get('execution_traceback'):
                 summary_parts.append(f"Traceback: {report['execution_traceback']}")
        elif report['overall_status'] == NO_TESTS_FOUND:
            summary_parts.append("Code executed without critical errors, but no tests were run to verify correctness.")
        elif report['overall_status'] == STATUS_LOGICAL_ERROR:
            summary_parts.append(f"Code executed, but {report['num_tests_failed']} of {report['num_tests_total']} test(s) failed.")
            if report['failed_test_details']:
                first_failed_test = report['failed_test_details'][0]
                summary_parts.append(f"First failed test: '{first_failed_test.get('name', 'Unnamed test')}' - {first_failed_test.get('error_message', first_failed_test.get('status', 'failed'))}.")
        elif report['overall_status'] == STATUS_SUCCESS:
            summary_parts.append(f"Code executed successfully and all {report['num_tests_total']} test(s) passed.")
        
        report['concise_summary'] = " ".join(summary_parts)
        if not report['concise_summary']:
             report['concise_summary'] = "Analysis complete. No specific issues to summarize based on current status."

        return report

if __name__ == '__main__':
    # Setup (Ensure you have an LLMService implementation and API keys configured)
    # This is a mock LLMService for demonstration.
    # Replace with your actual LLMService instantiation.
    class MockLLMService(LLMService):
        def __init__(self, api_key: str = "mock_key"):
            super().__init__(api_key=api_key, provider="mock")

        def invoke(self, messages: List[Dict[str, str]], expect_json: bool = False) -> Any:
            print(f"MockLLMService received: {messages}")
            last_message_role = messages[-1]["role"]
            last_message_content = messages[-1]["content"]

            if "planner" in messages[0]["content"].lower(): # crude check for planner
                if expect_json:
                    return {"steps": ["Step 1: Define function signature.", "Step 2: Implement logic.", "Step 3: Add docstring."]}
                return "Plan: Define function, implement logic, add docstring."
            elif "executor" in messages[0]["content"].lower(): # crude check for executor
                if "FAIL_TOKEN" in last_message_content: # Simulate executor producing code that should fail
                    return "print('Hello World with FAIL_TOKEN')"
                return "def hello_world():\\n    print('Hello, World!')"
            elif "critic" in messages[0]["content"].lower(): # crude check for critic (though bypassed in Task 1.5)
                if expect_json:
                    if "FAIL_TOKEN" in last_message_content:
                         return {"status": "FAILURE_LLM", "score": 0.1, "reason": "LLM detected issue via prompt."}
                    return {"status": "SUCCESS_LLM", "score": 0.9, "reason": "LLM deemed it okay via prompt."}
                return "LLM says code is fine."
            
            if expect_json:
                return {"error": "Unknown mock interaction"}
            return "Unknown mock interaction"

    # Instantiate the mock LLM service
    llm_service_instance = MockLLMService()

    # Instantiate agents with their respective system prompts
    planner_agent = Planner(name="Alice-Planner", llm_service=llm_service_instance)
    executor_agent = Executor(name="Bob-Executor", llm_service=llm_service_instance)
    # For Task 1.5, the Critic's system prompt is not directly used by the LLM for evaluation,
    # but it's good practice to initialize it.
    critic_agent = Critic(name="Charlie-Critic", llm_service=llm_service_instance) # Default prompt will be used

    # --- Test Case 1: Expected Success ---
    print("\n--- Test Case 1: Expected Success ---")
    user_task_description_success = "Write a Python function that prints 'Hello, World!'"
    
    print(f"User Task: {user_task_description_success}")

    # 1. Planner
    plan_output_success = planner_agent.run(user_request=user_task_description_success)
    print(f"Planner Output: {plan_output_success}")
    
    # 2. Executor
    # Ensure plan_output is a dict, as expected by Executor.
    # If plan_output contains an error, handle it or pass as is, depending on desired flow.
    if "error" not in plan_output_success:
        generated_code_success = executor_agent.run(plan=plan_output_success, original_request=user_task_description_success)
        print(f"Executor Generated Code:\\n{generated_code_success}")

        # 3. Critic (using placeholder logic for Task 1.5)
        # The 'plan_output_success' is passed as the 'plan' argument to critic.run
        critic_report_success = critic_agent.run(
            generated_code=generated_code_success,
            task_description=user_task_description_success,
            plan=plan_output_success
        )
        print(f"Critic Report (Placeholder): {critic_report_success}")
        assert critic_report_success.get("status") == "SUCCESS"
        assert critic_report_success.get("score") == 1.0
    else:
        print(f"Skipping Executor and Critic for success case due to Planner error: {plan_output_success.get('error')}")

    # --- Test Case 2: Expected Failure (using FAIL_TOKEN) ---
    print("\n--- Test Case 2: Expected Failure ---")
    # To trigger failure, we'll have the mock LLM for the executor include "FAIL_TOKEN"
    # We can make the user task description hint at this for the mock.
    user_task_description_fail = "Write a Python function that prints 'Hello World with FAIL_TOKEN' (this should trigger critic failure)."
    print(f"User Task: {user_task_description_fail}")

    # 1. Planner
    plan_output_fail = planner_agent.run(user_request=user_task_description_fail)
    print(f"Planner Output: {plan_output_fail}")

    # 2. Executor
    if "error" not in plan_output_fail:
        # The mock LLM for Executor is set up to return code with "FAIL_TOKEN"
        # if "FAIL_TOKEN" is in the task description it receives.
        generated_code_fail = executor_agent.run(plan=plan_output_fail, original_request=user_task_description_fail)
        print(f"Executor Generated Code:\\n{generated_code_fail}")
        assert "FAIL_TOKEN" in generated_code_fail # Ensure the test setup is correct

        # 3. Critic (using placeholder logic for Task 1.5)
        critic_report_fail = critic_agent.run(
            generated_code=generated_code_fail,
            task_description=user_task_description_fail,
            plan=plan_output_fail
        )
        print(f"Critic Report (Placeholder): {critic_report_fail}")
        assert critic_report_fail.get("status") == "FAILURE_RUNTIME"
        assert critic_report_fail.get("score") == 0.0
        assert "FAIL_TOKEN" in critic_report_fail.get("error_details", "")
    else:
        print(f"Skipping Executor and Critic for failure case due to Planner error: {plan_output_fail.get('error')}")
    
    print("\nBasic sequential flow with placeholder Critic logic (Task 1.5) demonstrated.") 