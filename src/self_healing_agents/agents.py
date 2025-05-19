from typing import Dict, List, Any, Optional, Tuple
import json
from .llm_service import LLMService
from .prompts import PLANNER_SYSTEM_PROMPT, EXECUTOR_SYSTEM_PROMPT_V1, CRITIC_SYSTEM_PROMPT, CRITIC_TEST_GENERATION_SYSTEM_PROMPT
from abc import ABC, abstractmethod
import io # Added for capturing stdout
import sys # Added for capturing stderr (though exec captures it directly)
import traceback # Added for formatting exceptions

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

    def _execute_sandboxed_code(self, code_string: str) -> Dict[str, Any]:
        """
        Executes the provided Python code string in a sandboxed environment (code env).
        Captures stdout, stderr, and any exceptions.
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
        # For now, we allow 'print' and basic builtins.
        # We can expand this carefully if specific math functions etc. are required.
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
                "NameError": NameError, # Allow NameError to be caught if code tries to use undefined vars
                "__import__": __import__, # Allow import statements
                "__build_class__": __build_class__, # Allow class definitions
                "repr": repr, # Add repr function for string representation in tests
                "isinstance": isinstance, # Add isinstance function for type checking in tests
            },
            "__build_class__": __build_class__, # Also add here for broader visibility to exec
            # Custom functions or modules could be injected here if safe and necessary
        }
        # No external modules like 'os' or 'sys' are provided by default.

        # Add common module-level attributes
        allowed_globals["__name__"] = "__main__" # Add this directly to allowed_globals
        
        # Add debug logging
        print(f"DEBUG: allowed_globals keys: {list(allowed_globals.keys())}")
        print(f"DEBUG: __builtins__ keys: {list(allowed_globals['__builtins__'].keys())}")
        
        captured_stdout = io.StringIO()
        # stderr from exec is typically captured by the try-except block for Exception
        
        execution_result = {
            "stdout": "",
            "stderr": "", # Will store formatted exception info
            "error_type": None,
            "error_message": "",
            "traceback": "",
            "success": False
        }

        # Redirect stdout
        original_stdout = sys.stdout
        sys.stdout = captured_stdout
        
        try:
            # Execute the code with restricted globals and locals
            # Using an empty dict for locals ensures no unexpected local variables are present.
            print(f"DEBUG: About to execute code:\n{processed_code_string[:200]}...")
            exec(processed_code_string, allowed_globals, {})
            execution_result["success"] = True
        except SyntaxError as e:
            execution_result["error_type"] = "SyntaxError"
            execution_result["error_message"] = str(e)
            # This format is already quite simplified and directly useful.
            simplified_traceback = f"SyntaxError at line {e.lineno}, offset {e.offset}: {e.msg}"
            execution_result["traceback"] = simplified_traceback
            execution_result["stderr"] = simplified_traceback # For SyntaxError, stderr can be the simplified traceback too
        except Exception as e:
            execution_result["error_type"] = e.__class__.__name__
            execution_result["error_message"] = str(e)
            
            # Additional debug logging for errors
            print(f"DEBUG: Exception occurred: {e.__class__.__name__}: {str(e)}")
            print(f"DEBUG: Exception line: {traceback.extract_tb(e.__traceback__)[-1].lineno}, code: {traceback.extract_tb(e.__traceback__)[-1].line}")
            
            full_traceback_str = traceback.format_exc()
            execution_result["stderr"] = full_traceback_str # Store the full traceback in stderr field

            # Simplify traceback for the 'traceback' field using traceback.extract_tb
            simplified_tb_lines = []
            # extracted_tb is a list of FrameSummary objects: (filename, lineno, name, line)
            extracted_frames = traceback.extract_tb(e.__traceback__)
            for frame in extracted_frames:
                if frame.filename == "<string>":
                    # Format: "  File "<string>", line X, in Y"
                    # The 'name' is the function name, or <module> for top-level
                    simplified_tb_lines.append(f'  File "{frame.filename}", line {frame.lineno}, in {frame.name}')
                    if frame.line: # The actual line of code from the <string>
                        simplified_tb_lines.append(f'    {frame.line.strip()}')
            
            # Add the error type and message as the final part of the simplified traceback
            simplified_tb_lines.append(f"{e.__class__.__name__}: {str(e)}")
            
            # Fallback if no <string> frames were found but there was an error
            if not any('File "<string>"' in line for line in simplified_tb_lines) and simplified_tb_lines:
                 # If only "ErrorType: ErrorMessage" is present, it's fine.
                 # If extract_tb returned nothing for <string> but it's not a SyntaxError,
                 # this means the error didn't originate directly in <string> frames that extract_tb picked up
                 # in which case, the simple "ErrorType: ErrorMessage" is the best simplified form.
                 pass # The last line (ErrorType: ErrorMessage) is already what we want
            elif not simplified_tb_lines: # Should not happen if an exception occurred
                simplified_tb_lines.append(f"{e.__class__.__name__}: {str(e)}")

            execution_result["traceback"] = "\\n".join(simplified_tb_lines)
        finally:
            # Restore stdout
            sys.stdout = original_stdout
            execution_result["stdout"] = captured_stdout.getvalue()
            captured_stdout.close()

        return execution_result

    def _generate_test_cases(self, task_description: str, generated_code: str) -> Tuple[Optional[str], List[Dict[str, Any]]]:
        """
        Generates test cases for the given code and task description using an LLM call.
        Returns a tuple: (Optional[str], List[Dict[str, Any]]).
        """
        print(f"Critic '{self.name}': Generating test cases for task: '{task_description[:50]}...'")
        
        user_prompt_content = self.test_generation_system_prompt.format(
            task_description=task_description,
            generated_code=generated_code
        )
        # The CRITIC_TEST_GENERATION_SYSTEM_PROMPT itself is the main content after formatting.
        # We are framing it as a system message that contains the user's request details embedded within it.
        # Or, more accurately, the prompt is a single detailed message to the LLM.
        # Let's use it as the user message, with a minimal system message if needed by LLMService, or just this.
        # For simplicity, if LLMService expects system/user, we can make this the user part.
        # Given the prompt asks for "Output ONLY the JSON", it's acting like a direct instruction.

        messages = [
            # A minimal system message can sometimes help frame the LLM's role if the main prompt is complex.
            # However, our CRITIC_TEST_GENERATION_SYSTEM_PROMPT is already quite directive.
            # Let's assume for now the LLMService handles a single detailed message well, 
            # or the test_generation_prompt itself can be the "user" part of a system/user pair.
            # For self_healing_agents.llm_service.LLMService structure:
            {"role": "system", "content": "You are a helpful AI assistant responding in JSON as requested." }, # Generic system prompt
            {"role": "user", "content": user_prompt_content}
        ]

        try:
            # Expecting a dictionary: {"function_to_test": "func_name", "test_cases": [...]}
            response_data = self.llm_service.invoke(messages, expect_json=True)
            
            if not isinstance(response_data, dict):
                print(f"Critic '{self.name}': Warning - LLM test case generation did not return a dictionary as expected. Response: {response_data}")
                return None, [] # Return None for function name, empty list for tests

            function_name = response_data.get("function_to_test")
            generated_test_cases = response_data.get("test_cases")

            if not isinstance(function_name, str) or not function_name.strip():
                print(f"Critic '{self.name}': Warning - LLM did not provide a valid 'function_to_test' string. Response: {response_data}")
                function_name = None # Or handle as an error preventing test execution
            
            if not isinstance(generated_test_cases, list):
                print(f"Critic '{self.name}': Warning - LLM did not provide 'test_cases' as a list. Response: {response_data}")
                return function_name, [] # Return potentially valid function name, but empty tests

            # Basic validation: check if it's a list of dictionaries with required keys
            valid_test_cases = []
            for item in generated_test_cases:
                if isinstance(item, dict) and all(key in item for key in ["test_case_name", "inputs", "expected_output"]):
                    valid_test_cases.append(item)
                else:
                    print(f"Critic '{self.name}': Warning - LLM returned a list item not matching test case structure: {item}")
            
            if not valid_test_cases and generated_test_cases: # LLM returned a list, but none were valid test cases
                print(f"Critic '{self.name}': Warning - LLM returned a list for 'test_cases', but no valid test cases found in it.")
            
            return function_name, valid_test_cases # Return both function name and the list of test cases
            
        except json.JSONDecodeError as e:
            print(f"Critic '{self.name}': Error decoding JSON from LLM for test cases: {e}")
            return None, []
        except Exception as e:
            print(f"Critic '{self.name}': Unexpected error during test case generation: {e}")
            return None, []

    def evaluate_code(self, code: str, task_description: str) -> Dict[str, Any]:
        """
        Evaluates code by executing it in a sandbox, generating test cases, 
        and (eventually) running them.
        """
        print(f"Critic '{self.name}': Attempting to execute code in sandbox.")
        execution_details = self._execute_sandboxed_code(code)
        print(f"Critic '{self.name}': Sandbox execution details: {execution_details}")

        # Initialise test case variables
        function_to_test: Optional[str] = None
        generated_test_specs: List[Dict[str, Any]] = []
        test_results: List[Dict[str, Any]] = [] # This will store results of actual test runs
        num_tests_passed = 0
        num_tests_total = 0
        
        # Variables to hold report details, to be populated within try or except block
        report_status = ""
        report_score = 0.0
        report_summary = ""

        try:
            # If code execution was successful, proceed to generate and run test cases
            if execution_details["success"]:
                function_to_test, generated_test_specs = self._generate_test_cases(task_description, code)
                print(f"Critic '{self.name}': Generated {len(generated_test_specs)} test specifications for function '{function_to_test}'.")

                num_tests_total = len(generated_test_specs)
                if function_to_test and generated_test_specs:
                    print(f"Critic '{self.name}': Attempting to run {num_tests_total} test cases...")
                    for test_spec in generated_test_specs:
                        test_case_name = test_spec.get("test_case_name", "unnamed_test")
                        inputs = test_spec.get("inputs", {})
                        expected_output = test_spec.get("expected_output")

                        input_args_str = ""
                        if isinstance(inputs, dict):
                            if len(inputs) == 1 and 'lists' in inputs:
                                input_args_str = f"lists={repr(inputs['lists'])}"
                            else:
                                input_args_str = ", ".join(f"{k}={repr(v)}" for k, v in inputs.items())
                        else:
                            print(f"Critic '{self.name}': Warning - test inputs for '{test_case_name}' is not a dict: {inputs}")
                            input_args_str = repr(inputs)

                        # Strip markdown fences from the code before including it in the test script
                        processed_code = code.strip()
                        if processed_code.startswith("```python") and processed_code.endswith("```"):
                            processed_code = processed_code[len("```python"):-(len("```"))].strip()
                        elif processed_code.startswith("```") and processed_code.endswith("```"):
                            processed_code = processed_code[len("```"):-(len("```"))].strip()

                        test_execution_script = f'''
# --- Start of Generated Code from Executor ---
{processed_code}
# --- End of Generated Code from Executor ---

# --- Test Execution for: {test_case_name} ---
actual_output = None
test_passed = False
error_occurred = False
error_message = ""
comparison_error_message = ""

print(f"DEBUG_TEST_SCRIPT: Executing test: {test_case_name}")
print(f"DEBUG_TEST_SCRIPT: Inputs prepared as: {input_args_str}")
print(f"DEBUG_TEST_SCRIPT: Expected output: {repr(expected_output)}")

try:
    actual_output = {function_to_test}({input_args_str})
    # Compare with expected_output_spec directly instead of using expected_output variable
    if repr(actual_output) == repr({repr(expected_output)}):
        test_passed = True
    else:
        comparison_error_message = f"Expected: {repr(expected_output)}, Got: {{repr(actual_output)}}" 

except Exception as e:
    error_occurred = True
    error_message = str(e)
    print(f"DEBUG_TEST_SCRIPT: Exception during test execution: {{error_message}}")

print("__TEST_RESULT_START__")
print(f"test_case_name={test_case_name}")
print(f"test_passed={{test_passed}}")
print(f"actual_output={{repr(actual_output)}}")
print(f"expected_output={repr(expected_output)}")
print(f"error_occurred={{error_occurred}}")
print(f"error_message={{error_message}}")
print(f"comparison_error_message={{comparison_error_message}}")
print("__TEST_RESULT_END__")
'''
                        print(f"Critic '{self.name}': Executing test script for '{test_case_name}'...")
                        single_test_exec_details = self._execute_sandboxed_code(test_execution_script)
                        
                        current_test_result = {
                            "name": test_case_name,
                            "inputs": inputs,
                            "expected_output_spec": expected_output,
                            "status": "error_running_test",
                            "actual_output": None,
                            "stdout": single_test_exec_details.get("stdout", ""),
                            "stderr": single_test_exec_details.get("stderr", ""),
                            "error_message": single_test_exec_details.get("error_message", "Unknown error during test script execution")
                        }

                        if single_test_exec_details["success"]:
                            stdout_lines = single_test_exec_details.get("stdout", "").splitlines()
                            test_data = {}
                            in_test_block = False
                            for line in stdout_lines:
                                if line == "__TEST_RESULT_START__":
                                    in_test_block = True
                                    continue
                                if line == "__TEST_RESULT_END__":
                                    in_test_block = False
                                    break
                                if in_test_block and '=' in line:
                                    key_val = line.split('=', 1)
                                    if len(key_val) == 2:
                                        test_data[key_val[0].strip()] = key_val[1].strip()
                            
                            if test_data.get("test_passed") == "True":
                                current_test_result["status"] = "passed"
                                num_tests_passed += 1
                            else:
                                current_test_result["status"] = "failed"
                                current_test_result["error_message"] = test_data.get("error_message") or test_data.get("comparison_error_message")
                            current_test_result["actual_output"] = test_data.get("actual_output")
                        else:
                            current_test_result["status"] = "framework_error"
                            current_test_result["error_message"] = f"Sandbox execution of test script failed. Type: {single_test_exec_details.get('error_type')}, Msg: {single_test_exec_details.get('error_message')}"
                        
                        test_results.append(current_test_result)
                    print(f"Critic '{self.name}': Finished running tests. Passed: {num_tests_passed}/{num_tests_total}")
                else:
                    if not function_to_test:
                        print(f"Critic '{self.name}': Skipping test execution as no function_to_test was identified.")
                    if not generated_test_specs:
                        print(f"Critic '{self.name}': Skipping test execution as no test specifications were generated.")

            # Determine report status, score, and summary based on execution and tests
            if not execution_details["success"]:
                report_status = f"FAILURE_{execution_details.get('error_type', 'UNKNOWN_EXECUTION_ERROR').upper()}"
                report_score = 0.0
                report_summary = f"Code execution failed in sandbox: {execution_details.get('error_type', 'Unknown error')}. No test cases generated or run."
            else:
                if num_tests_total == 0:
                    report_status = "SUCCESS_EXECUTION_NO_TESTS"
                    report_score = 0.7 
                    report_summary = f"Code executed successfully. No test cases were generated or run for function '{function_to_test}'."
                elif num_tests_passed == num_tests_total:
                    report_status = "SUCCESS"
                    report_score = 1.0
                    report_summary = f"Code executed successfully. All {num_tests_passed}/{num_tests_total} tests passed for function '{function_to_test}'."
                else:
                    report_status = "FAILURE_LOGIC"
                    report_score = 0.2 + 0.5 * (num_tests_passed / num_tests_total)
                    report_summary = f"Code executed successfully, but {num_tests_total - num_tests_passed}/{num_tests_total} tests failed for function '{function_to_test}'."

        except Exception as e: # Catch exceptions during the test case generation or execution loop
            print(f"CRITIC_ERROR: Unexpected error in evaluate_code: {e.__class__.__name__}: {e}")
            print(f"CRITIC_ERROR_TRACEBACK: {traceback.format_exc()}")
            report_status = f"FAILURE_CRITIC_INTERNAL_ERROR"
            report_score = 0.0
            report_summary = f"Critic agent encountered an internal error during test processing: {e}"
            # Ensure execution_details is not None for report structure; it might be if error is before its first assignment
            if execution_details is None: # This implies error happened before main code exec details were known
                execution_details = { 
                    "stdout": "", 
                    "stderr": f"Critic internal error: {e}", 
                    "error_type": e.__class__.__name__, 
                    "error_message": str(e), 
                    "traceback": traceback.format_exc(), 
                    "success": False 
                }
        
        report = {
            "status": report_status,
            "score": round(report_score, 4),
            "execution_stdout": execution_details["stdout"],
            "execution_stderr": execution_details["stderr"],
            "error_details": None if execution_details["success"] else {
                "type": execution_details["error_type"],
                "message": execution_details["error_message"],
                "traceback": execution_details["traceback"]
            },
            "test_results": test_results,
            "generated_test_specifications": generated_test_specs,
            "function_to_test": function_to_test,
            "summary": report_summary
        }
        return report

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