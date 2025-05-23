"""
Integration module for the code analysis system with the self-healing agents framework.

This module provides wrappers and utilities to use the code analyzer system 
instead of the sandbox environment for code execution and evaluation.
"""

import os
import sys
import json
import re
from typing import Dict, Any, List, Optional, Tuple

# Add the parent directory to the path so we can import our code analyzer
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the code analyzer components
from code_analyzer import CodeAnalyzer
# Import the prompt and LLMService from the main agents module
from self_healing_agents.prompts import CRITIC_TEST_GENERATION_SYSTEM_PROMPT
from self_healing_agents.llm_service import LLMService

class CodeAnalyzerCritic:
    """
    A wrapper around the CodeAnalyzer to make it compatible with the self-healing agents Critic interface.
    """
    
    def __init__(self, test_cases: Optional[List[Dict[str, Any]]] = None, llm_service: Optional[LLMService] = None):
        """
        Initialize the CodeAnalyzerCritic.
        
        Args:
            test_cases: Optional list of test cases to use for evaluation
            llm_service: LLM service for generating test cases dynamically
        """
        self.analyzer = CodeAnalyzer()
        self.test_cases = test_cases or []
        self.llm_service = llm_service
        
    def set_test_cases(self, test_cases: List[Dict[str, Any]]) -> None:
        """
        Set the test cases to be used for evaluation.
        
        Args:
            test_cases: List of test case dictionaries
        """
        self.test_cases = test_cases
    
    def strip_markdown_code_blocks(self, code: str) -> str:
        """
        Strip Markdown code block formatting from the code.
        
        Args:
            code: Code that might contain Markdown formatting
            
        Returns:
            Clean Python code without Markdown formatting
        """
        # Print the original code for debugging
        print(f"Original code:\n{code[:200]}...")
        
        if not code or len(code.strip()) < 3:
            return code
        
        # Check if the code is wrapped in Markdown code blocks
        # Case 1: Code wrapped in ```python ... ``` (most common)
        code_block_pattern = r'^```(?:python)?\s*([\s\S]*?)```\s*$'
        code_block_match = re.match(code_block_pattern, code.strip())
        
        if code_block_match:
            # Extract the code between the Markdown markers
            clean_code = code_block_match.group(1).strip()
            print(f"Detected Markdown code block. Cleaned code:\n{clean_code[:200]}...")
            return clean_code
        
        # Case 2: Code has nested code blocks (```python ... ``` inside other text)
        nested_code_blocks = re.findall(r'```python\s*(.*?)\s*```', code, re.DOTALL)
        if nested_code_blocks:
            # Find the largest code block (most likely the main solution)
            largest_block = max(nested_code_blocks, key=len)
            if len(largest_block) > 10:  # Make sure it's a substantial code block
                clean_code = largest_block.strip()
                print(f"Detected nested Markdown code block. Cleaned code:\n{clean_code[:200]}...")
                return clean_code
        
        # Case 3: Code has generic code blocks without language (``` ... ```)
        generic_blocks = re.findall(r'```\s*(.*?)\s*```', code, re.DOTALL)
        if generic_blocks:
            # Find the largest code block (most likely the main solution)
            largest_block = max(generic_blocks, key=len)
            if len(largest_block) > 10:  # Make sure it's a substantial code block
                clean_code = largest_block.strip()
                print(f"Detected generic Markdown code block. Cleaned code:\n{clean_code[:200]}...")
                return clean_code
        
        # Case 4: Check for multiple backtick blocks and find one with Python-like syntax
        if '```' in code:
            # Try to find a code block that contains Python keywords
            python_keywords = ['def ', 'import ', 'class ', 'return ', 'if ', 'for ', 'while ']
            all_blocks = re.findall(r'```.*?\n(.*?)```', code, re.DOTALL)
            
            for block in all_blocks:
                if any(keyword in block for keyword in python_keywords):
                    clean_code = block.strip()
                    print(f"Detected Python-like code block. Cleaned code:\n{clean_code[:200]}...")
                    return clean_code
        
        # If no Markdown formatting is detected, return the original code
        return code
    
    def extract_function_name(self, code: str, task_description: str) -> str:
        """
        Extract the main function name from the code or task description.
        
        Args:
            code: Generated code
            task_description: Task description
            
        Returns:
            Function name extracted from code or a default name
        """
        # Clean the code before extracting the function name
        clean_code = self.strip_markdown_code_blocks(code)
        
        # Basic extraction of function name from code
        function_match = re.search(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', clean_code)
        if function_match:
            return function_match.group(1)
        
        # Try to extract from task description (e.g., "implement a twoSum function")
        task_match = re.search(r'implement\s+(?:a|an)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+function', task_description, re.IGNORECASE)
        if task_match:
            return task_match.group(1)
        
        # Check for common algorithm names in the task description
        common_functions = [
            "twoSum", "maxSubArray", "merge", "maxProfit", "isValid", 
            "climbStairs", "reverse", "isPalindrome", "romanToInt"
        ]
        
        for func in common_functions:
            if func.lower() in task_description.lower():
                return func
        
        # Fall back to a default name
        return "solution"

    def generate_test_cases_with_llm(self, task_description: str, generated_code: str) -> List[Dict[str, Any]]:
        """
        Generate test cases using the LLM service (same as the existing Critic system).
        
        Args:
            task_description: Task description
            generated_code: The generated code to test
            
        Returns:
            List of test case dictionaries in CodeAnalyzer format
        """
        if not self.llm_service:
            print("No LLM service available for test case generation")
            return []
            
        print(f"Generating test cases using LLM for task: '{task_description[:50]}...'")
        
        user_prompt_content = CRITIC_TEST_GENERATION_SYSTEM_PROMPT.format(
            task_description=task_description,
            generated_code=generated_code
        )
        
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant responding in JSON as requested."},
            {"role": "user", "content": user_prompt_content}
        ]

        try:
            # Expecting a dictionary: {"function_to_test": "func_name", "test_cases": [...]}
            response_data = self.llm_service.invoke(messages, expect_json=True)
            
            if not isinstance(response_data, dict):
                print(f"Warning - LLM test case generation did not return a dictionary as expected. Response: {response_data}")
                return []

            function_name = response_data.get("function_to_test")
            generated_test_specs = response_data.get("test_cases")

            if not isinstance(function_name, str) or not function_name.strip():
                print(f"Warning - LLM did not provide a valid 'function_to_test' string. Response: {response_data}")
                function_name = "solution"  # Use fallback
            
            if not isinstance(generated_test_specs, list):
                print(f"Warning - LLM did not provide 'test_cases' as a list. Response: {response_data}")
                return []

            # Convert from the existing Critic format to CodeAnalyzer format
            converted_test_cases = []
            for test_spec in generated_test_specs:
                if isinstance(test_spec, dict) and all(key in test_spec for key in ["test_case_name", "inputs", "expected_output"]):
                    converted_test_case = {
                        "function_name": function_name,
                        "inputs": test_spec["inputs"],
                        "expected": test_spec["expected_output"]
                    }
                    converted_test_cases.append(converted_test_case)
                    print(f"  Converted test case: {test_spec['test_case_name']} -> {converted_test_case}")
                else:
                    print(f"Warning - LLM returned a list item not matching test case structure: {test_spec}")
            
            print(f"Successfully generated {len(converted_test_cases)} test cases using LLM")
            return converted_test_cases
            
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from LLM for test cases: {e}")
            return []
        except Exception as e:
            print(f"Unexpected error during LLM test case generation: {e}")
            return []

    def generate_test_cases_from_description(self, task_description: str, function_name: str) -> List[Dict[str, Any]]:
        """
        Generate test cases using LLM first, then fallback to parsing examples from task description.
        
        Args:
            task_description: Task description, potentially containing examples.
            function_name: Name of the function to test (used in the output test_case structure).
            
        Returns:
            List of test case dictionaries.
        """
        print(f"Attempting to generate test cases for function: {function_name}")
        
        # First try to use LLM if available
        if self.llm_service:
            # We need the generated_code parameter for LLM generation, but we don't have it here
            # This method is called before we have the generated code, so we can't use LLM here
            # We'll handle LLM generation in the main run method instead
            print("LLM service available, but generated code not available yet. Will use LLM in main run method.")
            return []
        
        # Fallback to the old parsing logic (keeping minimal for backward compatibility)
        print("No LLM service available. Using regex parsing fallback.")
        test_cases = []

        # Generic pattern to find examples: "Example X:", "Input:", "Output:"
        example_blocks = re.findall(r"Example\s*\d*:.*?Input:(.*?)Output:(.*?)(?=Example\s*\d*:|$)", task_description, re.DOTALL | re.IGNORECASE)
        
        if not example_blocks:
            example_blocks = re.findall(r"Input:(.*?)Output:(.*?)(?=Input:|$)", task_description, re.DOTALL | re.IGNORECASE)

        print(f"Found {len(example_blocks)} potential example block(s) in task description using regex.")

        for i, (input_text, output_text) in enumerate(example_blocks):
            input_text = input_text.strip()
            output_text = output_text.strip()
            
            print(f"Processing potential example {i+1}:")
            print(f"  Raw Input Text: {input_text}")
            print(f"  Raw Output Text: {output_text}")

            try:
                parsed_inputs = {}
                input_segments = re.findall(r"([a-zA-Z_][a-zA-Z0-9_]*\s*=\s*.*?)(?:,|$)", input_text)
                
                if input_segments:
                    for segment in input_segments:
                        key_value_match = re.match(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(.*)", segment.strip())
                        if key_value_match:
                            key, value_str = key_value_match.groups()
                            try:
                                parsed_inputs[key.strip()] = eval(value_str.strip())
                            except:
                                parsed_inputs[key.strip()] = value_str.strip()
                elif input_text:
                    # Simple heuristic for single inputs
                    try:
                        if input_text.startswith("{") and input_text.endswith("}"):
                             parsed_inputs = eval(input_text)
                        else:
                             param_name = "arg1"
                             if function_name == 'totalNQueens':
                                 param_name = 'n'
                             try:
                                parsed_inputs[param_name] = eval(input_text)
                             except:
                                parsed_inputs[param_name] = input_text
                    except Exception as e_single_input:
                        print(f"  Could not parse single input: {input_text}, error: {e_single_input}")
                        if not parsed_inputs:
                           parsed_inputs = {"raw_input": input_text}

                # Parse expected output
                try:
                    expected = eval(output_text)
                except:
                    expected = output_text
                
                if parsed_inputs and expected is not None:
                    test_case = {
                        "function_name": function_name,
                        "inputs": parsed_inputs,
                        "expected": expected
                    }
                    print(f"  Created test case: {test_case}")
                    test_cases.append(test_case)
                else:
                    print(f"  Failed to create a complete test case from this example block.")
                    
            except Exception as e:
                print(f"  Error processing example block: {e}")
        
        if not test_cases:
            print(f"No test cases could be generated from the task description for {function_name}.")
        
        return test_cases
    
    def run(self, generated_code: str, task_description: str, plan: Any = None) -> Dict[str, Any]:
        """
        Run the code analysis and evaluation, compatible with the Critic interface.
        
        Args:
            generated_code: The code to evaluate
            task_description: Description of the task
            plan: Optional plan information (not used by this implementation)
            
        Returns:
            Dictionary containing the evaluation results
        """
        # Clean the code by removing Markdown formatting
        clean_code = self.strip_markdown_code_blocks(generated_code)
        
        # First, extract the function name
        function_name = self.extract_function_name(clean_code, task_description)
        print(f"Extracted function name: {function_name}")
        
        # Generate test cases if we don't have any
        if not self.test_cases:
            # Try LLM generation first if available
            if self.llm_service:
                print("Using LLM to generate test cases")
                self.test_cases = self.generate_test_cases_with_llm(task_description, clean_code)
            
            # If LLM generation failed or not available, fallback to description parsing
            if not self.test_cases:
                print("Falling back to description parsing for test case generation")
                self.test_cases = self.generate_test_cases_from_description(task_description, function_name)
            
        # Check if we have test cases now
        if self.test_cases:
            print(f"Running with {len(self.test_cases)} test cases")
            # Run the test cases
            test_results = self.analyzer.test_with_cases(clean_code, self.test_cases)
            
            # Calculate a score based on test case results
            score = test_results["pass_percentage"] / 100.0
            
            # Determine status
            status = "SUCCESS" if test_results["overall_success"] else "FAILURE_EVALUATION"
            
            # Generate a summary
            if test_results["overall_success"]:
                summary = f"All {test_results['tests_total']} test cases passed successfully."
            else:
                summary = f"Failed {test_results['tests_total'] - test_results['tests_passed']} out of {test_results['tests_total']} test cases."
            
            # Format the test results for the Critic report
            formatted_test_results = []
            for result in test_results["detailed_results"]:
                formatted_result = {
                    "test_case": result["test_case"],
                    "passed": result["matches_expected"],
                    "input": self.test_cases[result["test_case"] - 1]["inputs"],
                    "expected": result["expected"],
                    "actual": result["actual"],
                    "execution_time_ms": result["execution_time_ms"],
                    "error": result["error"],
                    "stdout": result.get("stdout", ""),
                    "stderr": result.get("stderr", "")
                }
                formatted_test_results.append(formatted_result)
            
            # Get additional analysis
            code_analysis = self.analyzer.analyze_result({"success": True})  # Basic analysis
            
            # Construct the full report
            critic_report = {
                "status": status,
                "score": score,
                "summary": summary,
                "error_details": "",
                "test_results": formatted_test_results,
                "generated_code_for_report": generated_code,  # Keep the original code for reporting
                "execution_time_ms": code_analysis["execution_time_ms"],
                "memory_usage_mb": code_analysis["memory_usage_mb"],
                "recommendations": code_analysis["recommendations"]
            }
        else:
            print("No test cases available, running basic code analysis")
            # Just run the code and analyze the results
            result = self.analyzer.run_code(clean_code)
            analysis = self.analyzer.analyze_result(result)
            
            # Convert to Critic format
            critic_report = {
                "status": "SUCCESS" if analysis["success"] else "FAILURE_EVALUATION",
                "score": 1.0 if analysis["success"] else 0.0,
                "summary": "Code executed successfully" if analysis["success"] else f"Code execution failed: {result.get('error_type')}",
                "error_details": result.get("error_message", ""),
                "test_results": [],
                "generated_code_for_report": generated_code,  # Keep the original code for reporting
                "execution_time_ms": analysis["execution_time_ms"],
                "memory_usage_mb": analysis["memory_usage_mb"],
                "recommendations": analysis["recommendations"]
            }
        
        return critic_report

# Helper function to integrate with the enhanced harness
def integrate_code_analyzer_with_harness(harness_module):
    """
    Monkey patch the enhanced harness to use the CodeAnalyzerCritic.
    
    Args:
        harness_module: The harness module to patch
    """
    # Store the original Critic class
    original_critic_class = harness_module.Critic
    
    # Create a new Critic class that uses our CodeAnalyzerCritic
    class EnhancedCritic(original_critic_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Pass the LLM service from the parent Critic to the CodeAnalyzerCritic
            self.code_analyzer_critic = CodeAnalyzerCritic(llm_service=self.llm_service)
        
        def run(self, generated_code, task_description, plan=None):
            # Use our code analyzer critic instead of the original run method
            return self.code_analyzer_critic.run(generated_code, task_description, plan)
    
    # Replace the Critic class in the harness module
    harness_module.Critic = EnhancedCritic
    
    print("Successfully integrated CodeAnalyzer with the Enhanced Evaluation Harness!") 