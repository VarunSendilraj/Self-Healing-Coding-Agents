import os
import sys
import json
import subprocess
import tempfile
from typing import Dict, Any, List, Optional, Tuple

class CodeAnalyzer:
    """
    Analyze and execute Python code snippets and evaluate their performance and correctness.
    """
    
    def __init__(self, runner_path: str = "code_runner.py"):
        """
        Initialize the code analyzer.
        
        Args:
            runner_path: Path to the code_runner.py file
        """
        self.runner_path = os.path.abspath(runner_path)
        if not os.path.exists(self.runner_path):
            raise FileNotFoundError(f"Runner script not found at {self.runner_path}")
    
    def run_code(self, code: str, timeout: int = 5) -> Dict[str, Any]:
        """
        Run the provided code and return the execution results.
        
        Args:
            code: Python code to execute
            timeout: Maximum execution time in seconds
        
        Returns:
            dict: Results of code execution
        """
        try:
            # Create a temporary file for the code
            with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False) as temp_file:
                temp_file.write(code)
                temp_path = temp_file.name
            
            # Run the code using the runner script
            cmd = [sys.executable, self.runner_path, temp_path]
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            # Wait for the process to complete with timeout
            stdout, stderr = process.communicate(timeout=timeout + 1)  # Add 1 second buffer
            
            # Parse the results
            try:
                result = json.loads(stdout)
            except json.JSONDecodeError:
                result = {
                    "success": False,
                    "error_type": "JSONDecodeError",
                    "error_message": "Failed to parse runner output",
                    "stdout": stdout,
                    "stderr": stderr
                }
            
            # Cleanup
            os.unlink(temp_path)
            return result
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error_type": "TimeoutError",
                "error_message": f"Code execution timed out after {timeout} seconds"
            }
        except Exception as e:
            return {
                "success": False,
                "error_type": type(e).__name__,
                "error_message": str(e)
            }
    
    def analyze_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the execution results to provide insights.
        
        Args:
            result: Execution result dictionary
        
        Returns:
            dict: Analysis of the execution results
        """
        analysis = {
            "success": result.get("success", False),
            "execution_time_ms": round(result.get("exec_time", 0) * 1000, 2),
            "memory_usage_mb": round(result.get("memory_usage", 0), 2),
            "has_output": bool(result.get("stdout", "")),
            "has_errors": bool(result.get("stderr", "")) or result.get("error_type") is not None,
            "recommendations": []
        }
        
        # Add recommendations based on the results
        if not analysis["success"]:
            error_type = result.get("error_type")
            error_msg = result.get("error_message", "")
            
            if error_type == "SyntaxError":
                analysis["recommendations"].append("Fix syntax errors in the code")
            elif error_type == "NameError":
                analysis["recommendations"].append("Ensure all variables are defined before use")
            elif error_type == "TimeoutError":
                analysis["recommendations"].append("Optimize code to run more efficiently")
            elif error_type:
                analysis["recommendations"].append(f"Address {error_type}: {error_msg}")
        
        # Performance recommendations
        if analysis["execution_time_ms"] > 1000:
            analysis["recommendations"].append("Consider optimizing for better performance")
        
        if analysis["memory_usage_mb"] > 50:
            analysis["recommendations"].append("High memory usage detected")
        
        return analysis
    
    def test_with_cases(self, code: str, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Run the code against multiple test cases and analyze the results.
        
        Args:
            code: Python code to test
            test_cases: List of test case dictionaries, each containing:
                - inputs: Dict of input values
                - expected: Expected output
        
        Returns:
            dict: Test results and analysis
        """
        results = []
        
        for i, test_case in enumerate(test_cases):
            # Create a test wrapper that calls the function with the test inputs
            test_wrapper = code + "\n\n# Test execution\n"
            
            # Add code to execute the test case
            if "function_name" in test_case:
                func_name = test_case["function_name"]
                inputs = test_case.get("inputs", {})
                
                # Convert inputs to function arguments
                if isinstance(inputs, dict):
                    args_str = ", ".join([f"{k}={repr(v)}" for k, v in inputs.items()])
                elif isinstance(inputs, list):
                    args_str = ", ".join([repr(arg) for arg in inputs])
                else:
                    args_str = repr(inputs)
                
                test_wrapper += f"result = {func_name}({args_str})\n"
                test_wrapper += "print(repr(result))\n"
            
            # Run the test
            result = self.run_code(test_wrapper)
            
            # Check if the output matches the expected result
            expected = test_case.get("expected")
            actual = None
            
            if result["success"] and result["stdout"]:
                try:
                    # Try to evaluate the output as a Python expression
                    actual = eval(result["stdout"].strip())
                except:
                    actual = result["stdout"].strip()
            
            # Add the test result
            test_result = {
                "test_case": i + 1,
                "success": result["success"],
                "expected": expected,
                "actual": actual,
                "matches_expected": expected == actual if expected is not None else None,
                "execution_time_ms": round(result.get("exec_time", 0) * 1000, 2),
                "error": result.get("error_type"),
                "stdout": result.get("stdout", ""),
                "stderr": result.get("stderr", "")
            }
            
            results.append(test_result)
        
        # Compute overall statistics
        passed = sum(1 for r in results if r.get("matches_expected", False))
        total = len(results)
        
        return {
            "overall_success": passed == total and total > 0,
            "tests_passed": passed,
            "tests_total": total,
            "pass_percentage": round(100 * passed / total if total > 0 else 0, 2),
            "detailed_results": results
        }

if __name__ == "__main__":
    # Example usage
    analyzer = CodeAnalyzer()
    
    # Simple example: Test the twoSum function
    code = """
def twoSum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []
"""
    
    test_cases = [
        {
            "function_name": "twoSum",
            "inputs": {"nums": [2, 7, 11, 15], "target": 9},
            "expected": [0, 1]
        },
        {
            "function_name": "twoSum",
            "inputs": {"nums": [3, 2, 4], "target": 6},
            "expected": [1, 2]
        }
    ]
    
    results = analyzer.test_with_cases(code, test_cases)
    print(json.dumps(results, indent=2)) 