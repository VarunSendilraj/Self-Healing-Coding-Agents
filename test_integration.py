#!/usr/bin/env python
"""
Test script for the CodeAnalyzer integration with the Self-Healing Agents framework.

This script validates:
1. Proper Markdown code block handling
2. Early return logic for successful evaluations
3. Test case generation from descriptions
"""

import os
import sys
import json
import time

# Add the src directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
    
# Import our code analyzer components
from code_analyzer import CodeAnalyzer
from src.self_healing_agents.evaluation.code_analyzer_integration import CodeAnalyzerCritic

def test_markdown_code_block_handling():
    """Test the ability to handle various Markdown code block formats."""
    print("\n=== Testing Markdown Code Block Handling ===")
    
    # Initialize the CodeAnalyzerCritic
    critic = CodeAnalyzerCritic()
    
    # Test cases with different Markdown formats
    test_cases = [
        {
            "name": "Standard Python Block",
            "code": """```python
def twoSum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []
```"""
        },
        {
            "name": "Generic Code Block",
            "code": """```
def twoSum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []
```"""
        },
        {
            "name": "Nested Code Block",
            "code": """Here's my solution:

```python
def twoSum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []
```

I believe this is correct."""
        },
        {
            "name": "Plain Code (No Markdown)",
            "code": """def twoSum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []"""
        }
    ]
    
    # Test each case
    for case in test_cases:
        print(f"\nTesting: {case['name']}")
        cleaned_code = critic.strip_markdown_code_blocks(case["code"])
        
        # Verify the output contains the function definition
        if "def twoSum" in cleaned_code:
            print(f"✅ PASS: Successfully extracted code")
        else:
            print(f"❌ FAIL: Code extraction failed")
            print(f"Cleaned code:\n{cleaned_code}")

def test_function_name_extraction():
    """Test the ability to extract function names from code and task descriptions."""
    print("\n=== Testing Function Name Extraction ===")
    
    # Initialize the CodeAnalyzerCritic
    critic = CodeAnalyzerCritic()
    
    # Test cases with different code and task descriptions
    test_cases = [
        {
            "name": "From Code",
            "code": """def twoSum(nums, target):
                return []""",
            "task": "Implement a function",
            "expected": "twoSum"
        },
        {
            "name": "From Task Description",
            "code": "# Some comment",
            "task": "Implement a maxSubArray function that finds...",
            "expected": "maxSubArray"
        },
        {
            "name": "Common Algorithm",
            "code": "# TODO",
            "task": "Solve the twoSum problem",
            "expected": "twoSum"
        }
    ]
    
    # Test each case
    for case in test_cases:
        print(f"\nTesting: {case['name']}")
        function_name = critic.extract_function_name(case["code"], case["task"])
        
        if function_name == case["expected"]:
            print(f"✅ PASS: Extracted '{function_name}' as expected")
        else:
            print(f"❌ FAIL: Expected '{case['expected']}' but got '{function_name}'")

def test_test_case_generation():
    """Test the ability to generate test cases from task descriptions."""
    print("\n=== Testing Test Case Generation ===")
    
    # Initialize the CodeAnalyzerCritic
    critic = CodeAnalyzerCritic()
    
    # Test with a twoSum task description
    task_description = """
    Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.
    
    Example 1:
    Input: nums = [2,7,11,15], target = 9
    Output: [0,1]
    
    Example 2:
    Input: nums = [3,2,4], target = 6
    Output: [1,2]
    """
    
    print("\nGenerating test cases for twoSum...")
    test_cases = critic.generate_test_cases_from_description(task_description, "twoSum")
    
    if len(test_cases) > 0:
        print(f"✅ PASS: Generated {len(test_cases)} test cases")
        for i, test_case in enumerate(test_cases):
            print(f"  Test Case {i+1}: {test_case}")
    else:
        print("❌ FAIL: No test cases generated")

def run_full_integration_test():
    """Run a simple end-to-end test of the code analyzer critic."""
    print("\n=== Running Full Integration Test ===")
    
    # Initialize the CodeAnalyzerCritic
    critic = CodeAnalyzerCritic()
    
    # Define a correct twoSum implementation
    correct_code = """```python
def twoSum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []
```"""
    
    # Define the task description
    task_description = "Implement the twoSum function that returns indices of two numbers that add up to target."
    
    # Run the critic
    print("\nRunning critic with correct twoSum implementation...")
    result = critic.run(correct_code, task_description)
    
    # Check the result
    if result.get("status") == "SUCCESS" and result.get("score") > 0.9:
        print(f"✅ PASS: Critic correctly identified successful code (Score: {result.get('score')})")
    else:
        print(f"❌ FAIL: Critic should have identified this as successful code")
        print(f"Status: {result.get('status')}")
        print(f"Score: {result.get('score')}")
    
    # Print test results
    print("\nTest results:")
    for test_result in result.get("test_results", []):
        status = "✅ PASS" if test_result.get("passed") else "❌ FAIL"
        print(f"  {status}: Test case {test_result.get('test_case')}")
        print(f"    Input: {test_result.get('input')}")
        print(f"    Expected: {test_result.get('expected')}")
        print(f"    Actual: {test_result.get('actual')}")

def test_code_execution():
    """
    Test function to create a simple Python snippet and run it through the code analyzer.
    The test will write to a file to verify it's running as a subprocess.
    """
    from code_analyzer import CodeAnalyzer
    
    # Create a test script that writes to a file to prove it's running as a subprocess
    test_code = """
import os
import time

# Get the current process ID
pid = os.getpid()

# Create a file with the process ID
with open('subprocess_test.txt', 'w') as f:
    f.write(f"Process ID: {pid}\\n")
    f.write(f"Working directory: {os.getcwd()}\\n")
    f.write(f"This file was created at: {time.ctime()}\\n")
    f.write(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}\\n")

# Return a value for test evaluation
def test_function(x):
    return x * 2

result = test_function(21)
"""
    
    # Create test case
    test_case = {
        "function_name": "test_function",
        "inputs": {"x": 21},
        "expected": 42
    }
    
    # Initialize the code analyzer
    analyzer = CodeAnalyzer()
    
    # Run the test
    print("Running code through the CodeAnalyzer...")
    test_results = analyzer.test_with_cases(test_code, [test_case])
    
    # Check if the file was created (which proves subprocess execution)
    if os.path.exists('subprocess_test.txt'):
        print("\nSUCCESS: File was created by the subprocess")
        with open('subprocess_test.txt', 'r') as f:
            print("\nContents of subprocess_test.txt:")
            print(f.read())
    else:
        print("\nFAILURE: No file was created, code might be running in a sandbox")
    
    # Print test results
    print("\nTest Results:")
    print(f"Overall success: {test_results['overall_success']}")
    print(f"Tests passed: {test_results['tests_passed']} / {test_results['tests_total']}")
    
    # Also try running through the enhanced harness integration
    print("\nTesting enhanced_harness_integration.py...")
    try:
        from self_healing_agents.evaluation import enhanced_harness_integration
        print("Successfully imported enhanced_harness_integration")
    except ImportError as e:
        print(f"Import error: {e}")

if __name__ == "__main__":
    print("Running integration tests for CodeAnalyzer...")
    
    # Run the tests
    test_markdown_code_block_handling()
    test_function_name_extraction()
    test_test_case_generation()
    run_full_integration_test()
    
    test_code_execution()
    
    print("\nTests completed!") 