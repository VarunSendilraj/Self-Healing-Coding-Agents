#!/usr/bin/env python
"""
Test script to verify that code is being executed via subprocess rather than sandbox.
"""

import os
import sys
import time

# Add the src directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir
if project_root not in sys.path:
    sys.path.insert(0, project_root)

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
    test_code_execution() 