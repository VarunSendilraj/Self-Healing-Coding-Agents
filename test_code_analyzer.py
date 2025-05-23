#!/usr/bin/env python
"""
Minimal test script to verify that the code analyzer integration is working correctly.
"""

import os
import sys

# Add the src directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

def main():
    """Main test function."""
    try:
        # Import the code analyzer integration module
        from self_healing_agents.evaluation import enhanced_harness
        from self_healing_agents.evaluation.code_analyzer_integration import integrate_code_analyzer_with_harness, CodeAnalyzerCritic
        from code_analyzer import CodeAnalyzer
        
        # Create a test file to show we are in the parent process
        with open('parent_process.txt', 'w') as f:
            f.write(f"Parent process ID: {os.getpid()}\n")
            f.write(f"This file was created by the parent process\n")
        
        print(f"Parent process ID: {os.getpid()}")
        
        # Get the full path to code_runner.py
        runner_path = os.path.join(current_dir, "code_runner.py")
        if not os.path.exists(runner_path):
            print(f"Warning: code_runner.py not found at {runner_path}")
            print("Searching for code_runner.py...")
            
            # List files in current directory to look for code_runner.py
            files = os.listdir(current_dir)
            if "code_runner.py" in files:
                runner_path = os.path.join(current_dir, "code_runner.py")
                print(f"Found code_runner.py in current directory: {runner_path}")
            else:
                print("code_runner.py not found in current directory.")
                print("Available files:", files)
        
        print(f"Using runner path: {runner_path}")
        
        # Create a custom code analyzer with the explicit runner path
        analyzer = CodeAnalyzer(runner_path=runner_path)
        
        # Create a code analyzer critic with our custom analyzer
        class CustomCodeAnalyzerCritic(CodeAnalyzerCritic):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.analyzer = analyzer
        
        # Create a test code that creates a file
        test_code = """
def test_function():
    import os
    
    # Get the current process ID
    pid = os.getpid()
    
    # Create a file with the process ID
    with open('analyzer_subprocess_test.txt', 'w') as f:
        f.write(f"Process ID: {pid}\\n")
        f.write(f"This file was created by the code analyzer test\\n")
    
    return 42
        """
        
        # Create test case
        test_case = {
            "function_name": "test_function",
            "inputs": {},
            "expected": 42
        }
        
        # Test the analyzer directly first
        print("Testing CodeAnalyzer directly...")
        test_wrapper = test_code + "\n\n# Test execution\nresult = test_function()\nprint(repr(result))\n"
        result = analyzer.run_code(test_wrapper)
        print("Direct CodeAnalyzer result:", result)
        
        # Check if the file was created
        if os.path.exists('analyzer_subprocess_test.txt'):
            print("\nSUCCESS: File was created by the subprocess")
            with open('analyzer_subprocess_test.txt', 'r') as f:
                print("\nContents of analyzer_subprocess_test.txt:")
                print(f.read())
        else:
            print("\nFAILURE: No file was created, code might be running in a sandbox")
        
        # Now test with the critic
        critic = CustomCodeAnalyzerCritic()
        critic.set_test_cases([test_case])
        
        # Run the code analysis
        print("\nRunning code analysis with CodeAnalyzerCritic...")
        result = critic.run(test_code, "Create a function that writes to a file and returns 42")
        
        # Check if the file was created (which would prove subprocess execution)
        if os.path.exists('analyzer_subprocess_test.txt'):
            print("\nSUCCESS: File was created by the subprocess")
            with open('analyzer_subprocess_test.txt', 'r') as f:
                print("\nContents of analyzer_subprocess_test.txt:")
                print(f.read())
        else:
            print("\nFAILURE: No file was created, code might be running in a sandbox")
        
        # Print test result
        print("\nTest result status:", result.get("status"))
        print("Test result score:", result.get("score"))
        
    except Exception as e:
        print(f"Error running test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 