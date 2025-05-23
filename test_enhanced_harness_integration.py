#!/usr/bin/env python
"""
Test script to verify that enhanced_harness_integration.py correctly integrates the CodeAnalyzer.
"""

import os
import sys
import tempfile
import json

# Add the src directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

def patch_code_analyzer():
    """Patch the CodeAnalyzer class to use the correct runner path."""
    from code_analyzer import CodeAnalyzer
    
    # Store the original __init__ method
    original_init = CodeAnalyzer.__init__
    
    # Define a new __init__ method that uses the correct runner path
    def patched_init(self, runner_path=None):
        if runner_path is None:
            runner_path = os.path.join(current_dir, "code_runner.py")
            print(f"Using patched runner path: {runner_path}")
        original_init(self, runner_path)
    
    # Apply the patch
    CodeAnalyzer.__init__ = patched_init
    print("CodeAnalyzer.__init__ patched to use the correct runner path")

def main():
    """Main test function."""
    try:
        # Patch the CodeAnalyzer to use the correct runner path
        patch_code_analyzer()
        
        # Import required modules
        from code_analyzer import CodeAnalyzer
        from self_healing_agents.evaluation import enhanced_harness
        from self_healing_agents.evaluation.code_analyzer_integration import integrate_code_analyzer_with_harness, CodeAnalyzerCritic
        
        # Create a test file to show we are in the parent process
        with open('parent_process.txt', 'w') as f:
            f.write(f"Parent process ID: {os.getpid()}\n")
            f.write(f"This file was created by the parent process\n")
        
        print(f"Parent process ID: {os.getpid()}")
        
        # Create a test code that creates a file
        test_code = """
def test_function():
    import os
    
    # Get the current process ID
    pid = os.getpid()
    
    # Create a file with the process ID
    with open('direct_integration_test.txt', 'w') as f:
        f.write(f"Process ID: {pid}\\n")
        f.write(f"This file was created by direct integration test\\n")
    
    return 42
"""
        
        # First, test direct execution with CodeAnalyzer
        print("\n=== Testing direct execution with CodeAnalyzer ===")
        analyzer = CodeAnalyzer()
        test_wrapper = test_code + "\n\n# Test execution\nresult = test_function()\nprint(repr(result))\n"
        result = analyzer.run_code(test_wrapper)
        print("CodeAnalyzer.run_code result:", result)
        
        # Check if the file was created
        if os.path.exists('direct_integration_test.txt'):
            print("\nSUCCESS: File was created by direct CodeAnalyzer execution")
            with open('direct_integration_test.txt', 'r') as f:
                print("\nContents of direct_integration_test.txt:")
                print(f.read())
        else:
            print("\nFAILURE: No file was created by direct CodeAnalyzer execution")
        
        # Now test with CodeAnalyzerCritic
        print("\n=== Testing execution with CodeAnalyzerCritic ===")
        critic = CodeAnalyzerCritic()
        result = critic.run(test_code, "Test function")
        print("CodeAnalyzerCritic.run result:", result.get("status"))
        
        # Check if the file was created
        if os.path.exists('direct_integration_test.txt'):
            print("\nSUCCESS: File was created by CodeAnalyzerCritic execution")
            with open('direct_integration_test.txt', 'r') as f:
                print("\nContents of direct_integration_test.txt:")
                print(f.read())
        else:
            print("\nFAILURE: No file was created by CodeAnalyzerCritic execution")
        
        # Now test the integration with enhanced_harness
        print("\n=== Testing integration with enhanced_harness ===")
        print(f"Original Critic class: {enhanced_harness.Critic}")
        
        # Patch the enhanced_harness module
        integrate_code_analyzer_with_harness(enhanced_harness)
        print("Enhanced harness patched with code analyzer integration")
        print(f"Patched Critic class: {enhanced_harness.Critic}")
        
        # Create a simple instance of the patched Critic
        critic = enhanced_harness.Critic("TestCritic", None)
        
        # Check if the critic has the code_analyzer_critic attribute
        has_analyzer = hasattr(critic, 'code_analyzer_critic')
        print(f"Does critic have code_analyzer_critic attribute? {has_analyzer}")
        
        if has_analyzer:
            print("Integration successful! The enhanced harness is using the CodeAnalyzerCritic.")
            # Try running code through the integrated critic
            print("\n=== Running code through integrated critic ===")
            result = critic.code_analyzer_critic.run(test_code, "Test function")
            print("Integrated critic result:", result.get("status"))
            
            # Check if the file was created
            if os.path.exists('direct_integration_test.txt'):
                print("\nSUCCESS: File was created by integrated critic execution")
                with open('direct_integration_test.txt', 'r') as f:
                    print("\nContents of direct_integration_test.txt:")
                    print(f.read())
            else:
                print("\nFAILURE: No file was created by integrated critic execution")
        else:
            print("Integration failed! The patched Critic does not have the code_analyzer_critic attribute.")
        
    except Exception as e:
        print(f"Error running test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 