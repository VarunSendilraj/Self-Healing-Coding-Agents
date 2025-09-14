#!/usr/bin/env python
"""
Test script to demonstrate self-healing behavior with a challenging task.
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
        original_init(self, runner_path)
    
    # Apply the patch
    CodeAnalyzer.__init__ = patched_init

def create_challenging_task():
    """Create a more challenging task to trigger self-healing."""
    task = {
        "id": "self_healing_demo_task",
        "description": """You need to implement a function that solves a complex algorithmic problem.

Given a string s and a pattern p, implement regular expression matching with support for '.' and '*' where:
- '.' Matches any single character.
- '*' Matches zero or more of the preceding element.

The matching should cover the entire input string (not partial).

Example 1:
Input: s = "aa", p = "a"
Output: false
Explanation: "a" does not match the entire string "aa".

Example 2:
Input: s = "aa", p = "a*"
Output: true
Explanation: '*' means zero or more of the preceding element, 'a'. Therefore, by repeating 'a' once, it becomes "aa".

Example 3:
Input: s = "ab", p = ".*"
Output: true
Explanation: ".*" means "zero or more (*) of any character (.)".

Constraints:
- 1 <= s.length <= 20
- 1 <= p.length <= 30
- s contains only lowercase English letters.
- p contains only lowercase English letters, '.', and '*'.
""",
        "initial_executor_prompt": """Implement a function called `isMatch` that takes two strings s and p and returns True if s matches the pattern p, False otherwise. Handle the '.' and '*' characters as described in the problem."""
    }
    return task

def create_medium_task():
    """Create a medium difficulty task that's more likely to trigger issues."""
    task = {
        "id": "medium_task_demo",
        "description": """Implement a function to find the longest palindromic substring in a given string.

A palindrome is a string that reads the same backward as forward.

Example 1:
Input: s = "babad"
Output: "bab"
Note: "aba" is also a valid answer.

Example 2:
Input: s = "cbbd"
Output: "bb"

Example 3:
Input: s = "a"
Output: "a"

Example 4:
Input: s = "ac"
Output: "a"

Constraints:
- 1 <= s.length <= 1000
- s consist of only digits and English letters.
""",
        "initial_executor_prompt": "Implement a function called `longestPalindrome` that takes a string s and returns the longest palindromic substring."
    }
    return task

def main():
    """Main test function."""
    try:
        # Apply the patch for CodeAnalyzer
        patch_code_analyzer()
        
        # Import the enhanced harness integration module
        from self_healing_agents.evaluation.enhanced_harness_integration import main as run_enhanced_harness
        
        # Create test tasks - you can uncomment different ones to test different difficulty levels
        test_task = create_challenging_task()  # Very challenging regex matching
        # test_task = create_medium_task()  # Medium difficulty palindrome
        
        # Create a temporary file with the test task
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump([test_task], f, indent=2)
            temp_file = f.name
        
        print(f"Created test task file: {temp_file}")
        print(f"Test task: {test_task['description'][:100]}...")
        
        try:
            # Run the enhanced harness with our test task
            print("\n" + "="*60)
            print("🚀 RUNNING ENHANCED HARNESS WITH DUMBIFIED PROMPTS")
            print("🎯 Goal: Trigger self-healing behavior with suboptimal initial code")
            print("="*60)
            
            # Override sys.argv to pass the test file
            original_argv = sys.argv
            sys.argv = ['enhanced_harness_integration.py', temp_file]
            
            # Run the enhanced harness
            run_enhanced_harness()
            
        finally:
            # Restore original sys.argv
            sys.argv = original_argv
            
            # Clean up the temporary file
            if os.path.exists(temp_file):
                os.unlink(temp_file)
                print(f"Cleaned up temporary file: {temp_file}")
        
        print("\n" + "="*60)
        print("🎉 TEST COMPLETED")
        print("="*60)
        print("Check the enhanced_evaluation_harness.log file to see:")
        print("1. Initial code generation (likely suboptimal)")
        print("2. Test case failures")
        print("3. Direct fix attempts")
        print("4. Self-healing iterations")
        print("5. Final successful code")
        
    except Exception as e:
        print(f"Error in test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 