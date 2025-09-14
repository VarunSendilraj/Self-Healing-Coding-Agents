#!/usr/bin/env python
"""
Demonstration script to show how self-healing would work with broken initial code.
This manually simulates what happens when the system needs to fix errors.
"""

import os
import sys

# Add the src directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

def patch_code_analyzer():
    """Patch the CodeAnalyzer class to use the correct runner path."""
    from src.code_analyzer import CodeAnalyzer
    
    # Store the original __init__ method
    original_init = CodeAnalyzer.__init__
    
    # Define a new __init__ method that uses the correct runner path
    def patched_init(self, runner_path=None):
        if runner_path is None:
            runner_path = os.path.join(current_dir, "src", "code_runner.py")
        original_init(self, runner_path)
    
    # Apply the patch
    CodeAnalyzer.__init__ = patched_init

def main():
    """Demonstrate what self-healing looks like."""
    patch_code_analyzer()
    
    from self_healing_agents.evaluation.code_analyzer_integration import CodeAnalyzerCritic
    from self_healing_agents.llm_service import LLMService
    from self_healing_agents.prompts import CRITIC_TEST_GENERATION_SYSTEM_PROMPT
    
    # Initialize LLM service
    llm_service = LLMService(
        provider="deepseek",
        model_name="deepseek-coder", 
        api_key=os.getenv("DEEPSEEK_API_KEY", "test")
    )
    
    # Initialize critic with LLM service
    critic = CodeAnalyzerCritic(llm_service=llm_service)
    
    print("🧪 SELF-HEALING DEMONSTRATION")
    print("="*50)
    
    # Task description
    task_description = """
    Implement a function to check if a number is prime.
    A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself.
    
    Examples:
    Input: 2 → Output: True
    Input: 3 → Output: True  
    Input: 4 → Output: False
    Input: 17 → Output: True
    Input: 25 → Output: False
    """
    
    # Simulate broken initial code (this is what a dumb prompt might generate)
    broken_code = """
def isPrime(n):
    if n < 2:
        return False
    for i in range(2, n):  # This is inefficient and will timeout for large numbers
        if n % i == 0:
            return False
    return True
    """
    
    print("📝 INITIAL (BROKEN) CODE:")
    print(broken_code)
    print()
    
    # Run the critic on broken code
    print("🔍 RUNNING CRITIC ON BROKEN CODE...")
    critique = critic.run(broken_code, task_description)
    
    print(f"📊 CRITIQUE RESULTS:")
    print(f"Status: {critique['status']}")
    print(f"Score: {critique['score']}")
    print(f"Summary: {critique['summary']}")
    print()
    
    if critique['test_results']:
        print("🧪 TEST CASE RESULTS:")
        for i, result in enumerate(critique['test_results'], 1):
            status = "✅ PASS" if result['passed'] else "❌ FAIL"
            print(f"  Test {i}: {status}")
            print(f"    Input: {result['input']}")
            print(f"    Expected: {result['expected']}")
            print(f"    Actual: {result['actual']}")
            print(f"    Time: {result['execution_time_ms']}ms")
            if result['stdout']:
                print(f"    Stdout: {result['stdout'].strip()}")
            if result['error']:
                print(f"    Error: {result['error']}")
            print()
    
    # Simulate what self-healing would do: generate improved code
    print("🔧 SIMULATING SELF-HEALING PROCESS...")
    print("The system would now:")
    print("1. Analyze the test failures")
    print("2. Generate improved code using the error feedback")
    print("3. Re-run tests until all pass")
    print()
    
    # Show what fixed code might look like
    fixed_code = """
def isPrime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    # Only check odd divisors up to sqrt(n)
    import math
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True
    """
    
    print("🎯 FIXED CODE (what self-healing would generate):")
    print(fixed_code)
    print()
    
    # Test the fixed code
    print("🔍 RUNNING CRITIC ON FIXED CODE...")
    fixed_critique = critic.run(fixed_code, task_description)
    
    print(f"📊 FIXED CODE RESULTS:")
    print(f"Status: {fixed_critique['status']}")
    print(f"Score: {fixed_critique['score']}")
    print(f"Summary: {fixed_critique['summary']}")
    print()
    
    print("🎉 SELF-HEALING DEMONSTRATION COMPLETE!")
    print("This shows how the system detects failures, generates fixes, and validates improvements.")

if __name__ == "__main__":
    main() 