#!/usr/bin/env python3
"""
Test script for the enhanced self-healing flow with direct error-fixing attempt.
This script tests the new agent system enhancement where it first tries a direct
error fix before engaging the Self-Healing Module.
"""

import sys
import os
from dotenv import load_dotenv
import json
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

# Import from the self_healing_agents module
from src.self_healing_agents.orchestration import Orchestrator
from src.self_healing_agents.llm_service import LLMService

# Test cases representing different types of errors
TEST_CASES = [
    {
        "name": "syntax_error_test",
        "description": "Write a Python function called 'factorial' that calculates the factorial of a non-negative integer n. If n is 0, return 1.",
    },
    {
        "name": "runtime_error_test",
        "description": "Write a function called 'find_max_subarray' that takes a list of integers and returns the maximum sum of any contiguous subarray."
    },
    {
        "name": "logical_error_test",
        "description": "Write a function called 'is_prime' that takes a positive integer and returns True if it's a prime number, and False otherwise."
    }
]

def setup_llm_service():
    """Set up and return an LLM service instance."""
    # Get API key from environment variables
    api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    provider = "deepseek" if os.getenv("DEEPSEEK_API_KEY") else "openai"
    
    if not api_key:
        print("Error: No API key found. Please set DEEPSEEK_API_KEY or OPENAI_API_KEY in your .env file.")
        sys.exit(1)
    
    return LLMService(api_key=api_key, provider=provider)

def run_test(test_case, llm_service):
    """Run a single test case and return the results."""
    print(f"\n{'='*80}")
    print(f"Testing: {test_case['name']}")
    print(f"Description: {test_case['description']}")
    print(f"{'='*80}")
    
    # Create orchestrator with the test case description
    orchestrator = Orchestrator(user_request=test_case["description"])
    
    # Inject the LLM service into the orchestrator's agent factory
    # This assumes the agent factory has a method to set the LLM service
    orchestrator.agents_factory.set_llm_service(llm_service)
    
    # Run the enhanced flow that includes direct fix attempt
    start_time = datetime.now()
    result = orchestrator.run_enhanced_flow()
    end_time = datetime.now()
    
    # Calculate execution time
    execution_time = (end_time - start_time).total_seconds()
    
    # Add execution details to the result
    result["execution_time"] = execution_time
    result["test_case"] = test_case["name"]
    result["description"] = test_case["description"]
    
    return result

def main():
    """Main entry point for the test script."""
    # Set up the LLM service
    llm_service = setup_llm_service()
    
    # Results array to store outcomes of all tests
    results = []
    
    # Run each test case
    for test_case in TEST_CASES:
        try:
            result = run_test(test_case, llm_service)
            results.append(result)
            
            # Save individual test result to a log file
            log_filename = f"{test_case['name']}.log"
            with open(log_filename, 'w') as f:
                json.dump(result, f, indent=2)
            
            print(f"\nTest {test_case['name']} completed. Results saved to {log_filename}")
            
            # Print summary of this test
            print(f"\nTest Summary:")
            print(f"  Status: {result.get('status', 'UNKNOWN')}")
            print(f"  Best Source: {result.get('best_source', 'N/A')}")
            print(f"  Best Score: {result.get('best_score', 0.0)}")
            print(f"  Execution Time: {result.get('execution_time', 0.0):.2f} seconds")
            
        except Exception as e:
            print(f"Error running test {test_case['name']}: {e}")
    
    # Save all results to a combined log file
    combined_log_filename = "direct_fix_test_results.log"
    with open(combined_log_filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nAll tests completed. Combined results saved to {combined_log_filename}")
    
    # Print overall summary
    print("\nOverall Summary:")
    for result in results:
        print(f"  {result.get('test_case', 'Unknown Test')}:")
        print(f"    Status: {result.get('status', 'UNKNOWN')}")
        print(f"    Best Source: {result.get('best_source', 'N/A')}")
        print(f"    Best Score: {result.get('best_score', 0.0)}")
        print(f"    Execution Time: {result.get('execution_time', 0.0):.2f} seconds")

if __name__ == "__main__":
    main()
