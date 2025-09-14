#!/usr/bin/env python
"""
Test script for the Enhanced Multi-Agent Self-Healing Harness

This script demonstrates the new system that can distinguish between planning
and execution failures and apply targeted self-healing to the appropriate agent.
"""

import sys
import os
import json
from typing import Dict, Any

# Add the src directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Import the enhanced multi-agent harness
from self_healing_agents.evaluation.enhanced_multi_agent_harness import run_enhanced_multi_agent_task
from self_healing_agents.agents import Planner, Executor, Critic
from self_healing_agents.llm_service import LLMService
from self_healing_agents.prompts import PLANNER_SYSTEM_PROMPT, ULTRA_BUGGY_PROMPT

def create_test_tasks() -> list[Dict[str, Any]]:
    """Create test tasks that should trigger different types of failures."""
    
    return [
        {
            "id": "planning_failure_task",
            "description": "Write a function that finds the longest common subsequence between two strings using dynamic programming. The function should return both the length and the actual subsequence string.",
            "initial_executor_prompt": ULTRA_BUGGY_PROMPT,  # Use buggy prompt to trigger failures
            "expected_failure_type": "PLANNING_FAILURE"  # This complex task should expose planning issues
        },
        {
            "id": "execution_failure_task", 
            "description": "Write a simple function that adds two numbers and returns the result.",
            "initial_executor_prompt": ULTRA_BUGGY_PROMPT,  # Use buggy prompt to trigger failures
            "expected_failure_type": "EXECUTION_FAILURE"  # Simple task, likely execution issues
        },
        {
            "id": "mixed_failure_task",
            "description": "Create a function that implements a binary search algorithm on a sorted list. Handle edge cases like empty lists and elements not found.",
            "initial_executor_prompt": ULTRA_BUGGY_PROMPT,  # Use buggy prompt to trigger failures
            "expected_failure_type": "MIXED_FAILURE"  # Medium complexity, could be either
        }
    ]

def run_enhanced_multi_agent_demo():
    """Run the enhanced multi-agent self-healing demonstration."""
    
    print("=" * 80)
    print("🚀 ENHANCED MULTI-AGENT SELF-HEALING SYSTEM DEMONSTRATION")
    print("=" * 80)
    print()
    print("This demo showcases the new system that can:")
    print("✅ Classify failures as planning vs execution issues")
    print("✅ Apply targeted self-healing to the appropriate agent")
    print("✅ Validate plans before execution")
    print("✅ Track healing effectiveness across multiple iterations")
    print("✅ Maintain the original direct fix + self-healing workflow")
    print()
    
    # Initialize LLM service (using mock for demo)
    class MockLLMService(LLMService):
        def __init__(self):
            super().__init__(provider="openai", model_name="mock-model", api_key="mock_key")
            self.call_count = 0
            
        def invoke(self, messages, expect_json=False):
            self.call_count += 1
            last_message = messages[-1]["content"].lower()
            
            # Mock planner responses
            if "planner" in messages[0]["content"].lower() or "planning" in last_message:
                if expect_json:
                    if "longest common subsequence" in last_message:
                        # Intentionally vague plan to trigger planning failure
                        return {
                            "steps": [
                                "Create a function",
                                "Use dynamic programming", 
                                "Return the result"
                            ],
                            "approach": "Dynamic programming"
                        }
                    elif "binary search" in last_message:
                        return {
                            "steps": [
                                "Define function with parameters",
                                "Implement binary search logic",
                                "Handle edge cases",
                                "Return appropriate result"
                            ],
                            "approach": "Binary search algorithm",
                            "requirements": ["Sorted input list"]
                        }
                    else:  # Simple add function
                        return {
                            "steps": [
                                "Define function with two parameters",
                                "Add the parameters",
                                "Return the sum"
                            ],
                            "approach": "Simple arithmetic"
                        }
                        
            # Mock planner healer responses
            elif "planning self-healer" in messages[0]["content"].lower():
                if expect_json:
                    return {
                        "steps": [
                            "Define function lcs(str1, str2) with proper parameters",
                            "Create 2D DP table with dimensions (len(str1)+1) x (len(str2)+1)",
                            "Initialize base cases: dp[i][0] = 0, dp[0][j] = 0",
                            "Fill DP table: if str1[i-1] == str2[j-1]: dp[i][j] = dp[i-1][j-1] + 1",
                            "Else: dp[i][j] = max(dp[i-1][j], dp[i][j-1])",
                            "Backtrack to construct actual subsequence string",
                            "Return tuple (length, subsequence_string)"
                        ],
                        "requirements": ["No external imports needed"],
                        "approach": "Bottom-up dynamic programming with backtracking",
                        "improvements_made": [
                            "Added specific DP table dimensions",
                            "Detailed base case initialization", 
                            "Explicit recurrence relation",
                            "Added backtracking for subsequence construction"
                        ]
                    }
                    
            # Mock executor responses (intentionally buggy)
            elif "executor" in messages[0]["content"].lower() or "python code" in last_message:
                if "longest common subsequence" in last_message:
                    # Buggy LCS implementation
                    return """def lcs(s1, s2):
    # Missing proper DP implementation
    result = ""
    for i in s1:
        if i in s2:
            result += i
    return len(result)  # Wrong: should return both length and string"""
                    
                elif "binary search" in last_message:
                    # Buggy binary search
                    return """def binary_search(arr, target):
    left, right = 0, len(arr)  # Bug: should be len(arr) - 1
    while left < right:  # Bug: should be left <= right
        mid = (left + right) / 2  # Bug: should use // for integer division
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1"""
            
                else:  # Simple add function
                    return """def add_numbers(a, b):
    return a + b  # This should work fine"""
                    
            # Mock executor healer responses
            elif "execution self-healer" in messages[0]["content"].lower():
                if "binary search" in last_message:
                    return """def binary_search(arr, target):
    if not arr:  # Handle empty list
        return -1
    
    left, right = 0, len(arr) - 1  # Fixed: proper right boundary
    
    while left <= right:  # Fixed: proper condition
        mid = (left + right) // 2  # Fixed: integer division
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
            
    return -1  # Element not found"""
    
                elif "longest common subsequence" in last_message:
                    return """def lcs(str1, str2):
    m, n = len(str1), len(str2)
    
    # Create DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    # Backtrack to find actual subsequence
    lcs_str = ""
    i, j = m, n
    while i > 0 and j > 0:
        if str1[i-1] == str2[j-1]:
            lcs_str = str1[i-1] + lcs_str
            i -= 1
            j -= 1
        elif dp[i-1][j] > dp[i][j-1]:
            i -= 1
        else:
            j -= 1
    
    return dp[m][n], lcs_str"""
                    
            # Mock critic responses
            elif "critic" in messages[0]["content"].lower():
                if expect_json:
                    # Simulate test generation
                    if "longest common subsequence" in last_message:
                        return {
                            "function_to_test": "lcs",
                            "test_cases": [
                                {
                                    "test_case_name": "basic_lcs",
                                    "inputs": {"str1": "ABCDGH", "str2": "AEDFHR"},
                                    "expected_output": (3, "ADH")
                                }
                            ]
                        }
                    elif "binary_search" in last_message:
                        return {
                            "function_to_test": "binary_search", 
                            "test_cases": [
                                {
                                    "test_case_name": "found_element",
                                    "inputs": {"arr": [1, 3, 5, 7, 9], "target": 5},
                                    "expected_output": 2
                                },
                                {
                                    "test_case_name": "not_found",
                                    "inputs": {"arr": [1, 3, 5, 7, 9], "target": 4},
                                    "expected_output": -1
                                }
                            ]
                        }
                    else:  # add function
                        return {
                            "function_to_test": "add_numbers",
                            "test_cases": [
                                {
                                    "test_case_name": "basic_addition",
                                    "inputs": {"a": 2, "b": 3},
                                    "expected_output": 5
                                }
                            ]
                        }
                else:
                    # Return evaluation results
                    if "def add_numbers" in last_message and "return a + b" in last_message:
                        return {
                            "status": "SUCCESS",
                            "score": 1.0,
                            "summary": "All tests passed"
                        }
                    else:
                        return {
                            "status": "FAILURE_LOGIC", 
                            "score": 0.2,
                            "execution_error_type": "LogicError",
                            "execution_error_message": "Function implementation has logical errors",
                            "num_tests_total": 2,
                            "num_tests_passed": 0,
                            "num_tests_failed": 2,
                            "failed_test_details": [
                                {
                                    "name": "test_case_1",
                                    "error_message": "Expected different output",
                                    "expected_output_spec": "Correct result",
                                    "actual_output": "Incorrect result"
                                }
                            ],
                            "summary": "Implementation has bugs"
                        }
            
            return "Mock response"
    
    # Initialize agents
    llm_service = MockLLMService()
    planner = Planner("TestPlanner", llm_service, PLANNER_SYSTEM_PROMPT)
    executor = Executor("TestExecutor", llm_service)
    critic = Critic("TestCritic", llm_service)
    
    # Run test tasks
    test_tasks = create_test_tasks()
    results = []
    
    for i, task in enumerate(test_tasks, 1):
        print(f"\n{'='*60}")
        print(f"🧪 TEST CASE {i}: {task['id'].upper()}")
        print(f"{'='*60}")
        print(f"📝 Task: {task['description'][:80]}...")
        print(f"🎯 Expected Failure Type: {task['expected_failure_type']}")
        print()
        
        # Run the enhanced multi-agent task
        result = run_enhanced_multi_agent_task(
            task_definition=task,
            planner=planner,
            executor=executor, 
            critic=critic,
            llm_service_instance=llm_service,
            max_healing_iterations=3
        )
        
        results.append(result)
        
        # Print summary
        print(f"\n📊 TASK SUMMARY:")
        print(f"   Status: {result['final_status']}")
        print(f"   Final Score: {result['final_score']:.2f}")
        print(f"   Total Healing Iterations: {result['total_healing_iterations']}")
        print(f"   Planner Healings: {result['healing_breakdown']['planner_healings']}")
        print(f"   Executor Healings: {result['healing_breakdown']['executor_healings']}")
        print(f"   Direct Fix Attempts: {result['healing_breakdown']['direct_fix_attempts']}")
        
        # Show classification history
        if result['classification_history']:
            print(f"\n🔍 FAILURE CLASSIFICATIONS:")
            for j, classification in enumerate(result['classification_history'], 1):
                print(f"   Iteration {j}: {classification['failure_type'].value} "
                      f"(confidence: {classification['confidence']:.2f}, "
                      f"target: {classification['recommended_healing_target']})")
    
    # Overall summary
    print(f"\n{'='*80}")
    print("📈 OVERALL DEMONSTRATION SUMMARY")
    print(f"{'='*80}")
    
    total_tasks = len(results)
    successful_tasks = sum(1 for r in results if "SUCCESS" in r['final_status'])
    total_planner_healings = sum(r['healing_breakdown']['planner_healings'] for r in results)
    total_executor_healings = sum(r['healing_breakdown']['executor_healings'] for r in results)
    total_classifications = sum(len(r['classification_history']) for r in results)
    
    print(f"✅ Tasks Completed: {total_tasks}")
    print(f"🎯 Successful Tasks: {successful_tasks}/{total_tasks}")
    print(f"🔧 Total Planner Healings: {total_planner_healings}")
    print(f"🔧 Total Executor Healings: {total_executor_healings}")
    print(f"🔍 Total Failure Classifications: {total_classifications}")
    print(f"📞 Total LLM Calls: {llm_service.call_count}")
    
    print(f"\n🎉 Enhanced Multi-Agent Self-Healing System demonstrated successfully!")
    print(f"   The system can now intelligently choose between planner and executor healing")
    print(f"   based on failure classification, leading to more targeted and effective healing.")
    
    return results

if __name__ == "__main__":
    run_enhanced_multi_agent_demo() 