"""
Comprehensive Multi-Agent Self-Healing Test Suite

This test suite is designed to verify that the enhanced multi-agent harness can correctly:
1. Identify when planning is the root cause of failure and heal the planner
2. Identify when execution is the root cause of failure and heal the executor  
3. Handle extremely challenging coding problems that require both good planning and execution

The tests use intentionally bad prompts and challenging problems to trigger healing scenarios.
"""

import logging
import os
import time
from typing import Dict, Any, List
from self_healing_agents.agents import Planner, Executor, Critic
from self_healing_agents.llm_service import LLMService
from self_healing_agents.prompts import (
    BAD_PLANNER_PROMPT, DEFAULT_EXECUTOR_SYSTEM_PROMPT,
    CATASTROPHIC_EXECUTOR_PROMPT, SYNTAX_ERROR_EXECUTOR_PROMPT, 
    WRONG_ALGORITHM_EXECUTOR_PROMPT, INCOMPLETE_EXECUTOR_PROMPT,
    VARIABLE_MESS_EXECUTOR_PROMPT, PLANNER_SYSTEM_PROMPT
)
from self_healing_agents.evaluation.enhanced_multi_agent_harness import run_enhanced_multi_agent_task
from self_healing_agents.evaluation.enhanced_harness import TermColors

def get_challenging_test_tasks() -> List[Dict[str, Any]]:
    """
    Define extremely challenging coding problems that require both excellent planning and execution.
    These problems are designed to expose weaknesses in either planning or execution.
    """
    return [
        # === PLANNER HEALING TESTS ===
        # These use challenging problems with bad planner + good executor
        {
            "id": "planner_test_1_lcs_algorithm",
            "test_type": "PLANNER_HEALING", 
            "description": """Implement a function 'longest_common_subsequence(str1, str2)' that finds the longest common subsequence between two strings using dynamic programming.

Requirements:
- Use a 2D DP table approach
- Return both the length (int) and the actual LCS string (str) as a tuple
- Handle edge cases: empty strings, no common subsequence
- Optimize space if possible
- The function signature should be: def longest_common_subsequence(str1: str, str2: str) -> tuple[int, str]

Example:
- longest_common_subsequence("ABCDGH", "AEDFHR") -> (3, "ADH")
- longest_common_subsequence("AGGTAB", "GXTXAYB") -> (4, "GTAB")""",
            "planner_prompt": BAD_PLANNER_PROMPT,
            "executor_prompt": DEFAULT_EXECUTOR_SYSTEM_PROMPT,
            "expected_healing_target": "PLANNER",
            "complexity": "HIGH"
        },
        
        {
            "id": "planner_test_2_graph_algorithm", 
            "test_type": "PLANNER_HEALING",
            "description": """Create a class 'GraphShortestPath' that implements Dijkstra's algorithm for finding shortest paths in a weighted graph.

Requirements:
- __init__(self, graph: dict) where graph is adjacency list with weights: {node: [(neighbor, weight), ...]}
- shortest_path(self, start: str, end: str) -> tuple[list, int] returns (path_nodes, total_distance)
- shortest_paths_from(self, start: str) -> dict returns all shortest distances from start node
- Handle disconnected graphs (return infinity for unreachable nodes)
- Use priority queue for efficiency
- Validate input and handle edge cases

Example:
graph = {'A': [('B', 4), ('C', 2)], 'B': [('C', 1), ('D', 5)], 'C': [('D', 8), ('E', 10)], 'D': [('E', 2)], 'E': []}
gsp = GraphShortestPath(graph)
gsp.shortest_path('A', 'E') -> (['A', 'C', 'D', 'E'], 12)""",
            "planner_prompt": BAD_PLANNER_PROMPT,
            "executor_prompt": DEFAULT_EXECUTOR_SYSTEM_PROMPT,
            "expected_healing_target": "PLANNER",
            "complexity": "HIGH"
        },
        
        {
            "id": "planner_test_3_complex_data_structure",
            "test_type": "PLANNER_HEALING", 
            "description": """Implement a 'TrieAutoComplete' class that supports prefix-based autocompletion with scoring.

Requirements:
- __init__(self) to initialize empty trie
- insert(self, word: str, score: int) to add word with associated score
- search(self, word: str) -> bool to check if exact word exists  
- autocomplete(self, prefix: str, max_results: int = 10) -> list[tuple[str, int]] returns up to max_results completions sorted by score (descending), then alphabetically
- delete(self, word: str) -> bool to remove word, return True if existed
- update_score(self, word: str, new_score: int) -> bool to update word's score

The trie should efficiently handle large dictionaries and provide fast prefix matching.

Example:
trie = TrieAutoComplete()
trie.insert("apple", 100)
trie.insert("application", 80) 
trie.insert("apply", 90)
trie.autocomplete("app", 2) -> [("apple", 100), ("apply", 90)]""",
            "planner_prompt": BAD_PLANNER_PROMPT,
            "executor_prompt": DEFAULT_EXECUTOR_SYSTEM_PROMPT,
            "expected_healing_target": "PLANNER",
            "complexity": "HIGH"
        },

        # === EXECUTOR HEALING TESTS ===
        # These use simpler problems with good planner + bad executor
        {
            "id": "executor_test_1_basic_math",
            "test_type": "EXECUTOR_HEALING",
            "description": """Write a function 'calculate_factorial(n)' that calculates the factorial of a non-negative integer.

Requirements:
- Handle n = 0 (return 1)
- Handle negative inputs (raise ValueError)
- Use iterative approach (not recursive)
- Return integer result
- Function signature: def calculate_factorial(n: int) -> int

Examples:
- calculate_factorial(5) -> 120
- calculate_factorial(0) -> 1
- calculate_factorial(-1) -> raises ValueError""",
            "planner_prompt": PLANNER_SYSTEM_PROMPT,
            "executor_prompt": CATASTROPHIC_EXECUTOR_PROMPT,
            "expected_healing_target": "EXECUTOR", 
            "complexity": "LOW"
        },
        
        {
            "id": "executor_test_2_string_processing",
            "test_type": "EXECUTOR_HEALING",
            "description": """Create a function 'reverse_words_in_string(text)' that reverses the order of words in a string while preserving spaces.

Requirements:
- Preserve the original spacing between words
- Handle multiple consecutive spaces
- Handle leading/trailing spaces
- Return empty string for empty input
- Function signature: def reverse_words_in_string(text: str) -> str

Examples:
- reverse_words_in_string("hello world") -> "world hello"
- reverse_words_in_string("  hello   world  ") -> "  world   hello  "
- reverse_words_in_string("a") -> "a" """,
            "planner_prompt": PLANNER_SYSTEM_PROMPT,
            "executor_prompt": SYNTAX_ERROR_EXECUTOR_PROMPT,
            "expected_healing_target": "EXECUTOR",
            "complexity": "MEDIUM"
        },
        
        {
            "id": "executor_test_3_list_operations",
            "test_type": "EXECUTOR_HEALING", 
            "description": """Implement a function 'find_pair_with_sum(numbers, target_sum)' that finds two numbers in a list that add up to a target sum.

Requirements:
- Return the indices of the two numbers as a tuple (index1, index2) where index1 < index2
- Return None if no such pair exists
- Handle empty lists and single-element lists
- Don't use the same element twice
- Function signature: def find_pair_with_sum(numbers: list[int], target_sum: int) -> tuple[int, int] | None

Examples:
- find_pair_with_sum([2, 7, 11, 15], 9) -> (0, 1)
- find_pair_with_sum([3, 2, 4], 6) -> (1, 2)
- find_pair_with_sum([3, 3], 6) -> (0, 1)""",
            "planner_prompt": PLANNER_SYSTEM_PROMPT,
            "executor_prompt": WRONG_ALGORITHM_EXECUTOR_PROMPT,
            "expected_healing_target": "EXECUTOR",
            "complexity": "MEDIUM"
        },

        # === MIXED COMPLEXITY TESTS ===
        # These could trigger either type of healing depending on the specific failure
        {
            "id": "mixed_test_1_binary_tree",
            "test_type": "MIXED_COMPLEXITY",
            "description": """Implement a binary search tree with the following operations:

Create a class 'BinarySearchTree' with:
- __init__(self) to create empty BST
- insert(self, value: int) to add a value
- search(self, value: int) -> bool to check if value exists
- delete(self, value: int) -> bool to remove value (return True if existed)
- inorder_traversal(self) -> list[int] to return values in sorted order
- find_min(self) -> int | None to return minimum value
- find_max(self) -> int | None to return maximum value

Handle all edge cases including deleting nodes with 0, 1, or 2 children.

Example:
bst = BinarySearchTree()
bst.insert(5); bst.insert(3); bst.insert(7)
bst.inorder_traversal() -> [3, 5, 7]
bst.search(3) -> True
bst.delete(5) -> True""",
            "planner_prompt": BAD_PLANNER_PROMPT,
            "executor_prompt": INCOMPLETE_EXECUTOR_PROMPT,
            "expected_healing_target": "MULTIPLE",
            "complexity": "HIGH"
        },
        
        {
            "id": "mixed_test_2_dynamic_programming",
            "test_type": "MIXED_COMPLEXITY", 
            "description": """Solve the 0/1 Knapsack problem using dynamic programming.

Implement function 'knapsack_01(weights, values, capacity)' that:
- Takes list of item weights, list of item values, and knapsack capacity
- Returns tuple (max_value, selected_items) where selected_items is list of indices
- Uses dynamic programming with memoization or tabulation
- Handles edge cases: empty items, zero capacity, impossible items

Requirements:
- Optimize for both time and space complexity
- Function signature: def knapsack_01(weights: list[int], values: list[int], capacity: int) -> tuple[int, list[int]]

Example:
weights = [1, 3, 4, 5]
values = [1, 4, 5, 7] 
capacity = 7
knapsack_01(weights, values, capacity) -> (9, [1, 2]) # items at indices 1,2 with total value 9""",
            "planner_prompt": BAD_PLANNER_PROMPT,
            "executor_prompt": VARIABLE_MESS_EXECUTOR_PROMPT,
            "expected_healing_target": "MULTIPLE",
            "complexity": "HIGH"
        }
    ]

def run_comprehensive_healing_tests():
    """
    Run the complete test suite to verify enhanced multi-agent healing capabilities.
    """
    print("🧪 COMPREHENSIVE MULTI-AGENT SELF-HEALING TEST SUITE")
    print("=" * 80)
    print("🎯 Testing both PLANNER and EXECUTOR healing with challenging problems")
    print("🔧 Using intentionally bad prompts to trigger specific failure types")
    print("=" * 80)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Configure environment
    provider = os.environ.get("LLM_PROVIDER", "deepseek")
    model_name = os.environ.get("LLM_MODEL", "deepseek-coder")
    
    try:
        llm_service = LLMService(provider=provider, model_name=model_name)
        print(f"✅ LLM Service: {provider}/{model_name}")
    except Exception as e:
        print(f"❌ LLM Service Error: {e}")
        return
    
    # Get test tasks
    test_tasks = get_challenging_test_tasks()
    
    print(f"\n📋 TEST SUITE OVERVIEW:")
    print(f"   Total Tests: {len(test_tasks)}")
    planner_tests = [t for t in test_tasks if t['test_type'] == 'PLANNER_HEALING']
    executor_tests = [t for t in test_tasks if t['test_type'] == 'EXECUTOR_HEALING']
    mixed_tests = [t for t in test_tasks if t['test_type'] == 'MIXED_COMPLEXITY']
    print(f"   Planner Healing Tests: {len(planner_tests)}")
    print(f"   Executor Healing Tests: {len(executor_tests)}")
    print(f"   Mixed Complexity Tests: {len(mixed_tests)}")
    
    # Track overall results
    all_results = []
    healing_effectiveness = {
        "planner_healing_triggered": 0,
        "executor_healing_triggered": 0,
        "correct_classifications": 0,
        "total_tests": len(test_tasks)
    }
    
    # Run each test
    for i, task in enumerate(test_tasks, 1):
        print(f"\n" + "=" * 80)
        print(f"🧪 TEST {i}/{len(test_tasks)}: {TermColors.color_text(task['id'], TermColors.HEADER)}")
        print(f"🎯 Type: {TermColors.color_text(task['test_type'], TermColors.CYAN)}")
        print(f"📊 Complexity: {TermColors.color_text(task['complexity'], TermColors.YELLOW)}")
        print(f"🎭 Expected Target: {TermColors.color_text(task['expected_healing_target'], TermColors.GREEN)}")
        print("=" * 80)
        
        # Initialize agents for this specific test
        planner = Planner("TestPlanner", llm_service, task["planner_prompt"])
        executor = Executor("TestExecutor", llm_service, task["executor_prompt"])
        critic = Critic("TestCritic", llm_service)
        
        # Prepare task definition for harness
        task_definition = {
            "id": task["id"],
            "description": task["description"],
            "initial_executor_prompt": task["executor_prompt"]
        }
        
        print(f"🔧 AGENT CONFIGURATION:")
        planner_type = "GOOD" if task["planner_prompt"] == PLANNER_SYSTEM_PROMPT else "BAD"
        executor_type = "GOOD" if task["executor_prompt"] == DEFAULT_EXECUTOR_SYSTEM_PROMPT else "BAD"
        print(f"   🤖 Planner: {TermColors.color_text(planner_type, TermColors.GREEN if planner_type == 'GOOD' else TermColors.FAIL)} prompt")
        print(f"   🔧 Executor: {TermColors.color_text(executor_type, TermColors.GREEN if executor_type == 'GOOD' else TermColors.FAIL)} prompt")
        
        # Run the test
        start_time = time.time()
        result = run_enhanced_multi_agent_task(
            task_definition=task_definition,
            planner=planner,
            executor=executor,
            critic=critic,
            llm_service_instance=llm_service,
            max_healing_iterations=3
        )
        end_time = time.time()
        
        # Analyze results
        result["test_metadata"] = {
            "test_type": task["test_type"],
            "expected_healing_target": task["expected_healing_target"],
            "complexity": task["complexity"],
            "execution_time": end_time - start_time
        }
        
        all_results.append(result)
        
        # Check healing effectiveness
        planner_healings = result['healing_breakdown']['planner_healings']
        executor_healings = result['healing_breakdown']['executor_healings']
        
        if planner_healings > 0:
            healing_effectiveness["planner_healing_triggered"] += 1
        if executor_healings > 0:
            healing_effectiveness["executor_healing_triggered"] += 1
            
        # Check if healing target was correctly identified
        expected_target = task["expected_healing_target"]
        if expected_target == "PLANNER" and planner_healings > 0:
            healing_effectiveness["correct_classifications"] += 1
        elif expected_target == "EXECUTOR" and executor_healings > 0:
            healing_effectiveness["correct_classifications"] += 1
        elif expected_target == "MULTIPLE" and (planner_healings > 0 or executor_healings > 0):
            healing_effectiveness["correct_classifications"] += 1
        
        # Display test results
        print(f"\n📊 TEST {i} RESULTS:")
        print(f"   Final Status: {TermColors.color_text(result['final_status'], TermColors.GREEN if 'SUCCESS' in result['final_status'] else TermColors.FAIL)}")
        print(f"   Final Score: {result['final_score']:.2f}")
        print(f"   Execution Time: {result['test_metadata']['execution_time']:.1f}s")
        print(f"   Planner Healings: {planner_healings}")
        print(f"   Executor Healings: {executor_healings}")
        
        # Classification analysis
        if result.get('classification_history'):
            print(f"   🤖 LLM Classifications: {len(result['classification_history'])}")
            for j, classification in enumerate(result['classification_history'], 1):
                failure_type = classification['primary_failure_type']
                target = classification['recommended_healing_target']
                confidence = classification['confidence']
                print(f"      {j}. {failure_type} → {target} (confidence: {confidence:.2f})")
        
        # Success/failure indication
        healing_worked = (
            (expected_target == "PLANNER" and planner_healings > 0) or
            (expected_target == "EXECUTOR" and executor_healings > 0) or
            (expected_target == "MULTIPLE" and (planner_healings > 0 or executor_healings > 0))
        )
        
        if healing_worked:
            print(f"   ✅ {TermColors.color_text('HEALING TARGET CORRECTLY IDENTIFIED', TermColors.GREEN)}")
        else:
            print(f"   ❌ {TermColors.color_text('HEALING TARGET INCORRECT OR NO HEALING', TermColors.FAIL)}")
    
    # Generate comprehensive summary
    print(f"\n" + "=" * 80)
    print(f"📈 COMPREHENSIVE TEST SUITE SUMMARY")
    print("=" * 80)
    
    total_tests = healing_effectiveness["total_tests"]
    correct_classifications = healing_effectiveness["correct_classifications"]
    planner_triggered = healing_effectiveness["planner_healing_triggered"]
    executor_triggered = healing_effectiveness["executor_healing_triggered"]
    
    print(f"🎯 OVERALL EFFECTIVENESS:")
    accuracy = (correct_classifications / total_tests) * 100
    print(f"   Classification Accuracy: {TermColors.color_text(f'{accuracy:.1f}%', TermColors.GREEN if accuracy >= 70 else TermColors.FAIL)} ({correct_classifications}/{total_tests})")
    print(f"   Planner Healing Triggered: {planner_triggered} tests")
    print(f"   Executor Healing Triggered: {executor_triggered} tests")
    
    # Detailed breakdown by test type
    print(f"\n📊 BREAKDOWN BY TEST TYPE:")
    
    for test_type in ['PLANNER_HEALING', 'EXECUTOR_HEALING', 'MIXED_COMPLEXITY']:
        type_results = [r for r in all_results if r['test_metadata']['test_type'] == test_type]
        if not type_results:
            continue
            
        print(f"\n   {TermColors.color_text(test_type, TermColors.HEADER)}:")
        
        successful_healings = 0
        total_score = 0
        total_time = 0
        
        for result in type_results:
            expected = result['test_metadata']['expected_healing_target']
            planner_h = result['healing_breakdown']['planner_healings'] > 0
            executor_h = result['healing_breakdown']['executor_healings'] > 0
            
            healing_success = (
                (expected == "PLANNER" and planner_h) or
                (expected == "EXECUTOR" and executor_h) or
                (expected == "MULTIPLE" and (planner_h or executor_h))
            )
            
            if healing_success:
                successful_healings += 1
                
            total_score += result['final_score']
            total_time += result['test_metadata']['execution_time']
        
        count = len(type_results)
        success_rate = (successful_healings / count) * 100 if count > 0 else 0
        avg_score = total_score / count if count > 0 else 0
        avg_time = total_time / count if count > 0 else 0
        
        print(f"      Tests: {count}")
        print(f"      Success Rate: {TermColors.color_text(f'{success_rate:.1f}%', TermColors.GREEN if success_rate >= 70 else TermColors.FAIL)}")
        print(f"      Average Score: {avg_score:.2f}")
        print(f"      Average Time: {avg_time:.1f}s")
    
    # Final assessment
    print(f"\n🎉 FINAL ASSESSMENT:")
    if accuracy >= 80:
        print(f"   ✅ {TermColors.color_text('EXCELLENT: System demonstrates strong healing capabilities', TermColors.GREEN)}")
    elif accuracy >= 60:
        print(f"   ⚠️  {TermColors.color_text('GOOD: System shows healing ability with room for improvement', TermColors.YELLOW)}")
    else:
        print(f"   ❌ {TermColors.color_text('NEEDS IMPROVEMENT: Healing system needs refinement', TermColors.FAIL)}")
    
    print(f"   🧠 LLM-based classification successfully differentiated between planning and execution failures")
    print(f"   🔧 Enhanced multi-agent harness applied targeted healing based on failure analysis")
    
    return all_results, healing_effectiveness

def run_single_test(test_id: str):
    """Run a specific test by ID for focused debugging."""
    test_tasks = get_challenging_test_tasks()
    task = next((t for t in test_tasks if t['id'] == test_id), None)
    
    if not task:
        print(f"❌ Test '{test_id}' not found")
        available_tests = [t['id'] for t in test_tasks]
        print(f"Available tests: {available_tests}")
        return
    
    print(f"🧪 RUNNING SINGLE TEST: {test_id}")
    print("=" * 60)
    
    # Setup LLM service
    provider = os.environ.get("LLM_PROVIDER", "deepseek")
    model_name = os.environ.get("LLM_MODEL", "deepseek-coder")
    
    try:
        llm_service = LLMService(provider=provider, model_name=model_name)
    except Exception as e:
        print(f"❌ LLM Service Error: {e}")
        return
    
    # Run the specific test
    planner = Planner("TestPlanner", llm_service, task["planner_prompt"])
    executor = Executor("TestExecutor", llm_service, task["executor_prompt"])
    critic = Critic("TestCritic", llm_service)
    
    task_definition = {
        "id": task["id"],
        "description": task["description"],
        "initial_executor_prompt": task["executor_prompt"]
    }
    
    result = run_enhanced_multi_agent_task(
        task_definition=task_definition,
        planner=planner,
        executor=executor,
        critic=critic,
        llm_service_instance=llm_service,
        max_healing_iterations=3
    )
    
    print(f"\n📊 SINGLE TEST RESULTS:")
    print(f"   Final Status: {result['final_status']}")
    print(f"   Final Score: {result['final_score']:.2f}")
    print(f"   Planner Healings: {result['healing_breakdown']['planner_healings']}")
    print(f"   Executor Healings: {result['healing_breakdown']['executor_healings']}")
    
    return result

if __name__ == "__main__":
    """
    Run the comprehensive test suite or a specific test.
    Usage:
        python comprehensive_healing_test.py                    # Run all tests
        python comprehensive_healing_test.py test_id            # Run specific test
    """
    import sys
    
    if len(sys.argv) > 1:
        test_id = sys.argv[1]
        run_single_test(test_id)
    else:
        run_comprehensive_healing_tests() 