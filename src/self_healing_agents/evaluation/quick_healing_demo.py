"""
Quick Demo of Multi-Agent Self-Healing Capabilities

This script provides a focused demonstration of the enhanced multi-agent harness
with specific examples that clearly show planner vs executor healing.
"""

import logging
import os
import time
from self_healing_agents.agents import Planner, Executor, Critic
from self_healing_agents.llm_service import LLMService
from self_healing_agents.prompts import (
    BAD_PLANNER_PROMPT, DEFAULT_EXECUTOR_SYSTEM_PROMPT, PLANNER_SYSTEM_PROMPT,
    CATASTROPHIC_EXECUTOR_PROMPT, SYNTAX_ERROR_EXECUTOR_PROMPT
)
from self_healing_agents.evaluation.enhanced_multi_agent_harness import run_enhanced_multi_agent_task
from self_healing_agents.evaluation.enhanced_harness import TermColors

def demo_planner_healing():
    """
    Demonstrate planner healing with a challenging algorithm problem.
    Uses BAD planner + GOOD executor to trigger planner healing.
    """
    print("🧪 DEMO 1: PLANNER HEALING")
    print("=" * 60)
    print("🎯 Testing with: BAD Planner + GOOD Executor")
    print("📋 Task: Complex dynamic programming algorithm")
    print("=" * 60)
    
    # Setup LLM service
    provider = os.environ.get("LLM_PROVIDER", "deepseek")
    model_name = os.environ.get("LLM_MODEL", "deepseek-coder")
    
    try:
        llm_service = LLMService(provider=provider, model_name=model_name)
        print(f"✅ LLM Service: {provider}/{model_name}")
    except Exception as e:
        print(f"❌ LLM Service Error: {e}")
        return
    
    # Initialize agents with targeted prompts
    planner = Planner("BadPlanner", llm_service, BAD_PLANNER_PROMPT)
    executor = Executor("GoodExecutor", llm_service, DEFAULT_EXECUTOR_SYSTEM_PROMPT)
    critic = Critic("Critic", llm_service)
    
    # Complex task requiring detailed planning
    task = {
        "id": "planner_healing_demo",
        "description": """Implement a function 'edit_distance(str1, str2)' that calculates the minimum edit distance (Levenshtein distance) between two strings using dynamic programming.

Requirements:
- Use a 2D DP table where dp[i][j] represents edit distance between first i characters of str1 and first j characters of str2
- Support three operations: insert, delete, substitute (each costs 1)
- Handle edge cases: empty strings, identical strings
- Return the minimum number of operations needed
- Function signature: def edit_distance(str1: str, str2: str) -> int

Examples:
- edit_distance("kitten", "sitting") -> 3
- edit_distance("", "abc") -> 3
- edit_distance("abc", "abc") -> 0""",
        "initial_executor_prompt": DEFAULT_EXECUTOR_SYSTEM_PROMPT
    }
    
    print(f"🔧 AGENT CONFIG:")
    print(f"   🤖 Planner: {TermColors.color_text('BAD', TermColors.FAIL)} (vague, incomplete plans)")
    print(f"   🔧 Executor: {TermColors.color_text('GOOD', TermColors.GREEN)} (should generate working code)")
    
    # Run the test
    start_time = time.time()
    result = run_enhanced_multi_agent_task(
        task_definition=task,
        planner=planner,
        executor=executor,
        critic=critic,
        llm_service_instance=llm_service,
        max_healing_iterations=3
    )
    end_time = time.time()
    
    # Display results
    print(f"\n📊 PLANNER HEALING DEMO RESULTS:")
    print(f"   Final Status: {TermColors.color_text(result['final_status'], TermColors.GREEN if 'SUCCESS' in result['final_status'] else TermColors.FAIL)}")
    print(f"   Final Score: {result['final_score']:.2f}")
    print(f"   Execution Time: {end_time - start_time:.1f}s")
    print(f"   🧠 Planner Healings: {TermColors.color_text(str(result['healing_breakdown']['planner_healings']), TermColors.GREEN if result['healing_breakdown']['planner_healings'] > 0 else TermColors.FAIL)}")
    print(f"   ⚙️  Executor Healings: {result['healing_breakdown']['executor_healings']}")
    
    if result['healing_breakdown']['planner_healings'] > 0:
        print(f"   ✅ {TermColors.color_text('SUCCESS: Planner healing was triggered!', TermColors.GREEN)}")
    else:
        print(f"   ❌ {TermColors.color_text('UNEXPECTED: No planner healing occurred', TermColors.FAIL)}")
    
    # Show LLM classifications
    if result.get('classification_history'):
        print(f"\n🤖 LLM CLASSIFICATION DECISIONS:")
        for i, classification in enumerate(result['classification_history'], 1):
            failure_type = classification['primary_failure_type']
            target = classification['recommended_healing_target']
            confidence = classification['confidence']
            print(f"   {i}. {failure_type} → Heal {target} (confidence: {confidence:.2f})")
    
    return result

def demo_executor_healing():
    """
    Demonstrate executor healing with a simple problem.
    Uses GOOD planner + BAD executor to trigger executor healing.
    """
    print("\n🧪 DEMO 2: EXECUTOR HEALING")
    print("=" * 60)
    print("🎯 Testing with: GOOD Planner + BAD Executor")
    print("📋 Task: Simple string processing (should be easy to plan)")
    print("=" * 60)
    
    # Setup LLM service
    provider = os.environ.get("LLM_PROVIDER", "deepseek")
    model_name = os.environ.get("LLM_MODEL", "deepseek-coder")
    
    try:
        llm_service = LLMService(provider=provider, model_name=model_name)
    except Exception as e:
        print(f"❌ LLM Service Error: {e}")
        return
    
    # Initialize agents with targeted prompts
    planner = Planner("GoodPlanner", llm_service, PLANNER_SYSTEM_PROMPT)
    executor = Executor("BadExecutor", llm_service, CATASTROPHIC_EXECUTOR_PROMPT)
    critic = Critic("Critic", llm_service)
    
    # Simple task that should be easy to plan but executor will mess up
    task = {
        "id": "executor_healing_demo",
        "description": """Write a function 'is_palindrome(text)' that checks if a string is a palindrome (reads the same forwards and backwards).

Requirements:
- Ignore case sensitivity (e.g., "Aa" should be considered palindrome)
- Ignore spaces and punctuation (e.g., "A man a plan a canal Panama" is palindrome)
- Return True if palindrome, False otherwise
- Handle empty string (return True)
- Function signature: def is_palindrome(text: str) -> bool

Examples:
- is_palindrome("racecar") -> True
- is_palindrome("A man a plan a canal Panama") -> True
- is_palindrome("race a car") -> False
- is_palindrome("") -> True""",
        "initial_executor_prompt": CATASTROPHIC_EXECUTOR_PROMPT
    }
    
    print(f"🔧 AGENT CONFIG:")
    print(f"   🤖 Planner: {TermColors.color_text('GOOD', TermColors.GREEN)} (should create clear plans)")
    print(f"   🔧 Executor: {TermColors.color_text('BAD', TermColors.FAIL)} (generates broken code)")
    
    # Run the test
    start_time = time.time()
    result = run_enhanced_multi_agent_task(
        task_definition=task,
        planner=planner,
        executor=executor,
        critic=critic,
        llm_service_instance=llm_service,
        max_healing_iterations=3
    )
    end_time = time.time()
    
    # Display results
    print(f"\n📊 EXECUTOR HEALING DEMO RESULTS:")
    print(f"   Final Status: {TermColors.color_text(result['final_status'], TermColors.GREEN if 'SUCCESS' in result['final_status'] else TermColors.FAIL)}")
    print(f"   Final Score: {result['final_score']:.2f}")
    print(f"   Execution Time: {end_time - start_time:.1f}s")
    print(f"   🧠 Planner Healings: {result['healing_breakdown']['planner_healings']}")
    print(f"   ⚙️  Executor Healings: {TermColors.color_text(str(result['healing_breakdown']['executor_healings']), TermColors.GREEN if result['healing_breakdown']['executor_healings'] > 0 else TermColors.FAIL)}")
    
    if result['healing_breakdown']['executor_healings'] > 0:
        print(f"   ✅ {TermColors.color_text('SUCCESS: Executor healing was triggered!', TermColors.GREEN)}")
    else:
        print(f"   ❌ {TermColors.color_text('UNEXPECTED: No executor healing occurred', TermColors.FAIL)}")
    
    # Show LLM classifications
    if result.get('classification_history'):
        print(f"\n🤖 LLM CLASSIFICATION DECISIONS:")
        for i, classification in enumerate(result['classification_history'], 1):
            failure_type = classification['primary_failure_type']
            target = classification['recommended_healing_target']
            confidence = classification['confidence']
            print(f"   {i}. {failure_type} → Heal {target} (confidence: {confidence:.2f})")
    
    return result

def demo_mixed_scenario():
    """
    Demonstrate mixed scenario where both agents have issues.
    Uses BAD planner + BAD executor to see how system prioritizes healing.
    """
    print("\n🧪 DEMO 3: MIXED SCENARIO")
    print("=" * 60)
    print("🎯 Testing with: BAD Planner + BAD Executor")
    print("📋 Task: Medium complexity (both agents will struggle)")
    print("=" * 60)
    
    # Setup LLM service
    provider = os.environ.get("LLM_PROVIDER", "deepseek")
    model_name = os.environ.get("LLM_MODEL", "deepseek-coder")
    
    try:
        llm_service = LLMService(provider=provider, model_name=model_name)
    except Exception as e:
        print(f"❌ LLM Service Error: {e}")
        return
    
    # Initialize agents with both bad prompts
    planner = Planner("BadPlanner", llm_service, BAD_PLANNER_PROMPT)
    executor = Executor("BadExecutor", llm_service, SYNTAX_ERROR_EXECUTOR_PROMPT)
    critic = Critic("Critic", llm_service)
    
    # Medium complexity task
    task = {
        "id": "mixed_scenario_demo",
        "description": """Implement a function 'binary_search(sorted_list, target)' that performs binary search on a sorted list.

Requirements:
- Return the index of target if found, -1 if not found
- Handle empty lists
- Assume the input list is sorted
- Use iterative approach (not recursive)
- Function signature: def binary_search(sorted_list: list[int], target: int) -> int

Examples:
- binary_search([1, 3, 5, 7, 9], 5) -> 2
- binary_search([1, 3, 5, 7, 9], 6) -> -1
- binary_search([], 5) -> -1""",
        "initial_executor_prompt": SYNTAX_ERROR_EXECUTOR_PROMPT
    }
    
    print(f"🔧 AGENT CONFIG:")
    print(f"   🤖 Planner: {TermColors.color_text('BAD', TermColors.FAIL)} (vague, incomplete plans)")
    print(f"   🔧 Executor: {TermColors.color_text('BAD', TermColors.FAIL)} (syntax errors, broken code)")
    
    # Run the test
    start_time = time.time()
    result = run_enhanced_multi_agent_task(
        task_definition=task,
        planner=planner,
        executor=executor,
        critic=critic,
        llm_service_instance=llm_service,
        max_healing_iterations=3
    )
    end_time = time.time()
    
    # Display results
    print(f"\n📊 MIXED SCENARIO DEMO RESULTS:")
    print(f"   Final Status: {TermColors.color_text(result['final_status'], TermColors.GREEN if 'SUCCESS' in result['final_status'] else TermColors.FAIL)}")
    print(f"   Final Score: {result['final_score']:.2f}")
    print(f"   Execution Time: {end_time - start_time:.1f}s")
    print(f"   🧠 Planner Healings: {result['healing_breakdown']['planner_healings']}")
    print(f"   ⚙️  Executor Healings: {result['healing_breakdown']['executor_healings']}")
    
    total_healings = result['healing_breakdown']['planner_healings'] + result['healing_breakdown']['executor_healings']
    if total_healings > 0:
        print(f"   ✅ {TermColors.color_text(f'SUCCESS: {total_healings} healing(s) were triggered!', TermColors.GREEN)}")
    else:
        print(f"   ❌ {TermColors.color_text('UNEXPECTED: No healing occurred', TermColors.FAIL)}")
    
    # Show LLM classifications
    if result.get('classification_history'):
        print(f"\n🤖 LLM CLASSIFICATION DECISIONS:")
        for i, classification in enumerate(result['classification_history'], 1):
            failure_type = classification['primary_failure_type']
            target = classification['recommended_healing_target']
            confidence = classification['confidence']
            print(f"   {i}. {failure_type} → Heal {target} (confidence: {confidence:.2f})")
    
    return result

def run_quick_demo():
    """Run all three demo scenarios to showcase the enhanced multi-agent harness."""
    print("🚀 QUICK MULTI-AGENT SELF-HEALING DEMONSTRATION")
    print("=" * 80)
    print("🎯 This demo shows how the enhanced harness can intelligently identify")
    print("   and heal both planning and execution failures using LLM-based classification.")
    print("=" * 80)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    results = []
    
    # Run all three demos
    try:
        print("\n" + "🎬 STARTING DEMONSTRATIONS...")
        
        # Demo 1: Planner healing
        result1 = demo_planner_healing()
        if result1:
            results.append(result1)
        
        # Demo 2: Executor healing
        result2 = demo_executor_healing()
        if result2:
            results.append(result2)
        
        # Demo 3: Mixed scenario
        result3 = demo_mixed_scenario()
        if result3:
            results.append(result3)
        
    except KeyboardInterrupt:
        print(f"\n{TermColors.color_text('Demo interrupted by user', TermColors.WARNING)}")
        return
    except Exception as e:
        print(f"\n{TermColors.color_text(f'Demo error: {e}', TermColors.FAIL)}")
        return
    
    # Summary
    print(f"\n" + "=" * 80)
    print(f"📈 QUICK DEMO SUMMARY")
    print("=" * 80)
    
    if results:
        total_planner_healings = sum(r['healing_breakdown']['planner_healings'] for r in results)
        total_executor_healings = sum(r['healing_breakdown']['executor_healings'] for r in results)
        avg_score = sum(r['final_score'] for r in results) / len(results)
        
        print(f"🎯 DEMO RESULTS:")
        print(f"   Completed Demos: {len(results)}/3")
        print(f"   Total Planner Healings: {total_planner_healings}")
        print(f"   Total Executor Healings: {total_executor_healings}")
        print(f"   Average Final Score: {avg_score:.2f}")
        
        successful_healings = [r for r in results if 
                             r['healing_breakdown']['planner_healings'] > 0 or 
                             r['healing_breakdown']['executor_healings'] > 0]
        
        if len(successful_healings) >= 2:
            print(f"   ✅ {TermColors.color_text('SUCCESS: Enhanced harness demonstrated intelligent healing!', TermColors.GREEN)}")
        elif len(successful_healings) >= 1:
            print(f"   ⚠️  {TermColors.color_text('PARTIAL: Some healing demonstrated', TermColors.YELLOW)}")
        else:
            print(f"   ❌ {TermColors.color_text('ISSUE: No healing was triggered in demos', TermColors.FAIL)}")
    
    print(f"\n🎯 KEY TAKEAWAYS:")
    print(f"   🧠 LLM-based failure classification analyzes test results and context")
    print(f"   🔧 System can distinguish between planning and execution problems")
    print(f"   ⚙️  Targeted healing is applied to the specific problematic agent")
    print(f"   📊 Multiple iterations allow for progressive improvement")
    
    return results

if __name__ == "__main__":
    """
    Run the quick demonstration of multi-agent self-healing capabilities.
    """
    run_quick_demo() 