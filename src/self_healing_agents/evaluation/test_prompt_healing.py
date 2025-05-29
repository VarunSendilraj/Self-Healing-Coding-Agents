"""
Test script to verify the new prompt healing approach.
This tests that healing improves agent prompts holistically, not just for specific tasks.
"""

import logging
import os
from self_healing_agents.agents import Planner, Executor, Critic
from self_healing_agents.llm_service import LLMService
from self_healing_agents.prompts import BAD_PLANNER_PROMPT, DEFAULT_EXECUTOR_SYSTEM_PROMPT
from self_healing_agents.evaluation.enhanced_multi_agent_harness import run_enhanced_multi_agent_task
from self_healing_agents.evaluation.enhanced_harness import TermColors

def main():
    """Test the new prompt healing approach with a challenging task."""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("🔧 PROMPT HEALING SYSTEM TEST")
    print("=" * 60)
    print("🎯 Goal: Verify holistic prompt improvement (not task-specific)")
    print("=" * 60)
    
    # Configure environment
    provider = os.environ.get("LLM_PROVIDER", "deepseek")
    model_name = os.environ.get("LLM_MODEL", "deepseek-coder")
    
    # Initialize LLM service
    try:
        llm_service = LLMService(provider=provider, model_name=model_name)
        print(f"✅ LLM Service: {provider}/{model_name}")
    except Exception as e:
        print(f"❌ LLM Service Error: {e}")
        return
    
    # Use bad planner to trigger healing
    print(f"\n🔧 AGENT CONFIGURATION:")
    print(f"   🤖 Planner: BAD_PLANNER_PROMPT (vague, unhelpful)")
    print(f"   🔧 Executor: DEFAULT_EXECUTOR_SYSTEM_PROMPT")
    print(f"   🧐 Critic: Standard evaluation")
    print(f"   🔧 Healing: Prompt optimization (holistic, not task-specific)")
    
    planner = Planner("BadPlanner", llm_service, BAD_PLANNER_PROMPT)
    executor = Executor("Executor", llm_service, DEFAULT_EXECUTOR_SYSTEM_PROMPT)
    critic = Critic("Critic", llm_service)
    
    # Test with a moderately challenging task
    test_task = {
        "id": "prompt_healing_test",
        "description": """
Implement a function `binary_search(arr, target)` that:

1. Takes a sorted array `arr` and a target value `target`
2. Returns the index of the target if found, or -1 if not found
3. Uses the binary search algorithm for O(log n) time complexity
4. Handles edge cases: empty array, single element, target not in array
5. Includes proper input validation

Example:
binary_search([1, 3, 5, 7, 9], 5) should return 2
binary_search([1, 3, 5, 7, 9], 6) should return -1
binary_search([], 5) should return -1

The implementation should be efficient and handle all edge cases properly.
""",
        "initial_executor_prompt": DEFAULT_EXECUTOR_SYSTEM_PROMPT
    }
    
    print(f"\n📋 TEST TASK:")
    print(f"   ID: {test_task['id']}")
    print(f"   Task: Binary search implementation")
    print(f"   Challenge: Moderate complexity to trigger healing")
    
    # Run the enhanced harness
    print(f"\n🏃‍♂️ RUNNING PROMPT HEALING TEST...")
    print("=" * 60)
    
    result = run_enhanced_multi_agent_task(
        task_definition=test_task,
        planner=planner,
        executor=executor,
        critic=critic,
        llm_service_instance=llm_service,
        max_healing_iterations=3
    )
    
    # Analyze results
    print(f"\n📊 PROMPT HEALING RESULTS:")
    print("=" * 60)
    
    final_status = result['final_status']
    success = 'SUCCESS' in final_status
    
    print(f"   🎯 Final Status: {TermColors.color_text(final_status, TermColors.GREEN if success else TermColors.FAIL)}")
    final_score_str = f"{result['final_score']:.2f}"
    print(f"   📈 Final Score: {TermColors.color_text(final_score_str, TermColors.GREEN)}")
    print(f"   🔄 Total Healing Iterations: {TermColors.color_text(str(result['total_healing_iterations']), TermColors.CYAN)}")
    print(f"   🧠 Planner Healings: {result['healing_breakdown']['planner_healings']}")
    print(f"   ⚙️  Executor Healings: {result['healing_breakdown']['executor_healings']}")
    print(f"   🔨 Direct Fix Attempts: {result['healing_breakdown']['direct_fix_attempts']}")
    
    # Analyze prompt improvements
    if result['total_healing_iterations'] > 0:
        print(f"\n🔧 PROMPT IMPROVEMENT ANALYSIS:")
        print("=" * 60)
        
        for i, phase in enumerate(result['workflow_phases']):
            if 'HEALING' in phase.get('phase', ''):
                iteration_num = phase.get('iteration_num', i)
                healing_target = phase.get('healing_target', 'UNKNOWN')
                
                print(f"\n📋 Healing Iteration {iteration_num}:")
                print(f"   🎯 Target: {TermColors.color_text(healing_target, TermColors.CYAN)}")
                
                if healing_target == "PLANNER" and phase.get('improved_planner_prompt'):
                    improved_prompt = phase['improved_planner_prompt']
                    print(f"   🧠 Improved Planner Prompt (first 200 chars):")
                    print(f"      {improved_prompt[:200]}...")
                    
                    # Check if prompt is holistic (not task-specific)
                    is_holistic = analyze_prompt_holistic_quality(improved_prompt)
                    holistic_status = "✅ HOLISTIC" if is_holistic else "❌ TASK-SPECIFIC"
                    print(f"   📊 Prompt Quality: {holistic_status}")
                    
                elif healing_target == "EXECUTOR" and phase.get('improved_executor_prompt'):
                    improved_prompt = phase['improved_executor_prompt']
                    print(f"   ⚙️  Improved Executor Prompt (first 200 chars):")
                    print(f"      {improved_prompt[:200]}...")
                    
                    # Check if prompt is holistic (not task-specific)
                    is_holistic = analyze_prompt_holistic_quality(improved_prompt)
                    holistic_status = "✅ HOLISTIC" if is_holistic else "❌ TASK-SPECIFIC"
                    print(f"   📊 Prompt Quality: {holistic_status}")
                
                healing_successful = phase.get('healing_successful', False)
                success_text = 'YES' if healing_successful else 'NO'
                success_color = TermColors.GREEN if healing_successful else TermColors.FAIL
                print(f"   ✅ Success: {TermColors.color_text(success_text, success_color)}")
                
                if phase.get('improved_score'):
                    print(f"   📈 Improved Score: {phase['improved_score']:.2f}")
    
    # Final assessment
    print(f"\n🎯 PROMPT HEALING ASSESSMENT:")
    print("=" * 60)
    
    if result['total_healing_iterations'] > 0:
        print(f"✅ SUCCESS: Prompt healing system was activated!")
        print(f"🔧 The system improved agent prompts {result['total_healing_iterations']} time(s)")
        
        if result['healing_breakdown']['planner_healings'] > 0:
            print(f"🧠 Planner prompt was improved {result['healing_breakdown']['planner_healings']} time(s)")
        
        if result['healing_breakdown']['executor_healings'] > 0:
            print(f"⚙️  Executor prompt was improved {result['healing_breakdown']['executor_healings']} time(s)")
            
        print(f"📊 Final outcome: {final_status} with score {result['final_score']:.2f}")
        
        # Check if healing was holistic
        holistic_healing_count = 0
        for phase in result['workflow_phases']:
            if 'HEALING' in phase.get('phase', ''):
                if phase.get('improved_planner_prompt'):
                    if analyze_prompt_holistic_quality(phase['improved_planner_prompt']):
                        holistic_healing_count += 1
                if phase.get('improved_executor_prompt'):
                    if analyze_prompt_holistic_quality(phase['improved_executor_prompt']):
                        holistic_healing_count += 1
        
        if holistic_healing_count > 0:
            print(f"🎉 EXCELLENT: {holistic_healing_count} holistic prompt improvement(s) detected!")
            print(f"🔧 The system improved prompts generally, not just for this specific task")
        else:
            print(f"⚠️  WARNING: Healing appears to be task-specific rather than holistic")
            print(f"💡 Consider adjusting the healing prompts to be more general")
        
    else:
        print(f"⚠️  Prompt healing was not triggered (task succeeded initially)")
        print(f"💡 Try a more challenging task or adjust the planner prompt to be more problematic")
    
    print(f"\n🔧 PROMPT HEALING TEST COMPLETE!")
    return result

def analyze_prompt_holistic_quality(prompt: str) -> bool:
    """
    Analyze if a prompt is holistic (general) or task-specific.
    
    Args:
        prompt: The prompt text to analyze
        
    Returns:
        bool: True if prompt appears holistic, False if task-specific
    """
    prompt_lower = prompt.lower()
    
    # Task-specific indicators (bad)
    task_specific_indicators = [
        "binary search", "binary_search", "sorted array", "target value",
        "o(log n)", "log n", "index of the target", "arr", "target"
    ]
    
    # Holistic indicators (good)
    holistic_indicators = [
        "algorithm", "data structure", "edge case", "input validation",
        "error handling", "best practice", "efficient", "robust",
        "comprehensive", "general", "various", "diverse", "multiple"
    ]
    
    # Count indicators
    task_specific_count = sum(1 for indicator in task_specific_indicators if indicator in prompt_lower)
    holistic_count = sum(1 for indicator in holistic_indicators if indicator in prompt_lower)
    
    # Prompt is considered holistic if it has more general guidance than task-specific references
    return holistic_count > task_specific_count and task_specific_count <= 2

if __name__ == "__main__":
    main() 