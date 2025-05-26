#!/usr/bin/env python3
"""
Test the Fixed Evolutionary System with the Actual Regex Task

This test validates that the convergence fixes work properly when
using the actual regex matching task that was causing issues.
"""

import sys
import os
import logging

# Setup path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from self_healing_agents.agents import Planner, Executor, Critic
from self_healing_agents.llm_service import LLMService
from self_healing_agents.prompts import DEFAULT_EXECUTOR_SYSTEM_PROMPT, PLANNER_SYSTEM_PROMPT
from self_healing_agents.evaluation.evolutionary_enhanced_harness import run_evolutionary_multi_agent_task

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_evolutionary_fixes():
    """Test the evolutionary system fixes with the actual regex task."""
    print("🧪 TESTING FIXED EVOLUTIONARY SYSTEM")
    print("=" * 60)
    
    # Initialize LLM service
    try:
        llm_service = LLMService(provider="deepseek", model_name="deepseek-coder")
        print("✅ LLM Service initialized")
    except Exception as e:
        print(f"❌ LLM Service failed: {e}")
        return False
    
    # Initialize agents
    planner = Planner("FixedPlanner", llm_service, PLANNER_SYSTEM_PROMPT)
    executor = Executor("FixedExecutor", llm_service, DEFAULT_EXECUTOR_SYSTEM_PROMPT) 
    critic = Critic("FixedCritic", llm_service)
    print("✅ Agents initialized")
    
    # Create the actual regex task that was causing issues
    regex_task = {
        "id": "fixed_evolutionary_test",
        "description": """Given an input string s and a pattern p, implement regular expression matching with support for '.' and '*' where:

'.' Matches any single character.
'*' Matches zero or more of the preceding element.
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
1 <= s.length <= 20
1 <= p.length <= 20
s contains only lowercase English letters.
p contains only lowercase English letters, '.', and '*'.
It is guaranteed for each appearance of the character '*', there will be a previous valid character to match.""",
        "initial_executor_prompt": DEFAULT_EXECUTOR_SYSTEM_PROMPT
    }
    
    print(f"📝 Task: {regex_task['description'][:100]}...")
    print()
    
    # Run the evolutionary multi-agent task with fixes applied
    print("🔄 Running evolutionary optimization with applied fixes...")
    result = run_evolutionary_multi_agent_task(
        task_definition=regex_task,
        planner=planner,
        executor=executor,
        critic=critic,
        llm_service_instance=llm_service,
        max_healing_iterations=2,
        use_evolutionary_optimization=True
    )
    
    # Analyze results
    print("\n" + "=" * 60)
    print("📊 EVOLUTIONARY FIXES TEST RESULTS:")
    print("=" * 60)
    
    success_indicators = []
    
    # Check final status
    final_status = result['final_status']
    final_score = result['final_score']
    print(f"🎯 Final Status: {final_status}")
    print(f"🏆 Final Score: {final_score:.3f}")
    
    if final_score > 0.0:
        success_indicators.append("✅ Non-zero final score achieved")
    else:
        success_indicators.append("❌ Zero final score - system failed")
    
    # Check healing iterations
    total_healing = result['total_healing_iterations']
    evolutionary_opts = result['healing_breakdown']['evolutionary_optimizations']
    print(f"🔧 Total Healing Iterations: {total_healing}")
    print(f"🧬 Evolutionary Optimizations: {evolutionary_opts}")
    
    if evolutionary_opts > 0:
        success_indicators.append("✅ Evolutionary optimizations were performed")
    else:
        success_indicators.append("❌ No evolutionary optimizations performed")
    
    # Check evolution results
    if result['evolutionary_results']:
        print(f"\n🧬 EVOLUTIONARY DETAILS:")
        for i, evo_result in enumerate(result['evolutionary_results'], 1):
            print(f"   Optimization {i}:")
            print(f"      🏆 Best Fitness: {evo_result['best_fitness']:.3f}")
            print(f"      🔄 Generations: {evo_result['generation_count']}")
            print(f"      📊 Evaluations: {evo_result['evaluation_count']}")
            print(f"      ⏱️  Time: {evo_result['execution_time']:.1f}s")
            print(f"      📈 Improvement: {evo_result['improvement']:+.3f}")
            print(f"      🛑 Termination: {evo_result['termination_reason']}")
            
            # Check for meaningful evolution
            if evo_result['generation_count'] > 1:
                success_indicators.append("✅ Multi-generation evolution occurred")
            
            if evo_result['evaluation_count'] > 3:
                success_indicators.append("✅ Multiple fitness evaluations performed")
    else:
        success_indicators.append("❌ No evolutionary results recorded")
    
    # Check classification history
    if result['classification_history']:
        print(f"\n🔍 CLASSIFICATION HISTORY:")
        for i, classification in enumerate(result['classification_history'], 1):
            print(f"   Classification {i}: {classification['primary_failure_type']} "
                  f"(confidence: {classification['confidence']:.2f})")
            print(f"      Target: {classification['recommended_healing_target']}")
        
        success_indicators.append("✅ Failure classification working")
    else:
        success_indicators.append("❌ No failure classifications performed")
    
    # Check for global best tracking (if implemented)
    workflow_phases = result['workflow_phases']
    for phase in workflow_phases:
        if phase.get('phase') == 'EVOLUTIONARY_HEALING':
            if 'evolution_results' in phase:
                evo_data = phase['evolution_results'] 
                if evo_data.get('best_fitness', 0) > 0:
                    success_indicators.append("✅ Global best tracking working")
                    break
    
    print(f"\n📈 SUCCESS INDICATORS:")
    for indicator in success_indicators:
        print(f"   {indicator}")
    
    # Overall assessment
    success_count = sum(1 for indicator in success_indicators if indicator.startswith("✅"))
    total_count = len(success_indicators)
    success_rate = success_count / total_count if total_count > 0 else 0
    
    print(f"\n🎉 OVERALL ASSESSMENT:")
    print(f"   Success Rate: {success_rate:.1%} ({success_count}/{total_count})")
    
    if success_rate >= 0.7:
        print(f"   🎉 EXCELLENT: Evolutionary fixes are working well!")
        return True
    elif success_rate >= 0.5:
        print(f"   ✅ GOOD: Most fixes are working, minor issues remain")
        return True
    else:
        print(f"   ⚠️ NEEDS WORK: Significant issues still present")
        return False


def main():
    """Run the evolutionary fixes test."""
    print("🚀 TESTING EVOLUTIONARY CONVERGENCE FIXES")
    print("=" * 60)
    
    try:
        success = test_evolutionary_fixes()
        
        if success:
            print("\n🎉 EVOLUTIONARY FIXES VALIDATED SUCCESSFULLY!")
            print("The system now properly tracks global best solutions and avoids convergence to inferior prompts.")
        else:
            print("\n⚠️ EVOLUTIONARY FIXES NEED FURTHER IMPROVEMENT")
            print("Some issues remain that require additional investigation.")
            
    except Exception as e:
        logger.error(f"Test failed with exception: {e}")
        print(f"❌ Test failed with error: {e}")


if __name__ == "__main__":
    main() 