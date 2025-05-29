#!/usr/bin/env python3
"""
Test script for the simple evolutionary optimizer.
This demonstrates the direct fitness evaluation approach.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import logging
from self_healing_agents.llm_service import LLMService
from self_healing_agents.agents import Planner, Executor, Critic
from self_healing_agents.evolution.simple_evolutionary_optimizer import SimpleEvolutionaryOptimizer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_simple_evolution():
    """Test the simple evolutionary optimizer."""
    
    print("🧪 Testing Simple Evolutionary Optimizer")
    print("="*60)
    
    # Initialize LLM service
    llm_service = LLMService(
        provider="deepseek",
        model_name="deepseek-chat",
        temperature=0.7,
        max_tokens=2048
    )
    
    # Create agents
    planner = Planner("test_planner", llm_service)
    executor = Executor("test_executor", llm_service)
    critic = Critic("test_critic", llm_service)
    
    # Define the task
    task_description = '''Implement regular expression matching with support for "." and "*" where:
- "." matches any single character
- "*" matches zero or more of the preceding element
- The matching should cover the entire input string (not partial)

Function should be named "isMatch" and take two parameters: s (string) and p (pattern).'''
    
    print(f"📋 Task: {task_description}")
    
    # Create some basic test cases for the simple fitness evaluator
    basic_test_cases = [
        {
            "test_case_name": "test_exact_match",
            "inputs": {"s": "aa", "p": "aa"},
            "expected_output": True
        },
        {
            "test_case_name": "test_star_zero_match",
            "inputs": {"s": "a", "p": "ab*a"},
            "expected_output": True
        },
        {
            "test_case_name": "test_dot_match",
            "inputs": {"s": "ab", "p": ".b"},
            "expected_output": True
        },
        {
            "test_case_name": "test_star_multiple",
            "inputs": {"s": "aaa", "p": "a*"},
            "expected_output": True
        },
        {
            "test_case_name": "test_empty_empty",
            "inputs": {"s": "", "p": ""},
            "expected_output": True
        },
        {
            "test_case_name": "test_no_match",
            "inputs": {"s": "aa", "p": "a"},
            "expected_output": False
        }
    ]
    
    print(f"✅ Using {len(basic_test_cases)} predefined test cases for fitness evaluation")
    for i, test_case in enumerate(basic_test_cases[:3], 1):
        inputs = test_case.get('inputs', {})
        expected = test_case.get('expected_output', 'unknown')
        print(f"   {i}. {inputs} → {expected}")
    print(f"   ... and {len(basic_test_cases)-3} more")
    
    # Test the baseline first
    print(f"\n🔍 Getting baseline performance...")
    
    # Run the baseline pipeline
    try:
        plan = planner.run(user_request=task_description)
        if isinstance(plan, dict) and plan.get("error"):
            print(f"❌ Baseline planner failed: {plan}")
            return
        
        code = executor.run(plan=plan, original_request=task_description)
        if isinstance(code, dict) and code.get("error"):
            print(f"❌ Baseline executor failed: {code}")
            return
        
        # Get a quick evaluation using the critic
        critic_report = critic.evaluate_code(code, task_description)
        baseline_score = critic_report.get("score", 0.0)
        
        print(f"📊 Baseline Score: {baseline_score:.3f}")
        print(f"📝 Baseline Code Preview: {code[:100]}...")
        
    except Exception as e:
        print(f"❌ Baseline evaluation failed: {e}")
        baseline_score = 0.0
    
    # Now test evolution on the planner
    print(f"\n🧬 Testing Simple Evolution on PLANNER")
    print("-" * 40)
    
    optimizer = SimpleEvolutionaryOptimizer(
        llm_service=llm_service,
        planner_agent=planner,
        executor_agent=executor,
        critic_agent=critic,
        task_description=task_description,
        original_test_cases=basic_test_cases
    )
    
    # Create failure context for targeted evolution
    failure_context = {
        "original_task": task_description,
        "baseline_score": baseline_score,
        "specific_test_failures": []
    }
    
    # Add test failure examples
    failure_context["specific_test_failures"] = [
        {
            "test_name": "test_star_zero_match",
            "inputs": {"s": "a", "p": "ab*a"},
            "expected_output": True,
            "actual_output": False,
            "error": "Star pattern not handling zero matches correctly"
        }
    ]
    
    # Optimize planner prompt
    try:
        planner_results = optimizer.optimize_prompt(
            base_prompt=planner.system_prompt,
            agent_type="PLANNER",
            failure_context=failure_context
        )
        
        print(f"\n📊 PLANNER EVOLUTION RESULTS:")
        print(f"   🏆 Best Fitness: {planner_results.best_fitness:.3f}")
        print(f"   📈 Improvement: {planner_results.best_fitness - baseline_score:+.3f}")
        print(f"   🔢 Generations: {planner_results.generation_count}")
        print(f"   📊 Evaluations: {planner_results.evaluation_count}")
        print(f"   ⏱️  Time: {planner_results.execution_time:.1f}s")
        print(f"   🛑 Reason: {planner_results.termination_reason}")
        
        if planner_results.best_fitness > baseline_score:
            print(f"   🎉 IMPROVEMENT ACHIEVED!")
            print(f"   📝 Best Evolved Prompt Preview: {planner_results.best_prompt[:100]}...")
        
    except Exception as e:
        print(f"❌ Planner evolution failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test evolution on the executor  
    print(f"\n🧬 Testing Simple Evolution on EXECUTOR")
    print("-" * 40)
    
    try:
        executor_results = optimizer.optimize_prompt(
            base_prompt=executor.system_prompt,
            agent_type="EXECUTOR", 
            failure_context=failure_context
        )
        
        print(f"\n📊 EXECUTOR EVOLUTION RESULTS:")
        print(f"   🏆 Best Fitness: {executor_results.best_fitness:.3f}")
        print(f"   📈 Improvement: {executor_results.best_fitness - baseline_score:+.3f}")
        print(f"   🔢 Generations: {executor_results.generation_count}")
        print(f"   📊 Evaluations: {executor_results.evaluation_count}")
        print(f"   ⏱️  Time: {executor_results.execution_time:.1f}s")
        print(f"   🛑 Reason: {executor_results.termination_reason}")
        
        if executor_results.best_fitness > baseline_score:
            print(f"   🎉 IMPROVEMENT ACHIEVED!")
    
    except Exception as e:
        print(f"❌ Executor evolution failed: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n✅ Simple Evolution Test Complete!")
    print("="*60)

if __name__ == "__main__":
    test_simple_evolution() 