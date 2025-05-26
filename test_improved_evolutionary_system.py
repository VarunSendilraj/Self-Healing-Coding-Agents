#!/usr/bin/env python3
"""
Test script for the improved evolutionary prompt optimization system.

This script demonstrates:
1. Task-specific fitness evaluation that directly tests regex matching performance
2. Error-targeted prompt generation that holistically addresses specific failures
3. Comprehensive logging and diagnostics
"""

import sys
import os
import logging

# Ensure the src directory is in the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from self_healing_agents.agents import Planner, Executor, Critic
from self_healing_agents.llm_service import LLMService
from self_healing_agents.prompts import DEFAULT_EXECUTOR_SYSTEM_PROMPT, PLANNER_SYSTEM_PROMPT
from self_healing_agents.evolution import (
    EvolutionaryPromptOptimizer, 
    EvolutionConfig,
    create_planner_optimizer,
    create_executor_optimizer
)

def setup_logging():
    """Set up comprehensive logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('improved_evolutionary_test.log')
        ]
    )

def test_task_specific_fitness_evaluation():
    """Test the task-specific fitness evaluation system."""
    print("\n" + "="*80)
    print("🧪 TESTING TASK-SPECIFIC FITNESS EVALUATION")
    print("="*80)
    
    # Configure environment
    provider = os.environ.get("LLM_PROVIDER", "deepseek")
    model_name = os.environ.get("LLM_MODEL", "deepseek-coder")
    
    # Initialize LLM service
    try:
        llm_service = LLMService(provider=provider, model_name=model_name)
        print(f"✅ LLM Service: {provider}/{model_name}")
    except Exception as e:
        print(f"❌ LLM Service Error: {e}")
        return False
    
    # Initialize agents
    planner = Planner("TestPlanner", llm_service, PLANNER_SYSTEM_PROMPT)
    executor = Executor("TestExecutor", llm_service, DEFAULT_EXECUTOR_SYSTEM_PROMPT)
    critic = Critic("TestCritic", llm_service)
    
    # Create evolutionary optimizer
    config = EvolutionConfig(
        population_size=3,
        max_generations=2,
        max_evaluations=10
    )
    optimizer = EvolutionaryPromptOptimizer(llm_service, config)
    
    # Set up task-specific context
    regex_task = """Given an input string s and a pattern p, implement regular expression matching with support for '.' and '*' where:

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
It is guaranteed for each appearance of the character '*', there will be a previous valid character to match."""
    
    print(f"📋 Setting up task-specific fitness evaluation for regex matching")
    optimizer.fitness_evaluator.set_task_specific_context(
        task_description=regex_task,
        planner_agent=planner,
        executor_agent=executor,
        critic_agent=critic
    )
    
    # Test different prompts
    test_prompts = [
        "You are a Python programmer. Write code according to the specification.",
        "You are an expert software engineer specializing in dynamic programming and regex implementation. Write efficient, correct Python code that handles edge cases properly.",
        "You are a coding assistant. Generate Python functions as requested."
    ]
    
    print(f"\n🔬 Testing {len(test_prompts)} prompts with task-specific evaluation:")
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n--- Testing Prompt {i} ---")
        print(f"Prompt: {prompt}")
        
        # Evaluate with task-specific fitness
        fitness_score = optimizer.fitness_evaluator.evaluate_prompt(
            prompt, "EXECUTOR", None
        )
        
        print(f"Task-Specific Fitness Score: {fitness_score:.3f}")
        
        # Get detailed metrics
        metrics = optimizer.fitness_evaluator.get_detailed_metrics(prompt, "EXECUTOR")
        if metrics:
            print(f"  Success Rate: {metrics.success_rate:.3f}")
            print(f"  Efficiency: {metrics.efficiency:.3f}")
            print(f"  Coherence: {metrics.coherence:.3f}")
            print(f"  Task Scores: {metrics.task_scores}")
    
    print(f"\n✅ Task-specific fitness evaluation test completed")
    return True

def test_error_targeted_prompt_generation():
    """Test the error-targeted prompt generation system."""
    print("\n" + "="*80)
    print("🎯 TESTING ERROR-TARGETED PROMPT GENERATION")
    print("="*80)
    
    # Configure environment
    provider = os.environ.get("LLM_PROVIDER", "deepseek")
    model_name = os.environ.get("LLM_MODEL", "deepseek-coder")
    
    # Initialize LLM service
    try:
        llm_service = LLMService(provider=provider, model_name=model_name)
        print(f"✅ LLM Service: {provider}/{model_name}")
    except Exception as e:
        print(f"❌ LLM Service Error: {e}")
        return False
    
    # Create evolution operators
    from self_healing_agents.evolution.evolution_operators import EvolutionOperators
    operators = EvolutionOperators(llm_service)
    
    # Test different failure contexts
    failure_contexts = [
        {
            "original_task": "Implement regex matching",
            "failure_report": {
                "overall_status": "FAILURE",
                "quantitative_score": 0.2,
                "feedback": "The implementation has syntax errors and doesn't handle edge cases properly"
            },
            "classification": {
                "primary_failure_type": "syntax_error",
                "reasoning": ["Missing return statement", "Incorrect function definition", "Syntax issues"]
            }
        },
        {
            "original_task": "Implement regex matching",
            "failure_report": {
                "overall_status": "FAILURE",
                "quantitative_score": 0.4,
                "feedback": "The logic is incorrect and doesn't handle the '*' pattern correctly"
            },
            "classification": {
                "primary_failure_type": "logic_error",
                "reasoning": ["Incorrect algorithm", "Misunderstood requirements", "Logic flaws"]
            }
        },
        {
            "original_task": "Implement regex matching",
            "failure_report": {
                "overall_status": "FAILURE",
                "quantitative_score": 0.6,
                "feedback": "Implementation is incomplete and missing edge case handling"
            },
            "classification": {
                "primary_failure_type": "incomplete_implementation",
                "reasoning": ["Missing edge cases", "Incomplete solution", "Partial implementation"]
            }
        }
    ]
    
    base_prompt = "You are a Python programmer. Write code according to the specification."
    
    print(f"\n🔧 Testing error-targeted mutations:")
    
    for i, failure_context in enumerate(failure_contexts, 1):
        error_type = failure_context["classification"]["primary_failure_type"]
        print(f"\n--- Test {i}: {error_type.replace('_', ' ').title()} ---")
        
        # Apply error-targeted mutation
        improved_prompt = operators.mutate(
            base_prompt, 
            "EXECUTOR", 
            mutation_rate=0.8,
            failure_context=failure_context
        )
        
        print(f"Original Prompt: {base_prompt}")
        print(f"Error Type: {error_type}")
        print(f"Error Details: {failure_context['failure_report']['feedback']}")
        print(f"Improved Prompt: {improved_prompt}")
        print(f"Improvement Length: {len(improved_prompt) - len(base_prompt)} characters")
    
    print(f"\n🧬 Testing error-targeted crossover:")
    
    parent1 = "You are a Python programmer. Write clean, efficient code."
    parent2 = "You are a software engineer. Focus on correctness and edge cases."
    
    for i, failure_context in enumerate(failure_contexts[:2], 1):
        error_type = failure_context["classification"]["primary_failure_type"]
        print(f"\n--- Crossover Test {i}: {error_type.replace('_', ' ').title()} ---")
        
        # Apply error-targeted crossover
        offspring = operators.crossover(
            parent1, 
            parent2, 
            "EXECUTOR",
            failure_context=failure_context
        )
        
        print(f"Parent 1: {parent1}")
        print(f"Parent 2: {parent2}")
        print(f"Error Type: {error_type}")
        print(f"Offspring: {offspring}")
    
    print(f"\n✅ Error-targeted prompt generation test completed")
    return True

def test_full_evolutionary_optimization():
    """Test the complete evolutionary optimization with error targeting."""
    print("\n" + "="*80)
    print("🧬 TESTING FULL EVOLUTIONARY OPTIMIZATION")
    print("="*80)
    
    # Configure environment
    provider = os.environ.get("LLM_PROVIDER", "deepseek")
    model_name = os.environ.get("LLM_MODEL", "deepseek-coder")
    
    # Initialize LLM service
    try:
        llm_service = LLMService(provider=provider, model_name=model_name)
        print(f"✅ LLM Service: {provider}/{model_name}")
    except Exception as e:
        print(f"❌ LLM Service Error: {e}")
        return False
    
    # Initialize agents
    planner = Planner("TestPlanner", llm_service, PLANNER_SYSTEM_PROMPT)
    executor = Executor("TestExecutor", llm_service, DEFAULT_EXECUTOR_SYSTEM_PROMPT)
    critic = Critic("TestCritic", llm_service)
    
    # Create evolutionary optimizer with small population for testing
    config = EvolutionConfig(
        population_size=3,
        max_generations=2,
        max_evaluations=15,
        early_stopping_patience=2
    )
    optimizer = EvolutionaryPromptOptimizer(llm_service, config)
    
    # Set up task-specific context
    regex_task = """Given an input string s and a pattern p, implement regular expression matching with support for '.' and '*' where:

'.' Matches any single character.
'*' Matches zero or more of the preceding element.
The matching should cover the entire input string (not partial)."""
    
    print(f"📋 Setting up evolutionary optimization for regex matching")
    optimizer.fitness_evaluator.set_task_specific_context(
        task_description=regex_task,
        planner_agent=planner,
        executor_agent=executor,
        critic_agent=critic
    )
    
    # Create failure context for error-targeted evolution
    failure_context = {
        "original_task": regex_task,
        "failure_report": {
            "overall_status": "FAILURE",
            "quantitative_score": 0.3,
            "feedback": "The implementation has logic errors and doesn't handle the '*' pattern correctly"
        },
        "classification": {
            "primary_failure_type": "logic_error",
            "reasoning": ["Incorrect algorithm", "Misunderstood requirements", "Logic flaws"]
        }
    }
    
    # Run evolutionary optimization
    base_prompt = "You are a Python programmer. Write code according to the specification."
    
    print(f"\n🚀 Running evolutionary optimization...")
    print(f"Base Prompt: {base_prompt}")
    print(f"Target Error: {failure_context['classification']['primary_failure_type']}")
    
    results = optimizer.optimize_prompt(
        base_prompt=base_prompt,
        agent_type="EXECUTOR",
        failure_context=failure_context
    )
    
    # Display results
    print(f"\n📊 EVOLUTIONARY OPTIMIZATION RESULTS:")
    print(f"✅ Best Fitness: {results.best_fitness:.3f}")
    print(f"🔄 Generations: {results.generation_count}")
    print(f"📈 Evaluations: {results.evaluation_count}")
    print(f"⏱️  Time: {results.execution_time:.1f}s")
    print(f"🛑 Termination: {results.termination_reason}")
    
    print(f"\n📝 EVOLVED PROMPT:")
    print(f"Original: {base_prompt}")
    print(f"Evolved:  {results.best_prompt}")
    
    improvement = results.best_fitness - (results.convergence_history[0] if results.convergence_history else 0)
    print(f"\n📈 FITNESS IMPROVEMENT: {improvement:+.3f}")
    
    if results.convergence_history:
        print(f"📊 Convergence History: {[f'{f:.3f}' for f in results.convergence_history]}")
    
    print(f"\n✅ Full evolutionary optimization test completed")
    return True

def main():
    """Run all tests."""
    setup_logging()
    
    print("🧬 IMPROVED EVOLUTIONARY PROMPT OPTIMIZATION SYSTEM TEST")
    print("=" * 80)
    print("🎯 Features being tested:")
    print("  1. Task-specific fitness evaluation with direct regex testing")
    print("  2. Error-targeted prompt generation addressing specific failures")
    print("  3. Holistic prompt improvements (not just error appending)")
    print("  4. Full evolutionary optimization pipeline")
    print("=" * 80)
    
    tests = [
        ("Task-Specific Fitness Evaluation", test_task_specific_fitness_evaluation),
        ("Error-Targeted Prompt Generation", test_error_targeted_prompt_generation),
        ("Full Evolutionary Optimization", test_full_evolutionary_optimization)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            print(f"\n🧪 Running: {test_name}")
            success = test_func()
            results.append((test_name, success))
            if success:
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*80)
    print("📊 TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The improved evolutionary system is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the logs for details.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 