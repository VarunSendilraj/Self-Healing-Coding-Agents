#!/usr/bin/env python3
"""
Diagnostic Script: Analyze Why Evolutionary Optimization Converges to Worse Prompts

This script investigates the fitness evaluation and selection mechanisms
to understand why technically superior prompts are not retained.
"""

import sys
import os
import logging
from typing import Dict, List, Any

# Setup path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from self_healing_agents.agents import Planner, Executor, Critic
from self_healing_agents.llm_service import LLMService
from self_healing_agents.prompts import DEFAULT_EXECUTOR_SYSTEM_PROMPT, ULTRA_BUGGY_PROMPT, PLANNER_SYSTEM_PROMPT
from self_healing_agents.evolution import EvolutionaryPromptOptimizer, EvolutionConfig
from self_healing_agents.evolution.fitness_evaluator import PromptFitnessEvaluator, FitnessMetrics

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_fitness_calculation_bias():
    """Analyze potential biases in fitness calculation."""
    print("\n🔍 DIAGNOSTIC 1: FITNESS CALCULATION BIAS ANALYSIS")
    print("=" * 70)
    
    # Test fitness components
    from self_healing_agents.evolution.evolution_config import EvolutionConfig
    
    config = EvolutionConfig()
    
    # Simulate different metric combinations
    test_cases = [
        ("High Success, Low Others", FitnessMetrics(success_rate=0.9, efficiency=0.1, coherence=0.1, generalization=0.1)),
        ("Low Success, High Others", FitnessMetrics(success_rate=0.1, efficiency=0.9, coherence=0.9, generalization=0.9)),
        ("Balanced Good", FitnessMetrics(success_rate=0.7, efficiency=0.7, coherence=0.7, generalization=0.7)),
        ("Balanced Poor", FitnessMetrics(success_rate=0.3, efficiency=0.3, coherence=0.3, generalization=0.3)),
    ]
    
    print(f"Config weights: Success={config.success_rate_weight}, Efficiency={config.efficiency_weight}, "
          f"Coherence={config.coherence_weight}, Generalization={config.generalization_weight}")
    
    weight_sum = (config.success_rate_weight + config.efficiency_weight + 
                 config.coherence_weight + config.generalization_weight)
    print(f"⚠️ WEIGHT SUM: {weight_sum:.3f} (should be 1.0 for normalized scoring)")
    
    for name, metrics in test_cases:
        fitness = (
            metrics.success_rate * config.success_rate_weight +
            metrics.efficiency * config.efficiency_weight +
            metrics.coherence * config.coherence_weight +
            metrics.generalization * config.generalization_weight
        )
        
        print(f"\n{name}:")
        print(f"  Success: {metrics.success_rate:.3f} × {config.success_rate_weight:.3f} = {metrics.success_rate * config.success_rate_weight:.3f}")
        print(f"  Efficiency: {metrics.efficiency:.3f} × {config.efficiency_weight:.3f} = {metrics.efficiency * config.efficiency_weight:.3f}")
        print(f"  Coherence: {metrics.coherence:.3f} × {config.coherence_weight:.3f} = {metrics.coherence * config.coherence_weight:.3f}")
        print(f"  Generalization: {metrics.generalization:.3f} × {config.generalization_weight:.3f} = {metrics.generalization * config.generalization_weight:.3f}")
        print(f"  🎯 TOTAL FITNESS: {fitness:.3f}")
        
    print(f"\n💡 INSIGHT: If weights don't sum to 1.0, fitness scores will be biased!")


def analyze_selection_pressure():
    """Analyze selection pressure and elite preservation."""
    print("\n🔍 DIAGNOSTIC 2: SELECTION PRESSURE ANALYSIS")
    print("=" * 70)
    
    config = EvolutionConfig()
    population_size = config.population_size
    elite_rate = config.elite_preservation_rate
    
    num_elites = max(1, int(population_size * elite_rate))
    
    print(f"Population size: {population_size}")
    print(f"Elite preservation rate: {elite_rate:.1%}")
    print(f"Number of elites preserved: {num_elites}")
    print(f"Number of non-elites competing: {population_size - num_elites}")
    
    # Simulate selection with different fitness distributions
    import numpy as np
    
    scenarios = [
        ("Uniform Distribution", np.linspace(0.1, 0.9, population_size)),
        ("High Variance", np.array([0.9, 0.8, 0.2, 0.15, 0.1])),
        ("Low Variance", np.array([0.6, 0.59, 0.58, 0.57, 0.56])),
        ("One Outlier", np.array([0.95, 0.3, 0.3, 0.3, 0.3]))
    ]
    
    for scenario_name, fitness_scores in scenarios:
        print(f"\n{scenario_name}: {fitness_scores}")
        sorted_indices = np.argsort(fitness_scores)[::-1]  # Highest first
        sorted_scores = fitness_scores[sorted_indices]
        
        elites = sorted_scores[:num_elites]
        candidates = sorted_scores[num_elites:]
        
        print(f"  Elites (auto-preserved): {elites}")
        print(f"  Competing for remaining slots: {candidates}")
        
        if len(candidates) > 0:
            selection_pressure = np.max(elites) / np.mean(candidates) if np.mean(candidates) > 0 else float('inf')
            print(f"  Selection pressure ratio: {selection_pressure:.2f}")


def analyze_prompt_caching_issues():
    """Check if prompt caching is causing evaluation inconsistencies."""
    print("\n🔍 DIAGNOSTIC 3: PROMPT CACHING ANALYSIS")
    print("=" * 70)
    
    test_prompts = [
        "You are a programmer. Write code.",
        "You are a programmer. Write code.",  # Identical
        "You are a programmer. Write good code.",  # Slightly different
        "You are a programmer.\n\nWrite code.",  # Whitespace difference
    ]
    
    # Simulate cache key generation
    import hashlib
    
    print("Testing cache key generation:")
    for i, prompt in enumerate(test_prompts):
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:16]
        cache_key = f"EXECUTOR_{prompt_hash}"
        print(f"  Prompt {i+1}: '{prompt}' -> {cache_key}")
    
    # Check for potential cache collisions
    hashes = [hashlib.md5(p.encode()).hexdigest()[:16] for p in test_prompts]
    if len(set(hashes)) != len(hashes):
        print("⚠️ CACHE COLLISION DETECTED!")
    else:
        print("✅ No cache collisions detected")


def analyze_evolution_operators_bias():
    """Analyze potential biases in crossover and mutation operators."""
    print("\n🔍 DIAGNOSTIC 4: EVOLUTION OPERATORS BIAS")
    print("=" * 70)
    
    config = EvolutionConfig()
    
    print(f"Crossover rate: {config.crossover_rate:.1%}")
    print(f"Mutation rate: {config.mutation_rate:.1%}")
    print(f"Adaptive mutation: {config.adaptive_mutation}")
    print(f"Min diversity threshold: {config.min_diversity_threshold}")
    
    # Check probability of no changes
    prob_no_crossover = 1 - config.crossover_rate
    prob_no_mutation = 1 - config.mutation_rate
    prob_no_change = prob_no_crossover * prob_no_mutation
    
    print(f"\nProbability analysis:")
    print(f"  No crossover: {prob_no_crossover:.1%}")
    print(f"  No mutation: {prob_no_mutation:.1%}")
    print(f"  No change at all: {prob_no_change:.1%}")
    
    if prob_no_change > 0.3:
        print("⚠️ HIGH CHANCE OF NO EVOLUTION: Many offspring may be identical to parents")


def test_actual_vs_estimated_fitness():
    """Test the disconnect between evolutionary fitness and actual task performance."""
    print("\n🔍 DIAGNOSTIC 5: ACTUAL VS ESTIMATED FITNESS")
    print("=" * 70)
    
    # Initialize LLM service
    try:
        llm_service = LLMService(provider="deepseek", model_name="deepseek-coder")
        print("✅ LLM Service initialized")
    except Exception as e:
        print(f"❌ LLM Service failed: {e}")
        return
    
    # Initialize agents
    planner = Planner("TestPlanner", llm_service, PLANNER_SYSTEM_PROMPT)
    executor = Executor("TestExecutor", llm_service, DEFAULT_EXECUTOR_SYSTEM_PROMPT)
    critic = Critic("TestCritic", llm_service)
    
    # Test task
    test_task_description = """Given an input string s and a pattern p, implement regular expression matching with support for '.' and '*' where:

'.' Matches any single character.
'*' Matches zero or more of the preceding element.
The matching should cover the entire input string (not partial).

Example 1:
Input: s = "aa", p = "a"
Output: false
Explanation: "a" does not match the entire string "aa"."""
    
    # Test prompts of varying quality
    test_prompts = [
        ("Original Good", DEFAULT_EXECUTOR_SYSTEM_PROMPT),
        ("Simple Bad", "You are a programmer. Output only the raw Python code."),
        ("Verbose Good", """You are an expert Python programmer specializing in algorithm implementation.

Your task is to implement clean, efficient, and well-documented Python code that solves the given problem.

Key requirements:
1. Write only Python code - no explanations or markdown
2. Include proper variable names and clear logic
3. Handle edge cases appropriately
4. Ensure the solution is correct and efficient
5. Use appropriate data structures and algorithms

Focus on correctness, readability, and efficiency."""),
    ]
    
    # Create fitness evaluator
    config = EvolutionConfig(validation_tasks_count=1)
    fitness_evaluator = PromptFitnessEvaluator(config, llm_service)
    
    # Set up task-specific context
    fitness_evaluator.set_task_specific_context(
        task_description=test_task_description,
        planner_agent=planner,
        executor_agent=executor,
        critic_agent=critic
    )
    
    print("Testing prompts on actual task:")
    for name, prompt in test_prompts:
        print(f"\n--- {name} ---")
        print(f"Prompt: {prompt}")
        
        # Get evolutionary fitness
        evo_fitness = fitness_evaluator.evaluate_prompt(prompt, "EXECUTOR", None)
        
        # Get actual task performance
        original_prompt = executor.system_prompt
        try:
            executor.system_prompt = prompt
            
            # Run actual pipeline
            plan = planner.run(user_request=test_task_description)
            if isinstance(plan, dict) and plan.get("error"):
                actual_score = 0.0
                print(f"  Planning failed: {plan.get('error')}")
            else:
                code = executor.run(plan=plan, original_request=test_task_description)
                if isinstance(code, dict) and code.get("error"):
                    actual_score = 0.0
                    print(f"  Execution failed: {code.get('error')}")
                else:
                    critique = critic.run(
                        generated_code=code,
                        task_description=test_task_description,
                        plan=plan
                    )
                    if isinstance(critique, dict):
                        actual_score = critique.get('quantitative_score', critique.get('score', 0.0))
                    else:
                        actual_score = 0.0
                        
        except Exception as e:
            actual_score = 0.0
            print(f"  Pipeline failed: {e}")
        finally:
            executor.system_prompt = original_prompt
        
        print(f"  🧬 Evolutionary Fitness: {evo_fitness:.3f}")
        print(f"  🎯 Actual Task Score: {actual_score:.3f}")
        print(f"  📊 Difference: {evo_fitness - actual_score:+.3f}")
        
        if abs(evo_fitness - actual_score) > 0.2:
            print(f"  ⚠️ LARGE DISCONNECT!")


def propose_fixes():
    """Propose specific fixes for the convergence problem."""
    print("\n💡 PROPOSED FIXES FOR CONVERGENCE PROBLEM")
    print("=" * 70)
    
    fixes = [
        ("1. Fix Weight Normalization", 
         "Ensure fitness weights sum to 1.0 for proper score normalization"),
        
        ("2. Increase Task-Specific Weight", 
         "Give higher weight to success_rate when using actual task evaluation"),
        
        ("3. Add Fitness Decay Penalty", 
         "Penalize prompts that don't improve over multiple generations"),
        
        ("4. Improve Selection Diversity", 
         "Use tournament selection or Pareto ranking instead of pure fitness"),
        
        ("5. Add Momentum to Best Solutions", 
         "Track and preserve the globally best prompt across generations"),
        
        ("6. Fix Cache Invalidation", 
         "Clear fitness cache when switching to task-specific evaluation"),
        
        ("7. Implement Fitness Validation", 
         "Re-evaluate top prompts with actual task to confirm fitness"),
        
        ("8. Add Convergence Detection", 
         "Stop evolution when population becomes too similar"),
        
        ("9. Use Multi-Objective Optimization", 
         "Optimize for both general fitness and task-specific performance"),
        
        ("10. Add Fitness Smoothing", 
         "Average fitness across multiple evaluations to reduce noise")
    ]
    
    for title, description in fixes:
        print(f"{title}: {description}")


def main():
    """Run comprehensive convergence diagnostics."""
    print("🚨 EVOLUTIONARY CONVERGENCE PROBLEM DIAGNOSTICS")
    print("=" * 70)
    print("Analyzing why technically better prompts are not retained...")
    
    try:
        analyze_fitness_calculation_bias()
        analyze_selection_pressure()
        analyze_prompt_caching_issues()
        analyze_evolution_operators_bias()
        test_actual_vs_estimated_fitness()
        propose_fixes()
        
    except Exception as e:
        logger.error(f"Diagnostic failed: {e}")
    
    print(f"\n🎉 DIAGNOSTICS COMPLETE!")
    print("Review the analysis above to understand convergence issues.")


if __name__ == "__main__":
    main() 