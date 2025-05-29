#!/usr/bin/env python3
"""
Script to examine the evolved prompt from the evolutionary system.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from self_healing_agents.llm_service import LLMService
from self_healing_agents.evolution.evolutionary_prompt_optimizer import EvolutionaryPromptOptimizer
from self_healing_agents.evolution.evolution_config import EvolutionConfig
from self_healing_agents.agents import Executor, Planner

def examine_evolved_prompt():
    """Examine what the evolutionary system produces."""
    
    print("🔍 Examining Evolved Prompt...")
    
    # Initialize LLM service
    llm_service = LLMService(
        provider="deepseek",
        model_name="deepseek-chat",
        temperature=0.7,
        max_tokens=2048
    )
    
    # Quick config for fast test
    config = EvolutionConfig(
        population_size=3,  # Fixed: minimum 3 for proper evolution
        max_generations=2,  # Increased to see evolution in action
        max_evaluations=10  # Increased budget
    )
    
    # Create optimizer
    optimizer = EvolutionaryPromptOptimizer(llm_service, config)
    
    # 🆕 CRITICAL: Set up task-specific fitness evaluation for regex matching
    task_description = 'Implement regular expression matching with support for "." and "*" where "." matches any single character and "*" matches zero or more of the preceding element.'
    
    # Configure the fitness evaluator to use the actual regex task
    planner_agent = Planner("test_planner", llm_service)
    executor_agent = Executor("test_executor", llm_service)
    
    optimizer.fitness_evaluator.set_task_specific_context(
        task_description=task_description,
        planner_agent=planner_agent,
        executor_agent=executor_agent,
        critic_agent=None  # Will use built-in test evaluation
    )
    
    print(f"🎯 TASK-SPECIFIC FITNESS: Configured for regex matching task")
    
    # Base prompt
    base_prompt = "You are a programmer. Output only the raw Python code without any markdown formatting or explanations."
    
    # Failure context with specific regex issue
    failure_context = {
        'original_task': 'Implement regular expression matching with support for "." and "*"',
        'failure_report': {
            'status': 'FAILURE_LOGIC',
            'score': 0.65,
            'feedback': 'Failed test: test_star_matches_zero_elements. Expected True for s="a", p="ab*a" but got False.'
        },
        'classification': {
            'primary_failure_type': 'logic_error',
            'reasoning': ['Incorrect handling of star operator with zero matches']
        },
        'specific_test_failures': [
            {
                'test_name': 'test_star_matches_zero_elements',
                'inputs': {'s': 'a', 'p': 'ab*a'},
                'expected_output': True,
                'actual_output': False,
                'regex_pattern_issue': {
                    'pattern': 'ab*a',
                    'string': 'a',
                    'likely_issue': 'star_zero_match_handling'
                }
            }
        ]
    }
    
    print(f"📋 Base Prompt: {base_prompt}")
    print(f"🎯 Target Issue: {failure_context['classification']['primary_failure_type']}")
    
    # Run evolution
    results = optimizer.optimize_prompt(
        base_prompt=base_prompt,
        agent_type="EXECUTOR",
        failure_context=failure_context
    )
    
    print(f"\n🎯 FINAL EVOLVED PROMPT:")
    print("="*80)
    print(results.best_prompt)
    print("="*80)
    
    # Analyze strategic improvements
    evolved_prompt_lower = results.best_prompt.lower()
    
    strategic_analysis = {
        'persona_enhancement': any(term in evolved_prompt_lower for term in ['expert', 'specialist', 'experienced', 'senior']),
        'chain_of_thought': any(term in evolved_prompt_lower for term in ['step-by-step', 'think', 'reasoning', 'analyze']),
        'regex_specific': any(term in evolved_prompt_lower for term in ['regex', 'pattern', 'matching', 'star', 'zero']),
        'error_specific': any(term in evolved_prompt_lower for term in ['edge case', 'boundary', 'zero match']),
        'algorithm_guidance': any(term in evolved_prompt_lower for term in ['algorithm', 'logic', 'implementation']),
        'examples_included': 'example' in evolved_prompt_lower or 'demonstration' in evolved_prompt_lower
    }
    
    print(f"\n🎯 STRATEGIC IMPROVEMENTS ANALYSIS:")
    for strategy, detected in strategic_analysis.items():
        status = "✅" if detected else "❌"
        print(f"   {status} {strategy.upper()}: {detected}")
    
    # Check length and complexity
    print(f"\n📊 PROMPT METRICS:")
    print(f"   📏 Length: {len(results.best_prompt)} characters")
    print(f"   📈 Improvement: {len(results.best_prompt) - len(base_prompt):+d} characters")
    print(f"   🏆 Fitness: {results.best_fitness:.3f}")
    print(f"   🔄 Generations: {results.generation_count}")
    
    return results

if __name__ == "__main__":
    examine_evolved_prompt() 