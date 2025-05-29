#!/usr/bin/env python3
"""
Simple test script to verify the evolutionary prompt optimization system.
"""

import sys
import os
import logging
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from self_healing_agents.llm_service import LLMService
from self_healing_agents.evolution.evolutionary_prompt_optimizer import EvolutionaryPromptOptimizer
from self_healing_agents.evolution.evolution_config import EvolutionConfig
from self_healing_agents.agents import Executor, Planner

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_evolutionary_optimization():
    """Test the evolutionary prompt optimization with a simple failure context."""
    
    print("🧬 Testing Evolutionary Prompt Optimization...")
    
    # Initialize LLM service
    llm_service = LLMService(
        provider="deepseek",
        model_name="deepseek-chat",
        temperature=0.7,
        max_tokens=2048
    )
    
    # Configuration for quick test
    config = EvolutionConfig(
        population_size=4,
        max_generations=2,
        max_evaluations=10,
        crossover_rate=0.7,
        mutation_rate=0.8
    )
    
    # Create optimizer
    optimizer = EvolutionaryPromptOptimizer(llm_service, config)
    
    # 🆕 CRITICAL: Set up task-specific fitness evaluation for regex matching
    task_description = 'Implement regular expression matching with support for "." and "*" where "." matches any single character and "*" matches zero or more of the preceding element.'
    
    # Configure the fitness evaluator to use the actual regex task
    # For testing, we'll create both planner and executor agents for evaluation
    planner_agent = Planner("test_planner", llm_service)
    executor_agent = Executor("test_executor", llm_service)
    
    optimizer.fitness_evaluator.set_task_specific_context(
        task_description=task_description,
        # Provide both agents for complete evaluation
        planner_agent=planner_agent,
        executor_agent=executor_agent,
        critic_agent=None  # Will use built-in test evaluation
    )
    
    logger.info(f"🎯 TASK-SPECIFIC FITNESS: Configured for regex matching task")
    
    # Base executor prompt (intentionally basic)
    base_prompt = "You are a programmer. Output only the raw Python code without any markdown formatting or explanations."
    
    # Create failure context based on the regex matching failure we saw in the logs
    failure_context = {
        'original_task': 'Implement regular expression matching with support for "." and "*"',
        'failure_report': {
            'status': 'FAILURE_LOGIC',
            'score': 0.65,
            'feedback': 'Failed test: test_star_matches_zero_elements. Expected True for s="a", p="ab*a" but got False.'
        },
        'classification': {
            'primary_failure_type': 'logic_error',
            'reasoning': ['Incorrect handling of star operator with zero matches', 'Pattern matching logic needs improvement']
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
    
    print(f"   📋 Base Prompt: {base_prompt}")
    print(f"   🎯 Failure Context: {failure_context['classification']['primary_failure_type']}")
    print(f"   🧪 Failed Test: {failure_context['specific_test_failures'][0]['test_name']}")
    
    # Run evolutionary optimization
    try:
        results = optimizer.optimize_prompt(
            base_prompt=base_prompt,
            agent_type="EXECUTOR",
            failure_context=failure_context
        )
        
        print(f"\n✅ EVOLUTION COMPLETE!")
        print(f"   📊 Final Status: {results.termination_reason}")
        print(f"   📈 Best Fitness: {results.best_fitness:.3f}")
        print(f"   🔄 Generations: {results.generation_count}")
        print(f"   🧪 Evaluations: {results.evaluation_count}")
        print(f"   ⏱️ Time: {results.execution_time:.2f}s")
        print(f"\n🎯 BEST EVOLVED PROMPT:")
        print(f"   Length: {len(results.best_prompt)} characters")
        print(f"   Preview: {results.best_prompt[:200]}...")
        
        # Check if we got strategic improvements
        best_prompt_lower = results.best_prompt.lower()
        strategic_indicators = {
            'persona': any(term in best_prompt_lower for term in ['expert', 'specialist', 'experienced', 'senior']),
            'cot': any(term in best_prompt_lower for term in ['step-by-step', 'think', 'reasoning', 'analyze']),
            'regex_specific': any(term in best_prompt_lower for term in ['regex', 'pattern', 'matching', 'star', 'zero']),
            'error_specific': any(term in best_prompt_lower for term in ['edge case', 'boundary', 'zero match'])
        }
        
        print(f"\n🎯 STRATEGIC IMPROVEMENTS DETECTED:")
        for strategy, detected in strategic_indicators.items():
            status = "✅" if detected else "❌"
            print(f"   {status} {strategy.upper()}: {detected}")
            
        return results
        
    except Exception as e:
        print(f"❌ EVOLUTION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = test_evolutionary_optimization()
    if result:
        print(f"\n🎉 TEST PASSED: Evolutionary optimization completed successfully!")
    else:
        print(f"\n💥 TEST FAILED: Evolutionary optimization did not complete.") 