#!/usr/bin/env python3
"""
Script to demonstrate detailed prompt evolution process.
"""

import os
import sys
from pathlib import Path

# Add src to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Set environment
os.environ['LLM_PROVIDER'] = 'deepseek'
os.environ['LLM_MODEL'] = 'deepseek-coder'

from self_healing_agents.evolution import EvolutionaryPromptOptimizer, EvolutionConfig
from self_healing_agents.llm_service import LLMService
from self_healing_agents.prompts import ULTRA_BUGGY_PROMPT


def show_prompt_evolution():
    """Demonstrate the prompt evolution process."""
    
    print("🧬 DETAILED PROMPT EVOLUTION DEMONSTRATION")
    print("=" * 60)
    
    # Initialize components
    llm_service = LLMService(provider='deepseek', model_name='deepseek-coder')
    
    # Configure for detailed observation
    config = EvolutionConfig(
        population_size=3,
        max_generations=2, 
        max_evaluations=12,
        validation_tasks_count=1,
        success_rate_weight=0.4,
        efficiency_weight=0.2,
        coherence_weight=0.3,
        generalization_weight=0.1
    )
    
    optimizer = EvolutionaryPromptOptimizer(llm_service, config)
    
    # Show original prompt
    original_prompt = ULTRA_BUGGY_PROMPT[:300] + "..."  # Truncate for display
    print(f"\n📝 ORIGINAL PROMPT (Ultra Buggy - {len(ULTRA_BUGGY_PROMPT)} chars):")
    print(f"{'─' * 50}")
    print(original_prompt)
    print(f"{'─' * 50}")
    
    # Show fitness formula
    print(f"\n🧮 FITNESS CALCULATION FORMULA:")
    print(f"F(prompt) = {config.success_rate_weight}×success + {config.efficiency_weight}×efficiency + {config.coherence_weight}×coherence + {config.generalization_weight}×generalization")
    print(f"Where each component is scored 0.0-1.0")
    
    # Run evolution
    print(f"\n🔄 RUNNING EVOLUTION...")
    print(f"   Population: {config.population_size} individuals")
    print(f"   Max generations: {config.max_generations}")
    print(f"   Evaluation budget: {config.max_evaluations}")
    
    results = optimizer.optimize_prompt(
        base_prompt=ULTRA_BUGGY_PROMPT,
        agent_type="EXECUTOR"
    )
    
    # Show evolved prompt
    print(f"\n✨ EVOLVED PROMPT ({len(results.best_prompt)} chars):")
    print(f"{'─' * 50}")
    print(results.best_prompt[:500] + ("..." if len(results.best_prompt) > 500 else ""))
    print(f"{'─' * 50}")
    
    # Show detailed metrics
    print(f"\n📊 EVOLUTION METRICS:")
    print(f"   🏆 Final Fitness: {results.best_fitness:.3f}")
    print(f"   🔄 Generations: {results.generation_count}")
    print(f"   📊 Evaluations: {results.evaluation_count}")
    print(f"   ⏱️  Time: {results.execution_time:.1f}s")
    print(f"   🛑 Termination: {results.termination_reason}")
    
    # Show improvement analysis
    original_length = len(ULTRA_BUGGY_PROMPT)
    evolved_length = len(results.best_prompt)
    length_change = ((evolved_length - original_length) / original_length) * 100
    
    print(f"\n📈 IMPROVEMENT ANALYSIS:")
    print(f"   📏 Length change: {length_change:+.1f}%")
    print(f"   📊 Fitness convergence:")
    
    for i, fitness in enumerate(results.convergence_history):
        print(f"      Gen {i}: {fitness:.3f}")
    
    # Show what changed
    print(f"\n🔍 KEY CHANGES IN EVOLVED PROMPT:")
    
    # Simple analysis of changes
    original_words = set(ULTRA_BUGGY_PROMPT.lower().split())
    evolved_words = set(results.best_prompt.lower().split())
    
    new_words = evolved_words - original_words
    removed_words = original_words - evolved_words
    
    if new_words:
        print(f"   ➕ Added concepts: {', '.join(list(new_words)[:10])}")
    if removed_words:
        print(f"   ➖ Removed concepts: {', '.join(list(removed_words)[:10])}")
    
    print(f"\n🎉 Evolution demonstration complete!")
    return results


if __name__ == "__main__":
    show_prompt_evolution() 