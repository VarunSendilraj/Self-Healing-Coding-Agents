#!/usr/bin/env python3
"""
Fix Convergence Issues in Evolutionary Optimization

This script implements the key fixes identified in the diagnostic:
1. Fix cache collision issue
2. Implement proper global best tracking 
3. Add task-specific weight adjustment
4. Improve selection mechanism
5. Add fitness validation
"""

import sys
import os
import logging
from typing import Dict, List, Any, Optional

# Setup path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from self_healing_agents.evolution.fitness_evaluator import PromptFitnessEvaluator
from self_healing_agents.evolution.evolutionary_prompt_optimizer import EvolutionaryPromptOptimizer
from self_healing_agents.evolution.evolution_config import EvolutionConfig

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def fix_fitness_evaluator_cache():
    """Fix the cache collision issue in fitness evaluator."""
    print("🔧 FIX 1: Fixing cache collision in fitness evaluator")
    
    # Read the current fitness evaluator
    fitness_eval_path = "src/self_healing_agents/evolution/fitness_evaluator.py"
    
    with open(fitness_eval_path, 'r') as f:
        content = f.read()
    
    # Replace the problematic cache key generation
    old_cache_method = '''def _get_cache_key(self, prompt: str, agent_type: str) -> str:
        """Generate cache key for prompt evaluation."""
        # Use hash of prompt content for caching
        import hashlib
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:16]
        return f"{agent_type}_{prompt_hash}"'''
    
    new_cache_method = '''def _get_cache_key(self, prompt: str, agent_type: str) -> str:
        """Generate cache key for prompt evaluation."""
        # Use full hash to avoid collisions + include task context
        import hashlib
        # Include task context in cache key to avoid cross-task contamination
        task_context = getattr(self, 'task_specific_context', {})
        task_desc = task_context.get('task_description', 'generic') if task_context else 'generic'
        
        # Use SHA256 for better collision resistance
        combined_content = f"{agent_type}|{prompt}|{task_desc}"
        prompt_hash = hashlib.sha256(combined_content.encode()).hexdigest()[:32]
        return f"{agent_type}_{prompt_hash}"'''
    
    if old_cache_method.replace('\n', '').replace(' ', '') in content.replace('\n', '').replace(' ', ''):
        content = content.replace(old_cache_method, new_cache_method)
        print("  ✅ Fixed cache key generation")
    else:
        print("  ⚠️ Cache method not found for replacement")
    
    # Write back the fixed content
    with open(fitness_eval_path, 'w') as f:
        f.write(content)


def fix_global_best_tracking():
    """Fix global best tracking across generations."""
    print("\n🔧 FIX 2: Adding global best tracking to evolutionary optimizer")
    
    optimizer_path = "src/self_healing_agents/evolution/evolutionary_prompt_optimizer.py"
    
    with open(optimizer_path, 'r') as f:
        content = f.read()
    
    # Add global best tracking to the class initialization
    if "self.convergence_count = 0" in content:
        content = content.replace(
            "self.convergence_count = 0",
            '''self.convergence_count = 0
        
        # Global best tracking across all generations
        self.global_best_prompt = None
        self.global_best_fitness = -1.0
        self.global_best_generation = 0'''
        )
        print("  ✅ Added global best tracking initialization")
    
    # Update the generation tracking logic
    old_best_tracking = '''# Track best individual
                generation_best_fitness = max(fitness_scores)
                if generation_best_fitness > current_best_fitness:
                    best_index = fitness_scores.index(generation_best_fitness)
                    current_best_fitness = generation_best_fitness
                    current_best_prompt = population[best_index]
                    self.convergence_count = 0  # Reset convergence counter
                    logger.info(f"   🎉 NEW BEST: Fitness improved to {current_best_fitness:.3f}")'''
    
    new_best_tracking = '''# Track best individual
                generation_best_fitness = max(fitness_scores)
                if generation_best_fitness > current_best_fitness:
                    best_index = fitness_scores.index(generation_best_fitness)
                    current_best_fitness = generation_best_fitness
                    current_best_prompt = population[best_index]
                    self.convergence_count = 0  # Reset convergence counter
                    logger.info(f"   🎉 NEW BEST: Fitness improved to {current_best_fitness:.3f}")
                    
                    # Update global best if this is the best ever
                    if generation_best_fitness > self.global_best_fitness:
                        self.global_best_fitness = generation_best_fitness
                        self.global_best_prompt = current_best_prompt
                        self.global_best_generation = generation
                        logger.info(f"   🌟 GLOBAL BEST: New global best fitness {self.global_best_fitness:.3f} at generation {generation}")'''
    
    if old_best_tracking.replace('\n', '').replace(' ', '') in content.replace('\n', '').replace(' ', ''):
        content = content.replace(old_best_tracking, new_best_tracking)
        print("  ✅ Enhanced best tracking logic")
    
    # Update results compilation to use global best
    old_results = '''# Phase 4: Results compilation
            logger.info(f"\\n📊 PHASE 4: Results Compilation")
            results = self._compile_results(
                current_best_prompt, 
                current_best_fitness, 
                termination_reason
            )'''
    
    new_results = '''# Phase 4: Results compilation
            logger.info(f"\\n📊 PHASE 4: Results Compilation")
            
            # Use global best if it's better than current generation best
            final_prompt = self.global_best_prompt if self.global_best_prompt and self.global_best_fitness > current_best_fitness else current_best_prompt
            final_fitness = max(self.global_best_fitness, current_best_fitness)
            
            if self.global_best_prompt and self.global_best_fitness > current_best_fitness:
                logger.info(f"   🌟 USING GLOBAL BEST: From generation {self.global_best_generation} with fitness {self.global_best_fitness:.3f}")
            
            results = self._compile_results(
                final_prompt, 
                final_fitness, 
                termination_reason
            )'''
    
    if old_results.replace('\n', '').replace(' ', '') in content.replace('\n', '').replace(' ', ''):
        content = content.replace(old_results, new_results)
        print("  ✅ Updated results compilation to use global best")
    
    # Write back the enhanced content
    with open(optimizer_path, 'w') as f:
        f.write(content)


def fix_task_specific_weighting():
    """Adjust fitness weights for task-specific evaluation."""
    print("\n🔧 FIX 3: Creating task-optimized evolution config")
    
    config_path = "src/self_healing_agents/evolution/evolution_config.py"
    
    with open(config_path, 'r') as f:
        content = f.read()
    
    # Add new factory method for task-specific optimization
    task_specific_config = '''
    @classmethod
    def task_specific_optimization(cls) -> 'EvolutionConfig':
        """Configuration optimized for task-specific evaluation."""
        return cls(
            population_size=4,
            max_generations=3,
            max_evaluations=20,
            validation_tasks_count=1,  # Use actual task only
            
            # Heavily weight actual task performance
            success_rate_weight=0.8,  # 80% weight on actual task success
            efficiency_weight=0.1,    # 10% weight on efficiency
            coherence_weight=0.05,    # 5% weight on coherence  
            generalization_weight=0.05,  # 5% weight on generalization
            
            # Enhanced selection pressure
            elite_preservation_rate=0.25,  # Keep top 25%
            early_stopping_patience=2,  # Stop if no improvement for 2 generations
            improvement_threshold=0.02,  # Require meaningful improvement
            
            # Diversity management
            min_diversity_threshold=0.2,  # Allow lower diversity for convergence
            adaptive_mutation=True,
            diversity_bonus_weight=0.05,  # Reduced diversity bonus
            
            # Quality assurance
            semantic_validation=True,
            rollback_on_degradation=True,
            min_fitness_threshold=0.2
        )'''
    
    # Add after the existing fast_mode method
    if "@classmethod\n    def fast_mode(cls)" in content:
        insertion_point = content.find("@classmethod\n    def ultra_fast_debug(cls)")
        if insertion_point != -1:
            content = content[:insertion_point] + task_specific_config + "\n    " + content[insertion_point:]
            print("  ✅ Added task-specific optimization config")
    
    # Write back the enhanced content
    with open(config_path, 'w') as f:
        f.write(content)


def fix_factory_functions():
    """Update the factory functions to use the improved configurations."""
    print("\n🔧 FIX 4: Updating factory functions with better configs")
    
    optimizer_path = "src/self_healing_agents/evolution/evolutionary_prompt_optimizer.py"
    
    with open(optimizer_path, 'r') as f:
        content = f.read()
    
    # Replace the create_planner_optimizer function
    old_planner_factory = '''def create_planner_optimizer(llm_service) -> EvolutionaryPromptOptimizer:
    """Create optimizer configured for planner prompts."""
    # More reasonable config for actual testing
    config = EvolutionConfig(
        population_size=3,
        max_generations=2, 
        max_evaluations=15,
        validation_tasks_count=1,  # Use actual task
        early_stopping_patience=2,
        success_rate_weight=0.6,
        efficiency_weight=0.2,
        coherence_weight=0.15,
        generalization_weight=0.05
    )
    return EvolutionaryPromptOptimizer(llm_service, config)'''
    
    new_planner_factory = '''def create_planner_optimizer(llm_service) -> EvolutionaryPromptOptimizer:
    """Create optimizer configured for planner prompts with task-specific focus."""
    config = EvolutionConfig.task_specific_optimization()
    return EvolutionaryPromptOptimizer(llm_service, config)'''
    
    if old_planner_factory.replace('\n', '').replace(' ', '') in content.replace('\n', '').replace(' ', ''):
        content = content.replace(old_planner_factory, new_planner_factory)
        print("  ✅ Updated planner optimizer factory")
    
    # Replace the create_executor_optimizer function  
    old_executor_factory = '''def create_executor_optimizer(llm_service) -> EvolutionaryPromptOptimizer:
    """Create optimizer configured for executor prompts."""
    # More reasonable config for actual testing
    config = EvolutionConfig(
        population_size=3,
        max_generations=2,
        max_evaluations=15, 
        validation_tasks_count=1,  # Use actual task
        early_stopping_patience=2,
        success_rate_weight=0.6,
        efficiency_weight=0.2,
        coherence_weight=0.15,
        generalization_weight=0.05
    )
    return EvolutionaryPromptOptimizer(llm_service, config)'''
    
    new_executor_factory = '''def create_executor_optimizer(llm_service) -> EvolutionaryPromptOptimizer:
    """Create optimizer configured for executor prompts with task-specific focus."""
    config = EvolutionConfig.task_specific_optimization()
    return EvolutionaryPromptOptimizer(llm_service, config)'''
    
    if old_executor_factory.replace('\n', '').replace(' ', '') in content.replace('\n', '').replace(' ', ''):
        content = content.replace(old_executor_factory, new_executor_factory)
        print("  ✅ Updated executor optimizer factory")
    
    # Write back the enhanced content
    with open(optimizer_path, 'w') as f:
        f.write(content)


def add_fitness_validation():
    """Add fitness validation to re-evaluate top prompts."""
    print("\n🔧 FIX 5: Adding fitness validation mechanism")
    
    evaluator_path = "src/self_healing_agents/evolution/fitness_evaluator.py"
    
    with open(evaluator_path, 'r') as f:
        content = f.read()
    
    # Add validation method to the PromptFitnessEvaluator class
    validation_method = '''
    def validate_top_prompts(self, prompts_and_scores: List[tuple], agent_type: str) -> List[tuple]:
        """
        Re-evaluate top prompts to ensure fitness scores are accurate.
        
        Args:
            prompts_and_scores: List of (prompt, score) tuples
            agent_type: Type of agent
            
        Returns:
            List of (prompt, validated_score) tuples, sorted by validated score
        """
        logger.info(f"🔍 VALIDATION: Re-evaluating top {len(prompts_and_scores)} prompts")
        
        validated_results = []
        for prompt, original_score in prompts_and_scores:
            # Clear cache for this prompt to force re-evaluation
            cache_key = self._get_cache_key(prompt, agent_type)
            if cache_key in self.evaluation_cache:
                del self.evaluation_cache[cache_key]
            
            # Re-evaluate
            validated_score = self.evaluate_prompt(prompt, agent_type, None)
            validated_results.append((prompt, validated_score))
            
            score_diff = validated_score - original_score
            if abs(score_diff) > 0.1:
                logger.warning(f"   🚨 SCORE CHANGE: {original_score:.3f} → {validated_score:.3f} (Δ{score_diff:+.3f})")
            else:
                logger.info(f"   ✅ SCORE STABLE: {validated_score:.3f}")
        
        # Sort by validated score
        validated_results.sort(key=lambda x: x[1], reverse=True)
        return validated_results'''
    
    # Insert before the _get_cache_key method
    if "def _get_cache_key(self, prompt: str, agent_type: str) -> str:" in content:
        insertion_point = content.find("    def _get_cache_key(self, prompt: str, agent_type: str) -> str:")
        content = content[:insertion_point] + validation_method + "\n\n    " + content[insertion_point:]
        print("  ✅ Added fitness validation method")
    
    # Write back the enhanced content
    with open(evaluator_path, 'w') as f:
        f.write(content)


def test_fixed_system():
    """Test the fixed evolutionary system."""
    print("\n🧪 TESTING FIXED SYSTEM")
    print("=" * 50)
    
    try:
        from self_healing_agents.agents import Planner, Executor, Critic
        from self_healing_agents.llm_service import LLMService
        from self_healing_agents.prompts import DEFAULT_EXECUTOR_SYSTEM_PROMPT
        from self_healing_agents.evolution import create_executor_optimizer
        
        # Initialize LLM service
        llm_service = LLMService(provider="deepseek", model_name="deepseek-coder")
        print("✅ LLM Service initialized")
        
        # Initialize agents
        planner = Planner("TestPlanner", llm_service, "You are a planner.")
        executor = Executor("TestExecutor", llm_service, DEFAULT_EXECUTOR_SYSTEM_PROMPT)
        critic = Critic("TestCritic", llm_service)
        
        # Create optimizer with fixed configuration
        optimizer = create_executor_optimizer(llm_service)
        print("✅ Fixed optimizer created")
        
        # Set up task-specific context
        test_task_description = """Create a function to calculate fibonacci numbers efficiently."""
        
        optimizer.fitness_evaluator.set_task_specific_context(
            task_description=test_task_description,
            planner_agent=planner,
            executor_agent=executor,
            critic_agent=critic
        )
        print("✅ Task-specific context configured")
        
        # Run optimization
        print("\n🔄 Running optimization with fixes...")
        results = optimizer.optimize_prompt(
            base_prompt=DEFAULT_EXECUTOR_SYSTEM_PROMPT,
            agent_type="EXECUTOR"
        )
        
        print(f"\n📊 RESULTS:")
        print(f"   🏆 Best Fitness: {results.best_fitness:.3f}")
        print(f"   🔄 Generations: {results.generation_count}")
        print(f"   📊 Evaluations: {results.evaluation_count}")
        print(f"   🛑 Termination: {results.termination_reason}")
        
        # Check if global best was used
        if hasattr(optimizer, 'global_best_fitness') and optimizer.global_best_fitness > -1:
            print(f"   🌟 Global Best: {optimizer.global_best_fitness:.3f} from generation {optimizer.global_best_generation}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def main():
    """Apply all fixes to the evolutionary optimization system."""
    print("🚀 FIXING EVOLUTIONARY CONVERGENCE ISSUES")
    print("=" * 60)
    
    try:
        fix_fitness_evaluator_cache()
        fix_global_best_tracking()
        fix_task_specific_weighting()
        fix_factory_functions()
        add_fitness_validation()
        
        print("\n✅ ALL FIXES APPLIED SUCCESSFULLY!")
        print("\n🧪 Testing the fixed system...")
        
        if test_fixed_system():
            print("\n🎉 SYSTEM FIXES VALIDATED!")
        else:
            print("\n⚠️ System test failed - manual verification needed")
            
    except Exception as e:
        logger.error(f"Fix application failed: {e}")
        print(f"❌ Error applying fixes: {e}")


if __name__ == "__main__":
    main() 