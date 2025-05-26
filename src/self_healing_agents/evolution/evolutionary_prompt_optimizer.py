"""
Main evolutionary prompt optimizer orchestrating the complete evolutionary algorithm.
"""

import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from .evolution_config import EvolutionConfig, PromptGenerationConfig
from .prompt_population import PromptPopulation
from .fitness_evaluator import PromptFitnessEvaluator
from .evolution_operators import EvolutionOperators

logger = logging.getLogger(__name__)


@dataclass
class EvolutionResults:
    """Results from evolutionary optimization."""
    best_prompt: str
    best_fitness: float
    generation_count: int
    evaluation_count: int
    convergence_history: List[float]
    diversity_history: List[float]
    execution_time: float
    termination_reason: str
    population_history: List[List[str]] = None
    
    def __post_init__(self):
        if self.population_history is None:
            self.population_history = []


class EvolutionaryPromptOptimizer:
    """
    Main orchestrator for evolutionary prompt optimization.
    
    This class implements the complete evolutionary algorithm pipeline:
    1. Population initialization with diverse prompts
    2. Fitness evaluation across validation tasks
    3. Selection using roulette wheel + elite preservation
    4. Evolution through LLM-guided crossover and mutation
    5. Population update and convergence monitoring
    """
    
    def __init__(
        self, 
        llm_service,
        config: Optional[EvolutionConfig] = None
    ):
        """Initialize evolutionary prompt optimizer."""
        self.llm_service = llm_service
        self.config = config or EvolutionConfig()
        
        # Validate configuration
        config_issues = self.config.validate()
        if config_issues:
            logger.warning(f"⚠️ CONFIG ISSUES: {config_issues}")
        
        # Initialize components
        self.fitness_evaluator = PromptFitnessEvaluator(self.config, llm_service)
        self.population_manager = PromptPopulation(self.config, llm_service)
        self.evolution_operators = EvolutionOperators(llm_service)
        
        # Evolution state
        self.current_generation = 0
        self.total_evaluations = 0
        self.best_fitness_history = []
        self.convergence_count = 0
        self.start_time = None
        
        # Global best tracking across all generations
        self.global_best_prompt = None
        self.global_best_fitness = -1.0
        self.global_best_generation = 0
        
    def optimize_prompt(
        self, 
        base_prompt: str, 
        agent_type: str,
        failure_context: Optional[Dict[str, Any]] = None,
        generation_config: Optional[PromptGenerationConfig] = None
    ) -> EvolutionResults:
        """
        Optimize a prompt using evolutionary algorithm.
        
        Args:
            base_prompt: Original prompt to improve
            agent_type: Type of agent (PLANNER or EXECUTOR)
            failure_context: Context about the original failure
            generation_config: Configuration for population generation
            
        Returns:
            EvolutionResults with optimized prompt and evolution statistics
        """
        logger.info(f"🧬 EVOLUTION START: Optimizing {agent_type} prompt")
        logger.info(f"   📊 Config: Pop={self.config.population_size}, Gen={self.config.max_generations}")
        logger.info(f"   🎯 Budget: Max {self.config.max_evaluations} evaluations")
        
        self.start_time = time.time()
        
        try:
            # Phase 1: Initialize population
            logger.info(f"\n🌱 PHASE 1: Population Initialization")
            population = self._initialize_population(base_prompt, agent_type, generation_config)
            
            # Phase 2: Initial fitness evaluation
            logger.info(f"\n🎯 PHASE 2: Initial Fitness Evaluation")
            fitness_scores = self._evaluate_population(population, agent_type, failure_context)
            
            # Track initial best
            best_index = fitness_scores.index(max(fitness_scores))
            current_best_fitness = fitness_scores[best_index]
            current_best_prompt = population[best_index]
            self.best_fitness_history.append(current_best_fitness)
            
            logger.info(f"   📈 Initial best fitness: {current_best_fitness:.3f}")
            
            # Phase 3: Evolution loop
            logger.info(f"\n🔄 PHASE 3: Evolution Loop")
            termination_reason = "max_generations"
            
            for generation in range(1, self.config.max_generations + 1):
                self.current_generation = generation
                logger.info(f"\n--- Generation {generation}/{self.config.max_generations} ---")
                
                # Check termination conditions
                if self._should_terminate():
                    termination_reason = "early_stopping"
                    break
                
                # Survivor selection
                surviving_prompts = self.population_manager.select_survivors(fitness_scores)
                
                # Generate offspring
                offspring_prompts = self._generate_offspring(surviving_prompts, agent_type, failure_context)
                
                # Evaluate offspring
                offspring_fitness = self._evaluate_population(offspring_prompts, agent_type, failure_context)
                
                # Update population with offspring
                all_prompts = surviving_prompts + offspring_prompts
                all_fitness = fitness_scores[:len(surviving_prompts)] + offspring_fitness
                
                # Select survivors for next generation
                population = self.population_manager.select_survivors(all_fitness)
                fitness_scores = all_fitness[:len(population)]
                
                # Track best individual
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
                        logger.info(f"   🌟 GLOBAL BEST: New global best fitness {self.global_best_fitness:.3f} at generation {generation}")
                    
                    # 🆕 LOG THE EVOLVED PROMPT PREVIEW
                    logger.info(f"   📝 BEST PROMPT PREVIEW ({len(current_best_prompt)} chars):")
                    logger.info(f"      {current_best_prompt}{'...' if len(current_best_prompt) > 150 else ''}")
                else:
                    self.convergence_count += 1
                
                self.best_fitness_history.append(current_best_fitness)
                
                # Log generation statistics
                self._log_generation_stats(generation, fitness_scores, population)
                
                # Check evaluation budget
                if self.total_evaluations >= self.config.max_evaluations:
                    termination_reason = "evaluation_budget"
                    break
            
            # Phase 4: Results compilation
            logger.info(f"\n📊 PHASE 4: Results Compilation")
            
            # Use global best if it's better than current generation best
            final_prompt = self.global_best_prompt if self.global_best_prompt and self.global_best_fitness > current_best_fitness else current_best_prompt
            final_fitness = max(self.global_best_fitness, current_best_fitness)
            
            if self.global_best_prompt and self.global_best_fitness > current_best_fitness:
                logger.info(f"   🌟 USING GLOBAL BEST: From generation {self.global_best_generation} with fitness {self.global_best_fitness:.3f}")
            
            results = self._compile_results(
                final_prompt, 
                final_fitness, 
                termination_reason
            )
            
            self._log_final_results(results)
            return results
            
        except Exception as e:
            logger.error(f"❌ EVOLUTION ERROR: {e}")
            # Return best result found so far
            return self._compile_results(
                base_prompt,  # Fallback to original
                0.0,
                f"error: {str(e)}"
            )
    
    def _initialize_population(
        self, 
        base_prompt: str, 
        agent_type: str,
        generation_config: Optional[PromptGenerationConfig]
    ) -> List[str]:
        """Initialize the population with diverse prompts."""
        return self.population_manager.initialize_population(
            base_prompt, agent_type, generation_config
        )
    
    def _evaluate_population(
        self, 
        population: List[str], 
        agent_type: str,
        failure_context: Optional[Dict[str, Any]]
    ) -> List[float]:
        """Evaluate fitness for entire population."""
        fitness_scores = self.fitness_evaluator.batch_evaluate(
            population, agent_type, failure_context
        )
        self.total_evaluations += len(population)
        return fitness_scores
    
    def _generate_offspring(self, parents: List[str], agent_type: str, failure_context: Optional[Dict[str, Any]] = None) -> List[str]:
        """Generate offspring through crossover and mutation."""
        logger.info(f"   👶 OFFSPRING: Generating new individuals")
        
        offspring = []
        offspring_count = max(1, self.config.population_size // 2)  # Generate 50% new offspring
        
        for i in range(offspring_count):
            try:
                # Select parents for crossover
                selected_parents = self.population_manager.select_parents(2)
                
                if len(selected_parents) >= 2:
                    # Crossover with failure context
                    if len(selected_parents) >= 2 and self._should_apply_crossover():
                        child = self.evolution_operators.crossover(
                            selected_parents[0], selected_parents[1], agent_type, failure_context
                        )
                    else:
                        # Use best parent as base
                        child = selected_parents[0]
                    
                    # Mutation with failure context
                    if self._should_apply_mutation():
                        mutation_rate = self._get_adaptive_mutation_rate()
                        child = self.evolution_operators.mutate(child, agent_type, mutation_rate, failure_context)
                    
                    offspring.append(child)
                else:
                    # Fallback: mutate existing individual with failure context
                    parent = selected_parents[0] if selected_parents else parents[0]
                    mutated = self.evolution_operators.mutate(parent, agent_type, self.config.mutation_rate, failure_context)
                    offspring.append(mutated)
                    
            except Exception as e:
                logger.warning(f"Offspring generation failed for individual {i}: {e}")
                # Fallback: use random parent with minor mutation
                if parents:
                    import random
                    fallback = self.evolution_operators.mutate(
                        random.choice(parents), agent_type, 0.1, failure_context
                    )
                    offspring.append(fallback)
        
        logger.info(f"   ✅ Generated {len(offspring)} offspring")
        return offspring
    
    def _should_apply_crossover(self) -> bool:
        """Determine if crossover should be applied."""
        import random
        return random.random() < self.config.crossover_rate
    
    def _should_apply_mutation(self) -> bool:
        """Determine if mutation should be applied."""
        import random
        return random.random() < self.config.mutation_rate
    
    def _get_adaptive_mutation_rate(self) -> float:
        """Get adaptive mutation rate based on population diversity."""
        if not self.config.adaptive_mutation:
            return self.config.mutation_rate
        
        # Increase mutation rate if diversity is low
        diversity = self.population_manager.get_diversity_score()
        if diversity < self.config.min_diversity_threshold:
            return min(0.8, self.config.mutation_rate * 1.5)
        
        return self.config.mutation_rate
    
    def _should_terminate(self) -> bool:
        """Check if evolution should terminate early."""
        # Early stopping based on no improvement
        if self.convergence_count >= self.config.early_stopping_patience:
            logger.info(f"   🛑 EARLY STOPPING: No improvement for {self.convergence_count} generations")
            return True
        
        # Check if we've found a very good solution
        if self.best_fitness_history and self.best_fitness_history[-1] >= 0.95:
            logger.info(f"   🎯 TARGET REACHED: Fitness {self.best_fitness_history[-1]:.3f} >= 0.95")
            return True
        
        return False
    
    def _log_generation_stats(self, generation: int, fitness_scores: List[float], population: List[str]) -> None:
        """Log statistics for current generation."""
        stats = self.population_manager.get_generation_stats()
        
        logger.info(f"   📊 Gen {generation} Stats:")
        logger.info(f"      🏆 Best: {stats.get('fitness_max', 0):.3f}")
        logger.info(f"      📈 Mean: {stats.get('fitness_mean', 0):.3f} ± {stats.get('fitness_std', 0):.3f}")
        logger.info(f"      🎲 Diversity: {stats.get('diversity_score', 0):.3f}")
        logger.info(f"      🔢 Evaluations: {self.total_evaluations}")
        
        # Check for diversity issues
        if stats.get('diversity_score', 0) < self.config.min_diversity_threshold:
            logger.warning(f"      ⚠️ LOW DIVERSITY: {stats.get('diversity_score', 0):.3f} < {self.config.min_diversity_threshold}")
    
    def _compile_results(
        self, 
        best_prompt: str, 
        best_fitness: float, 
        termination_reason: str
    ) -> EvolutionResults:
        """Compile final evolution results."""
        execution_time = time.time() - self.start_time if self.start_time else 0.0
        
        return EvolutionResults(
            best_prompt=best_prompt,
            best_fitness=best_fitness,
            generation_count=self.current_generation,
            evaluation_count=self.total_evaluations,
            convergence_history=self.best_fitness_history.copy(),
            diversity_history=self.population_manager.diversity_history.copy(),
            execution_time=execution_time,
            termination_reason=termination_reason
        )
    
    def _log_final_results(self, results: EvolutionResults) -> None:
        """Log comprehensive final results."""
        logger.info(f"\n" + "="*60)
        logger.info(f"🧬 EVOLUTIONARY OPTIMIZATION COMPLETE")
        logger.info(f"="*60)
        logger.info(f"🏆 Best Fitness: {results.best_fitness:.3f}")
        logger.info(f"🔢 Generations: {results.generation_count}")
        logger.info(f"📊 Evaluations: {results.evaluation_count}")
        logger.info(f"⏱️  Execution Time: {results.execution_time:.1f}s")
        logger.info(f"🛑 Termination: {results.termination_reason}")
        
        # Fitness improvement analysis
        if len(results.convergence_history) >= 2:
            initial_fitness = results.convergence_history[0]
            improvement = results.best_fitness - initial_fitness
            improvement_pct = (improvement / max(initial_fitness, 0.001)) * 100
            logger.info(f"📈 Improvement: +{improvement:.3f} ({improvement_pct:+.1f}%)")
        
        # Performance summary
        if results.evaluation_count > 0:
            evaluations_per_sec = results.evaluation_count / max(results.execution_time, 0.1)
            logger.info(f"⚡ Efficiency: {evaluations_per_sec:.1f} evaluations/sec")
        
        # Quality assessment
        if results.best_fitness >= 0.8:
            logger.info(f"✅ EXCELLENT: High-quality prompt evolved")
        elif results.best_fitness >= 0.6:
            logger.info(f"✅ GOOD: Satisfactory prompt improvement")
        elif results.best_fitness >= 0.4:
            logger.info(f"⚠️ MODERATE: Some improvement achieved")
        else:
            logger.info(f"❌ LIMITED: Minimal improvement")
        
        logger.info(f"="*60)


# Convenience factory functions
def create_planner_optimizer(llm_service) -> EvolutionaryPromptOptimizer:
    """Create optimizer configured for planner prompts with task-specific focus."""
    config = EvolutionConfig.task_specific_optimization()
    return EvolutionaryPromptOptimizer(llm_service, config)


def create_executor_optimizer(llm_service) -> EvolutionaryPromptOptimizer:
    """Create optimizer configured for executor prompts with task-specific focus."""
    config = EvolutionConfig.task_specific_optimization()
    return EvolutionaryPromptOptimizer(llm_service, config)


def create_fast_optimizer(llm_service) -> EvolutionaryPromptOptimizer:
    """Create optimizer configured for fast optimization."""
    config = EvolutionConfig.fast_mode()
    return EvolutionaryPromptOptimizer(llm_service, config) 