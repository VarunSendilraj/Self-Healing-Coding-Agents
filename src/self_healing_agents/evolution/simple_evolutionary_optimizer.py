"""
Simple evolutionary prompt optimizer that actually works.
Uses direct fitness evaluation against original test cases.
"""

import logging
import time
import random
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from .simple_fitness_evaluator import SimpleFitnessEvaluator
from .evolution_operators import EvolutionOperators

logger = logging.getLogger(__name__)


@dataclass
class SimpleEvolutionResults:
    """Results from simple evolutionary optimization."""
    best_prompt: str
    best_fitness: float
    generation_count: int
    evaluation_count: int
    fitness_history: List[float]
    execution_time: float
    termination_reason: str


class SimpleEvolutionaryOptimizer:
    """
    Simple evolutionary optimizer that focuses on what works:
    1. Direct fitness evaluation against original test cases
    2. Simple population management
    3. Error-targeted mutations
    4. No complex multi-objective scoring
    """
    
    def __init__(
        self, 
        llm_service,
        planner_agent,
        executor_agent,
        critic_agent,
        task_description: str,
        original_test_cases: List[Dict[str, Any]]
    ):
        """
        Initialize the simple evolutionary optimizer.
        
        Args:
            llm_service: LLM service for evolution operators
            planner_agent: The planner agent
            executor_agent: The executor agent
            critic_agent: The critic agent
            task_description: Original task description
            original_test_cases: Test cases from the critic
        """
        self.llm_service = llm_service
        self.task_description = task_description
        
        # Initialize components
        self.fitness_evaluator = SimpleFitnessEvaluator(
            planner_agent, executor_agent, critic_agent, 
            task_description, original_test_cases
        )
        self.evolution_operators = EvolutionOperators(llm_service)
        
        # Evolution parameters
        self.population_size = 4
        self.max_generations = 3
        self.max_evaluations = 20
        self.improvement_threshold = 0.05
        
        # State tracking
        self.evaluation_count = 0
        self.generation_count = 0
        
        logger.info(f"🧬 SIMPLE EVOLUTION: Initialized")
        logger.info(f"   📊 Population: {self.population_size}, Generations: {self.max_generations}")
        logger.info(f"   🎯 Test Cases: {len(original_test_cases)}")
    
    def optimize_prompt(
        self, 
        base_prompt: str, 
        agent_type: str,
        failure_context: Optional[Dict[str, Any]] = None
    ) -> SimpleEvolutionResults:
        """
        Optimize a prompt using simple evolutionary algorithm.
        
        Args:
            base_prompt: Original prompt to improve
            agent_type: Either "PLANNER" or "EXECUTOR"
            failure_context: Context about the original failure
            
        Returns:
            SimpleEvolutionResults with optimized prompt and stats
        """
        logger.info(f"🧬 STARTING SIMPLE EVOLUTION for {agent_type}")
        start_time = time.time()
        
        try:
            # Initialize population with base prompt
            population = [base_prompt]
            
            # Evaluate base prompt
            base_fitness = self.fitness_evaluator.evaluate_prompt(base_prompt, agent_type)
            fitness_scores = [base_fitness]
            self.evaluation_count += 1
            
            logger.info(f"📊 BASELINE: {base_fitness:.3f}")
            
            best_prompt = base_prompt
            best_fitness = base_fitness
            fitness_history = [base_fitness]
            
            # Evolution loop
            for generation in range(1, self.max_generations + 1):
                self.generation_count = generation
                logger.info(f"\n🔄 GENERATION {generation}/{self.max_generations}")
                
                # Generate offspring through mutation and crossover
                offspring = self._generate_offspring(
                    population, fitness_scores, agent_type, failure_context
                )
                
                if not offspring:
                    logger.warning("   ⚠️ No offspring generated, stopping evolution")
                    break
                
                # Evaluate offspring
                offspring_fitness = []
                for prompt in offspring:
                    if self.evaluation_count >= self.max_evaluations:
                        logger.warning(f"   ⚠️ Evaluation budget exhausted")
                        break
                    
                    fitness = self.fitness_evaluator.evaluate_prompt(prompt, agent_type)
                    offspring_fitness.append(fitness)
                    self.evaluation_count += 1
                
                if not offspring_fitness:
                    break
                
                # Combine population and offspring
                all_prompts = population + offspring[:len(offspring_fitness)]
                all_fitness = fitness_scores + offspring_fitness
                
                # Select best individuals for next generation
                population, fitness_scores = self._select_survivors(
                    all_prompts, all_fitness
                )
                
                # Update best
                generation_best_fitness = max(fitness_scores)
                if generation_best_fitness > best_fitness:
                    best_index = fitness_scores.index(generation_best_fitness)
                    best_prompt = population[best_index]
                    best_fitness = generation_best_fitness
                    
                    improvement = best_fitness - fitness_history[-1]
                    logger.info(f"   🎉 NEW BEST: {best_fitness:.3f} (+{improvement:.3f})")
                else:
                    logger.info(f"   📊 Best: {generation_best_fitness:.3f} (no improvement)")
                
                fitness_history.append(best_fitness)
                
                # Check termination conditions
                if best_fitness >= 0.95:
                    logger.info(f"   🎯 Near perfect fitness achieved!")
                    break
                
                if self.evaluation_count >= self.max_evaluations:
                    logger.info(f"   🛑 Evaluation budget exhausted")
                    break
            
            execution_time = time.time() - start_time
            
            # Determine termination reason
            if best_fitness >= 0.95:
                termination_reason = "target_achieved"
            elif self.evaluation_count >= self.max_evaluations:
                termination_reason = "evaluation_budget"
            elif generation >= self.max_generations:
                termination_reason = "max_generations"
            else:
                termination_reason = "early_stopping"
            
            results = SimpleEvolutionResults(
                best_prompt=best_prompt,
                best_fitness=best_fitness,
                generation_count=self.generation_count,
                evaluation_count=self.evaluation_count,
                fitness_history=fitness_history,
                execution_time=execution_time,
                termination_reason=termination_reason
            )
            
            self._log_final_results(results)
            return results
            
        except Exception as e:
            logger.error(f"❌ EVOLUTION ERROR: {e}")
            # Return best result found so far
            execution_time = time.time() - start_time
            return SimpleEvolutionResults(
                best_prompt=base_prompt,
                best_fitness=base_fitness if 'base_fitness' in locals() else 0.0,
                generation_count=self.generation_count,
                evaluation_count=self.evaluation_count,
                fitness_history=fitness_history if 'fitness_history' in locals() else [0.0],
                execution_time=execution_time,
                termination_reason=f"error: {str(e)}"
            )
    
    def _generate_offspring(
        self, 
        population: List[str], 
        fitness_scores: List[float],
        agent_type: str,
        failure_context: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Generate offspring through mutation and crossover."""
        offspring = []
        
        # Sort population by fitness
        sorted_pairs = sorted(zip(population, fitness_scores), key=lambda x: x[1], reverse=True)
        sorted_population = [prompt for prompt, _ in sorted_pairs]
        
        # Generate offspring
        offspring_count = max(1, self.population_size - 1)  # Keep some diversity
        
        for i in range(offspring_count):
            if random.random() < 0.7 and len(sorted_population) >= 2:
                # Crossover between two good parents
                parent1 = sorted_population[0]  # Best
                parent2 = random.choice(sorted_population[:min(3, len(sorted_population))])  # Top 3
                
                try:
                    child = self.evolution_operators.crossover(
                        parent1, parent2, agent_type, failure_context
                    )
                    offspring.append(child)
                    logger.debug(f"   🧬 Generated crossover offspring")
                except Exception as e:
                    logger.warning(f"   ⚠️ Crossover failed: {e}")
            else:
                # Mutation of best individual
                parent = sorted_population[0]
                try:
                    child = self.evolution_operators.mutate(
                        parent, agent_type, 0.4, failure_context
                    )
                    offspring.append(child)
                    logger.debug(f"   🔀 Generated mutation offspring")
                except Exception as e:
                    logger.warning(f"   ⚠️ Mutation failed: {e}")
        
        logger.info(f"   👶 Generated {len(offspring)} offspring")
        return offspring
    
    def _select_survivors(
        self, 
        prompts: List[str], 
        fitness_scores: List[float]
    ) -> Tuple[List[str], List[float]]:
        """Select the best individuals for the next generation."""
        # Sort by fitness (descending)
        sorted_pairs = sorted(zip(prompts, fitness_scores), key=lambda x: x[1], reverse=True)
        
        # Keep top individuals
        survivors = sorted_pairs[:self.population_size]
        
        survivor_prompts = [prompt for prompt, _ in survivors]
        survivor_fitness = [fitness for _, fitness in survivors]
        
        logger.debug(f"   👥 Selected {len(survivor_prompts)} survivors")
        return survivor_prompts, survivor_fitness
    
    def _log_final_results(self, results: SimpleEvolutionResults) -> None:
        """Log final optimization results."""
        logger.info(f"\n" + "="*50)
        logger.info(f"🧬 SIMPLE EVOLUTION COMPLETE")
        logger.info(f"="*50)
        logger.info(f"🏆 Best Fitness: {results.best_fitness:.3f}")
        logger.info(f"🔢 Generations: {results.generation_count}")
        logger.info(f"📊 Evaluations: {results.evaluation_count}")
        logger.info(f"⏱️  Time: {results.execution_time:.1f}s")
        logger.info(f"🛑 Reason: {results.termination_reason}")
        
        # Improvement analysis
        if len(results.fitness_history) >= 2:
            initial = results.fitness_history[0]
            final = results.best_fitness
            improvement = final - initial
            improvement_pct = (improvement / max(initial, 0.001)) * 100
            logger.info(f"📈 Improvement: +{improvement:.3f} ({improvement_pct:+.1f}%)")
        
        # Efficiency
        if results.evaluation_count > 0:
            evals_per_sec = results.evaluation_count / max(results.execution_time, 0.1)
            logger.info(f"⚡ Efficiency: {evals_per_sec:.1f} evals/sec")
        
        logger.info(f"="*50) 