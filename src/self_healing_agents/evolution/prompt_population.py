"""
Prompt population management for evolutionary optimization.
"""

import logging
import random
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from .evolution_config import EvolutionConfig, PromptGenerationConfig

logger = logging.getLogger(__name__)


@dataclass
class PromptIndividual:
    """Represents an individual prompt in the population."""
    prompt: str
    fitness: float = 0.0
    generation: int = 0
    parent_ids: List[str] = None
    diversity_score: float = 0.0
    
    def __post_init__(self):
        if self.parent_ids is None:
            self.parent_ids = []


class PromptPopulation:
    """
    Manages a population of prompts for evolutionary optimization.
    """
    
    def __init__(self, config: EvolutionConfig, llm_service):
        self.config = config
        self.llm_service = llm_service
        self.population: List[PromptIndividual] = []
        self.generation = 0
        self.fitness_history: List[List[float]] = []
        self.diversity_history: List[float] = []
        self.best_individual: Optional[PromptIndividual] = None
        
        # For diversity calculation (if sentence-transformers available)
        self._embeddings_model = None
        self._initialize_embeddings()
    
    def initialize_population(
        self, 
        base_prompt: str, 
        agent_type: str,
        generation_config: Optional[PromptGenerationConfig] = None
    ) -> List[str]:
        """
        Initialize population with diverse prompts based on the base prompt.
        
        Args:
            base_prompt: Original prompt to diversify from
            agent_type: Type of agent (PLANNER or EXECUTOR)
            generation_config: Configuration for prompt generation
            
        Returns:
            List of diverse prompts
        """
        logger.info(f"🌱 POPULATION INIT: Generating {self.config.population_size} diverse prompts for {agent_type}")
        
        if generation_config is None:
            generation_config = PromptGenerationConfig()
        
        # Always include the base prompt as first individual
        diverse_prompts = [base_prompt]
        
        # Generate diverse variations
        for i in range(1, self.config.population_size):
            try:
                strategy = self._select_generation_strategy(i, generation_config)
                new_prompt = self._generate_diverse_prompt(
                    base_prompt, agent_type, strategy, i
                )
                
                # Check diversity before adding
                if self._is_sufficiently_diverse(new_prompt, diverse_prompts, generation_config):
                    diverse_prompts.append(new_prompt)
                else:
                    # Try alternative strategy
                    alt_strategy = random.choice([
                        "instruction_focused", "example_enhanced", 
                        "constraint_focused", "role_definition"
                    ])
                    alt_prompt = self._generate_diverse_prompt(
                        base_prompt, agent_type, alt_strategy, i
                    )
                    diverse_prompts.append(alt_prompt)
                    
            except Exception as e:
                logger.warning(f"Failed to generate diverse prompt {i}: {e}")
                # Fallback: use base prompt with minor modifications
                diverse_prompts.append(self._create_fallback_prompt(base_prompt, i))
        
        # Create population individuals
        self.population = []
        for i, prompt in enumerate(diverse_prompts):
            individual = PromptIndividual(
                prompt=prompt,
                generation=0,
                parent_ids=[f"init_{i}"]
            )
            self.population.append(individual)
        
        logger.info(f"✅ POPULATION INIT: Generated {len(self.population)} individuals")
        self._log_population_diversity()
        
        # 🆕 LOG SAMPLE PROMPTS FROM POPULATION
        logger.info(f"🔍 SAMPLE GENERATED PROMPTS:")
        for i, prompt in enumerate(diverse_prompts[:2]):  # Show first 2 prompts
            logger.info(f"   Individual {i+1} ({len(prompt)} chars): {prompt}{'...' if len(prompt) > 100 else ''}")
        
        return diverse_prompts
    
    def add_offspring(self, new_prompts: List[str], parent_info: List[Tuple[str, str]] = None) -> None:
        """
        Add new offspring to the population.
        
        Args:
            new_prompts: List of new prompts to add
            parent_info: Optional list of (parent1_id, parent2_id) tuples
        """
        logger.info(f"👶 OFFSPRING: Adding {len(new_prompts)} new individuals to population")
        
        for i, prompt in enumerate(new_prompts):
            parent_ids = []
            if parent_info and i < len(parent_info):
                parent_ids = list(parent_info[i])
            
            individual = PromptIndividual(
                prompt=prompt,
                generation=self.generation + 1,
                parent_ids=parent_ids
            )
            self.population.append(individual)
    
    def select_survivors(self, fitness_scores: List[float]) -> List[str]:
        """
        Select surviving individuals based on fitness and diversity.
        
        Args:
            fitness_scores: Fitness scores for current population
            
        Returns:
            List of surviving prompts
        """
        logger.info(f"🏆 SELECTION: Selecting {self.config.population_size} survivors from {len(self.population)}")
        
        # Update fitness scores
        for i, individual in enumerate(self.population):
            if i < len(fitness_scores):
                individual.fitness = fitness_scores[i]
        
        # Calculate diversity scores
        self._update_diversity_scores()
        
        # Apply elite preservation
        num_elites = max(1, int(self.config.population_size * self.config.elite_preservation_rate))
        
        # Sort by fitness and select elites
        sorted_population = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        elites = sorted_population[:num_elites]
        
        # Select remaining individuals using fitness + diversity
        remaining_slots = self.config.population_size - num_elites
        candidates = sorted_population[num_elites:]
        
        selected_others = self._diversity_aware_selection(candidates, remaining_slots)
        
        # Combine elites and selected others
        survivors = elites + selected_others
        self.population = survivors
        
        # Update generation and tracking
        self.generation += 1
        self._update_tracking()
        
        # Update best individual
        if survivors:
            current_best = max(survivors, key=lambda x: x.fitness)
            if self.best_individual is None or current_best.fitness > self.best_individual.fitness:
                self.best_individual = current_best
        
        survivor_prompts = [ind.prompt for ind in survivors]
        logger.info(f"✅ SELECTION: Selected survivors with fitness range: {min(fitness_scores):.3f} - {max(fitness_scores):.3f}")
        
        return survivor_prompts
    
    def select_parents(self, num_parents: int = 2) -> List[str]:
        """
        Select parents for crossover using roulette wheel selection.
        
        Args:
            num_parents: Number of parents to select
            
        Returns:
            List of selected parent prompts
        """
        if len(self.population) < num_parents:
            return [ind.prompt for ind in self.population]
        
        # Use roulette wheel selection with fitness scores
        fitness_scores = [ind.fitness for ind in self.population]
        
        # Handle case where all fitness scores are 0 or negative
        min_fitness = min(fitness_scores)
        if min_fitness <= 0:
            adjusted_scores = [score - min_fitness + 0.1 for score in fitness_scores]
        else:
            adjusted_scores = fitness_scores
        
        total_fitness = sum(adjusted_scores)
        if total_fitness == 0:
            # Fallback to random selection
            selected_indices = random.sample(range(len(self.population)), num_parents)
        else:
            # Roulette wheel selection
            probabilities = [score / total_fitness for score in adjusted_scores]
            selected_indices = np.random.choice(
                len(self.population), 
                size=num_parents, 
                replace=False,
                p=probabilities
            )
        
        return [self.population[i].prompt for i in selected_indices]
    
    def get_diversity_score(self) -> float:
        """Calculate current population diversity score."""
        if len(self.population) < 2:
            return 0.0
        
        prompts = [ind.prompt for ind in self.population]
        return self._calculate_population_diversity(prompts)
    
    def get_best_prompt(self) -> Optional[str]:
        """Get the best prompt from current population or history."""
        if self.best_individual:
            return self.best_individual.prompt
        elif self.population:
            best = max(self.population, key=lambda x: x.fitness)
            return best.prompt
        return None
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get statistics for current generation."""
        if not self.population:
            return {}
        
        fitness_scores = [ind.fitness for ind in self.population]
        diversity_score = self.get_diversity_score()
        
        return {
            "generation": self.generation,
            "population_size": len(self.population),
            "fitness_mean": np.mean(fitness_scores),
            "fitness_std": np.std(fitness_scores),
            "fitness_max": np.max(fitness_scores),
            "fitness_min": np.min(fitness_scores),
            "diversity_score": diversity_score,
            "best_fitness": self.best_individual.fitness if self.best_individual else 0.0
        }
    
    def _select_generation_strategy(self, index: int, config: PromptGenerationConfig) -> str:
        """Select generation strategy based on configuration ratios."""
        strategies = [
            ("instruction_focused", config.instruction_focused_ratio),
            ("example_enhanced", config.example_enhanced_ratio),
            ("constraint_focused", config.constraint_focused_ratio),
            ("role_definition", config.role_definition_ratio),
            ("hybrid_approach", config.hybrid_approach_ratio)
        ]
        
        # Weighted random selection
        weights = [ratio for _, ratio in strategies]
        strategy_names = [name for name, _ in strategies]
        
        selected = np.random.choice(strategy_names, p=np.array(weights)/sum(weights))
        return selected
    
    def _generate_diverse_prompt(
        self, 
        base_prompt: str, 
        agent_type: str, 
        strategy: str, 
        index: int
    ) -> str:
        """Generate a diverse prompt using specified strategy."""
        
        strategy_prompts = {
            "instruction_focused": f"""
Create a variation of this {agent_type} prompt with clearer, more specific instructions:

ORIGINAL PROMPT:
{base_prompt}

GOALS:
- Make instructions more detailed and actionable
- Add specific directives and steps
- Improve clarity while maintaining core functionality
- Provide more explicit guidance

Generate the instruction-focused variation:""",

            "example_enhanced": f"""
Create a variation of this {agent_type} prompt with enhanced examples and demonstrations:

ORIGINAL PROMPT:
{base_prompt}

GOALS:
- Add helpful examples and demonstrations
- Include sample inputs/outputs where appropriate
- Make examples more comprehensive and clear
- Ensure examples align with instructions

Generate the example-enhanced variation:""",

            "constraint_focused": f"""
Create a variation of this {agent_type} prompt with explicit constraints and limitations:

ORIGINAL PROMPT:
{base_prompt}

GOALS:
- Add clear constraints and limitations
- Specify what should and shouldn't be done
- Include quality requirements
- Make boundaries explicit

Generate the constraint-focused variation:""",

            "role_definition": f"""
Create a variation of this {agent_type} prompt with enhanced role definition and identity:

ORIGINAL PROMPT:
{base_prompt}

GOALS:
- Strengthen the agent's professional identity
- Clarify capabilities and expertise
- Enhance role-specific language
- Make the agent's purpose clearer

Generate the role-enhanced variation:""",

            "hybrid_approach": f"""
Create a variation of this {agent_type} prompt that combines multiple improvement strategies:

ORIGINAL PROMPT:
{base_prompt}

GOALS:
- Improve instructions AND examples
- Enhance role definition AND constraints
- Create a well-rounded, comprehensive prompt
- Balance all aspects effectively

Generate the hybrid variation:"""
        }
        
        try:
            system_prompt = f"""You are an expert prompt engineer creating diverse variations of {agent_type} agent prompts.

Your task is to create meaningful variations that maintain core functionality while exploring different approaches to prompt design.

Key principles:
1. Preserve the agent's core role and capabilities
2. Maintain functional compatibility
3. Introduce meaningful improvements
4. Ensure semantic coherence
5. Create genuine diversity, not just cosmetic changes

Generate complete, functional prompts."""
            
            user_prompt = strategy_prompts.get(strategy, strategy_prompts["instruction_focused"])
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response = self.llm_service.invoke(messages, expect_json=False)
            return response.strip()
            
        except Exception as e:
            logger.error(f"Failed to generate {strategy} prompt: {e}")
            return self._create_fallback_prompt(base_prompt, index)
    
    def _is_sufficiently_diverse(
        self, 
        new_prompt: str, 
        existing_prompts: List[str], 
        config: PromptGenerationConfig
    ) -> bool:
        """Check if new prompt is sufficiently diverse from existing ones."""
        if not existing_prompts:
            return True
        
        # Simple token-based diversity check
        new_tokens = set(new_prompt.lower().split())
        
        for existing_prompt in existing_prompts:
            existing_tokens = set(existing_prompt.lower().split())
            
            # Calculate Jaccard similarity
            intersection = len(new_tokens.intersection(existing_tokens))
            union = len(new_tokens.union(existing_tokens))
            
            if union == 0:
                continue
                
            similarity = intersection / union
            
            if similarity > (1 - config.semantic_diversity_threshold):
                return False
        
        return True
    
    def _create_fallback_prompt(self, base_prompt: str, index: int) -> str:
        """Create a simple fallback prompt variation."""
        modifications = [
            lambda p: p.replace("You are", f"You are an expert"),
            lambda p: p + f"\n\nVariation {index}: Focus on precision and clarity.",
            lambda p: p.replace("should", "must") if "should" in p else p + "\n\nBe thorough and systematic.",
            lambda p: p + f"\n\nApproach {index}: Emphasize quality and effectiveness."
        ]
        
        mod_func = modifications[index % len(modifications)]
        return mod_func(base_prompt)
    
    def _update_diversity_scores(self) -> None:
        """Update diversity scores for all individuals in population."""
        prompts = [ind.prompt for ind in self.population]
        
        for i, individual in enumerate(self.population):
            # Calculate average distance to all other prompts
            diversity_sum = 0.0
            for j, other_prompt in enumerate(prompts):
                if i != j:
                    diversity_sum += self._calculate_prompt_distance(individual.prompt, other_prompt)
            
            individual.diversity_score = diversity_sum / max(1, len(prompts) - 1)
    
    def _calculate_prompt_distance(self, prompt1: str, prompt2: str) -> float:
        """Calculate semantic distance between two prompts."""
        if self._embeddings_model:
            try:
                # Use sentence transformers if available
                emb1 = self._embeddings_model.encode([prompt1])
                emb2 = self._embeddings_model.encode([prompt2])
                
                # Cosine distance
                from sklearn.metrics.pairwise import cosine_distances
                distance = cosine_distances(emb1, emb2)[0][0]
                return distance
            except Exception as e:
                logger.debug(f"Embedding-based distance failed: {e}")
        
        # Fallback: token-based Jaccard distance
        tokens1 = set(prompt1.lower().split())
        tokens2 = set(prompt2.lower().split())
        
        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))
        
        if union == 0:
            return 0.0
        
        jaccard_similarity = intersection / union
        return 1.0 - jaccard_similarity
    
    def _calculate_population_diversity(self, prompts: List[str]) -> float:
        """Calculate overall population diversity score."""
        if len(prompts) < 2:
            return 0.0
        
        total_distance = 0.0
        comparisons = 0
        
        for i in range(len(prompts)):
            for j in range(i + 1, len(prompts)):
                total_distance += self._calculate_prompt_distance(prompts[i], prompts[j])
                comparisons += 1
        
        return total_distance / max(1, comparisons)
    
    def _diversity_aware_selection(
        self, 
        candidates: List[PromptIndividual], 
        num_to_select: int
    ) -> List[PromptIndividual]:
        """Select individuals considering both fitness and diversity."""
        if len(candidates) <= num_to_select:
            return candidates
        
        # Calculate combined scores (fitness + diversity bonus)
        for candidate in candidates:
            diversity_bonus = candidate.diversity_score * self.config.diversity_bonus_weight
            candidate.combined_score = candidate.fitness + diversity_bonus
        
        # Sort by combined score and select top N
        candidates.sort(key=lambda x: x.combined_score, reverse=True)
        return candidates[:num_to_select]
    
    def _update_tracking(self) -> None:
        """Update fitness and diversity history."""
        if self.population:
            fitness_scores = [ind.fitness for ind in self.population]
            self.fitness_history.append(fitness_scores)
            
            diversity_score = self.get_diversity_score()
            self.diversity_history.append(diversity_score)
    
    def _log_population_diversity(self) -> None:
        """Log current population diversity information."""
        if len(self.population) >= 2:
            diversity = self.get_diversity_score()
            logger.info(f"📊 DIVERSITY: Population diversity score: {diversity:.3f}")
        else:
            logger.info("📊 DIVERSITY: Population too small for diversity calculation")
    
    def _initialize_embeddings(self) -> None:
        """Initialize sentence embeddings model if available."""
        try:
            from sentence_transformers import SentenceTransformer
            self._embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✅ EMBEDDINGS: Initialized sentence transformers for diversity calculation")
        except ImportError:
            logger.info("ℹ️ EMBEDDINGS: sentence-transformers not available, using token-based diversity")
            self._embeddings_model = None
        except Exception as e:
            logger.warning(f"⚠️ EMBEDDINGS: Failed to initialize sentence transformers: {e}")
            self._embeddings_model = None 