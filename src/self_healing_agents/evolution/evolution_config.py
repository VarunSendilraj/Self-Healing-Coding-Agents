"""
Configuration classes for evolutionary prompt optimization.
"""

from dataclasses import dataclass
from typing import Dict, Any, List


@dataclass
class EvolutionConfig:
    """Configuration for evolutionary prompt optimization."""
    
    # Population parameters
    population_size: int = 5
    max_generations: int = 8
    elite_preservation_rate: float = 0.2  # Keep top 20% unconditionally
    
    # Evolution operators
    crossover_rate: float = 0.8
    mutation_rate: float = 0.3
    diversity_bonus_weight: float = 0.1
    
    # Fitness function weights
    success_rate_weight: float = 0.5
    efficiency_weight: float = 0.2
    coherence_weight: float = 0.2
    generalization_weight: float = 0.1
    
    # Convergence criteria
    max_evaluations: int = 50  # Hard limit on LLM calls
    early_stopping_patience: int = 3  # Stop if no improvement for N generations
    improvement_threshold: float = 0.01  # Minimum improvement to continue
    
    # Diversity management
    min_diversity_threshold: float = 0.3  # Minimum population diversity
    adaptive_mutation: bool = True  # Increase mutation when diversity drops
    diversity_penalty_weight: float = 0.05  # Reduce fitness for similar prompts
    
    # Validation parameters
    validation_tasks_count: int = 3  # Number of tasks for fitness evaluation
    parallel_evaluation: bool = False  # Concurrent fitness assessments
    timeout_seconds: int = 60  # Timeout per prompt evaluation
    
    # Quality assurance
    semantic_validation: bool = True  # Check role consistency each generation
    rollback_on_degradation: bool = True  # Revert if quality drops
    min_fitness_threshold: float = 0.1  # Minimum acceptable fitness
    
    @classmethod
    def for_planner(cls) -> 'EvolutionConfig':
        """Optimized configuration for planner prompt evolution."""
        return cls(
            population_size=5,
            max_generations=6,
            mutation_rate=0.25,
            success_rate_weight=0.6,
            coherence_weight=0.3
        )
    
    @classmethod
    def for_executor(cls) -> 'EvolutionConfig':
        """Optimized configuration for executor prompt evolution."""
        return cls(
            population_size=5,
            max_generations=8,
            mutation_rate=0.35,
            success_rate_weight=0.5,
            efficiency_weight=0.3
        )
    
    @classmethod
    def fast_mode(cls) -> 'EvolutionConfig':
        """Quick evolution for time-constrained scenarios."""
        return cls(
            population_size=3,
            max_generations=4,
            max_evaluations=20,
            validation_tasks_count=2
        )
    
    
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
        )
    @classmethod
    def ultra_fast_debug(cls) -> 'EvolutionConfig':
        """Ultra-fast mode for debugging and demonstrations."""
        return cls(
            population_size=2,  # Minimal population
            max_generations=2,  # Just 2 generations
            max_evaluations=10,  # Very few LLM calls
            validation_tasks_count=1,  # Single validation task
            early_stopping_patience=1,  # Stop immediately if no improvement
            elite_preservation_rate=0.5,  # Keep half (so 1 out of 2)
            success_rate_weight=0.6,
            efficiency_weight=0.2,
            coherence_weight=0.15,
            generalization_weight=0.05  # Reduced to sum to 1.0
        )
    
    def validate(self) -> List[str]:
        """Validate configuration parameters and return any issues."""
        issues = []
        
        if self.population_size < 3:
            issues.append("Population size must be at least 3")
        
        if self.max_generations < 1:
            issues.append("Max generations must be at least 1")
        
        if not 0 <= self.elite_preservation_rate <= 1:
            issues.append("Elite preservation rate must be between 0 and 1")
        
        if not 0 <= self.crossover_rate <= 1:
            issues.append("Crossover rate must be between 0 and 1")
        
        if not 0 <= self.mutation_rate <= 1:
            issues.append("Mutation rate must be between 0 and 1")
        
        # Check fitness weights sum to reasonable range
        total_weight = (self.success_rate_weight + self.efficiency_weight + 
                       self.coherence_weight + self.generalization_weight)
        if not 0.8 <= total_weight <= 1.2:
            issues.append(f"Fitness weights sum to {total_weight:.2f}, should be close to 1.0")
        
        if self.validation_tasks_count < 1:
            issues.append("Validation tasks count must be at least 1")
        
        return issues


@dataclass 
class PromptGenerationConfig:
    """Configuration for initial prompt population generation."""
    
    # Diversification strategies
    instruction_focused_ratio: float = 0.2  # Clearer directives
    example_enhanced_ratio: float = 0.2     # More demonstrations  
    constraint_focused_ratio: float = 0.2   # Explicit limitations
    role_definition_ratio: float = 0.2      # Enhanced agent identity
    hybrid_approach_ratio: float = 0.2      # Mixed strategies
    
    # Generation parameters
    max_prompt_length: int = 1500  # Maximum tokens per prompt
    min_prompt_length: int = 200   # Minimum tokens per prompt
    semantic_diversity_threshold: float = 0.4  # Minimum semantic distance
    max_generation_attempts: int = 10  # Retry limit for diverse generation
    
    # Quality filters
    require_role_definition: bool = True
    require_output_format: bool = True
    require_clear_instructions: bool = True
    
    def validate(self) -> List[str]:
        """Validate generation configuration."""
        issues = []
        
        # Check ratios sum to 1.0
        total_ratio = (self.instruction_focused_ratio + self.example_enhanced_ratio +
                      self.constraint_focused_ratio + self.role_definition_ratio +
                      self.hybrid_approach_ratio)
        if not 0.95 <= total_ratio <= 1.05:
            issues.append(f"Generation ratios sum to {total_ratio:.2f}, should be 1.0")
        
        if self.max_prompt_length <= self.min_prompt_length:
            issues.append("Max prompt length must be greater than min prompt length")
        
        if not 0 <= self.semantic_diversity_threshold <= 1:
            issues.append("Semantic diversity threshold must be between 0 and 1")
        
        return issues 