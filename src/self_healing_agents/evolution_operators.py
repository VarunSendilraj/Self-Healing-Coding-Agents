"""
Defines classes for performing targeted, operator-based prompt evolution.
Task 3.2 from PRD_PHASE2.md.
"""
import random
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Literal

from self_healing_agents.llm_service import LLMService, LLMServiceError
from self_healing_agents.error_types import AgentType, ErrorType # Assuming ErrorType might be used by selector
from self_healing_agents.prompts import (
    ENHANCE_DECOMPOSITION_OPERATOR_PROMPT,
    STRENGTHEN_CONSTRAINTS_OPERATOR_PROMPT,
    ADD_EXAMPLES_OPERATOR_PROMPT,
    ERROR_SPECIFIC_ENHANCEMENT_OPERATOR_PROMPT,
    ADD_DEFENSIVE_PROGRAMMING_OPERATOR_PROMPT,
    ALGORITHM_SUGGESTION_OPERATOR_PROMPT
)
import logging

logger = logging.getLogger(__name__)

# Define operator names as literals for type hinting and consistency
PlannerOperator = Literal[
    "enhance_decomposition", 
    "strengthen_constraints", 
    "add_examples"
]
ExecutorOperator = Literal[
    "error_specific_enhancement", 
    "add_defensive_programming", 
    "algorithm_suggestion"
]
AllOperators = Literal[PlannerOperator, ExecutorOperator]


class BasePromptEvolver(ABC):
    """Abstract base class for agent-specific prompt evolvers."""
    def __init__(self, llm_service: LLMService, agent_type: AgentType):
        self.llm_service = llm_service
        self.agent_type = agent_type

    @abstractmethod
    def get_available_operators(self) -> List[AllOperators]:
        """Returns a list of operator names available for this evolver."""
        pass

    def _apply_operator_prompt(self, operator_prompt_template: str, original_prompt: str, feedback_str: str) -> Optional[str]:
        """Helper to apply a specific operator prompt using the LLM."""
        system_prompt_content = operator_prompt_template.format(original_prompt=original_prompt, feedback=feedback_str)
        messages = [{"role": "system", "content": system_prompt_content}]
        
        evolved_prompt_text = None
        try:
            logger.info(f"Applying operator for {self.agent_type.name} using template: {operator_prompt_template[:50]}... Original prompt: '{original_prompt[:70]}...' Feedback: '{feedback_str[:70]}...'")
            raw_evolved_text = self.llm_service.invoke(messages, expect_json=False)
            if isinstance(raw_evolved_text, str) and raw_evolved_text.strip():
                evolved_prompt_text = raw_evolved_text.strip()
                logger.info(f"  Successfully evolved prompt via operator: '{evolved_prompt_text[:70]}...'")
            else:
                logger.warning(f"  LLM did not return a valid string prompt for operator. Response: {raw_evolved_text}")
        except LLMServiceError as e:
            logger.error(f"  LLMServiceError during operator application: {e}")
        except Exception as e:
            logger.error(f"  Unexpected error during operator application: {e}", exc_info=True)
        return evolved_prompt_text

    def evolve(self, operator_name: AllOperators, original_prompt: str, feedback_str: str) -> Optional[str]:
        """Evolves a prompt using the specified operator."""
        raise NotImplementedError("Subclasses must implement 'evolve' or specific operator methods.")


class PlannerPromptEvolver(BasePromptEvolver):
    """Evolves prompts specifically for the Planner Agent."""
    OPERATORS_MAP = {
        "enhance_decomposition": ENHANCE_DECOMPOSITION_OPERATOR_PROMPT,
        "strengthen_constraints": STRENGTHEN_CONSTRAINTS_OPERATOR_PROMPT,
        "add_examples": ADD_EXAMPLES_OPERATOR_PROMPT,
    }
    
    def __init__(self, llm_service: LLMService):
        super().__init__(llm_service, AgentType.PLANNER)

    def get_available_operators(self) -> List[PlannerOperator]:
        return list(self.OPERATORS_MAP.keys())

    def evolve(self, operator_name: PlannerOperator, original_prompt: str, feedback_str: str) -> Optional[str]:
        if operator_name not in self.OPERATORS_MAP:
            logger.error(f"Unknown Planner operator: {operator_name}")
            return None
        operator_prompt_template = self.OPERATORS_MAP[operator_name]
        return self._apply_operator_prompt(operator_prompt_template, original_prompt, feedback_str)

    # Optionally, define direct methods for each operator if preferred for explicitness
    def enhance_decomposition(self, original_prompt: str, feedback_str: str) -> Optional[str]:
        return self.evolve("enhance_decomposition", original_prompt, feedback_str)

    def strengthen_constraints(self, original_prompt: str, feedback_str: str) -> Optional[str]:
        return self.evolve("strengthen_constraints", original_prompt, feedback_str)

    def add_examples(self, original_prompt: str, feedback_str: str) -> Optional[str]:
        return self.evolve("add_examples", original_prompt, feedback_str)


class ExecutorPromptEvolver(BasePromptEvolver):
    """Evolves prompts specifically for the Executor Agent."""
    OPERATORS_MAP = {
        "error_specific_enhancement": ERROR_SPECIFIC_ENHANCEMENT_OPERATOR_PROMPT,
        "add_defensive_programming": ADD_DEFENSIVE_PROGRAMMING_OPERATOR_PROMPT,
        "algorithm_suggestion": ALGORITHM_SUGGESTION_OPERATOR_PROMPT,
    }

    def __init__(self, llm_service: LLMService):
        super().__init__(llm_service, AgentType.EXECUTOR)
    
    def get_available_operators(self) -> List[ExecutorOperator]:
        return list(self.OPERATORS_MAP.keys())

    def evolve(self, operator_name: ExecutorOperator, original_prompt: str, feedback_str: str) -> Optional[str]:
        if operator_name not in self.OPERATORS_MAP:
            logger.error(f"Unknown Executor operator: {operator_name}")
            return None
        operator_prompt_template = self.OPERATORS_MAP[operator_name]
        return self._apply_operator_prompt(operator_prompt_template, original_prompt, feedback_str)

    # Direct methods
    def error_specific_enhancement(self, original_prompt: str, feedback_str: str) -> Optional[str]:
        return self.evolve("error_specific_enhancement", original_prompt, feedback_str)

    def add_defensive_programming(self, original_prompt: str, feedback_str: str) -> Optional[str]:
        return self.evolve("add_defensive_programming", original_prompt, feedback_str)

    def algorithm_suggestion(self, original_prompt: str, feedback_str: str) -> Optional[str]:
        return self.evolve("algorithm_suggestion", original_prompt, feedback_str)


class ContextAwareOperatorSelector:
    """
    Selects an appropriate evolution operator based on context (e.g., agent type, error type).
    For MVP, this uses a simple heuristic (round-robin or random).
    Future: Could use RL or more sophisticated heuristics.
    """
    def __init__(self, planner_evolver: PlannerPromptEvolver, executor_evolver: ExecutorPromptEvolver):
        self.planner_operators = planner_evolver.get_available_operators()
        self.executor_operators = executor_evolver.get_available_operators()
        # For round-robin or stateful selection (optional for now)
        self._planner_op_idx = 0
        self._executor_op_idx = 0

    def select_operator(self, agent_type: AgentType, critic_report_dict: Optional[Dict[str, Any]] = None) -> Optional[AllOperators]:
        """
        Selects an operator.
        
        Args:
            agent_type: The type of agent whose prompt is being modified.
            critic_report_dict: The critic report providing feedback context. (Currently unused by simple selector)
        
        Returns:
            The name of the selected operator, or None if no suitable operator found.
        """
        selected_operator: Optional[AllOperators] = None
        if agent_type == AgentType.PLANNER:
            if not self.planner_operators:
                logger.warning("No planner operators available for selection.")
                return None
            # Simple round-robin for now
            selected_operator = self.planner_operators[self._planner_op_idx % len(self.planner_operators)]
            self._planner_op_idx += 1
            logger.info(f"Selected Planner operator (round-robin): {selected_operator}")
        elif agent_type == AgentType.EXECUTOR:
            if not self.executor_operators:
                logger.warning("No executor operators available for selection.")
                return None
            # Simple round-robin
            selected_operator = self.executor_operators[self._executor_op_idx % len(self.executor_operators)]
            self._executor_op_idx += 1
            logger.info(f"Selected Executor operator (round-robin): {selected_operator}")
        else:
            logger.error(f"Cannot select operator for unknown agent type: {agent_type}")
            return None
            
        # Placeholder: Future logic could use critic_report_dict to make a more informed choice
        # For example, if critic_report_dict suggests a specific kind of error,
        # one might prioritize certain operators.
        # critic_error_type = ErrorType(critic_report_dict.get('error_type', '')) if critic_report_dict else None

        return selected_operator 