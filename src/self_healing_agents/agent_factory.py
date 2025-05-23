"""
Factory module for creating agent instances for the Self-Healing Agents system.
"""

from typing import Optional
# Using relative imports for project modules
from .agents import Planner, Executor, Critic
from .llm_service import LLMService
from .prompts import PLANNER_SYSTEM_PROMPT, EXECUTOR_SYSTEM_PROMPT_V1, CRITIC_SYSTEM_PROMPT
from .prompt_modifier import PromptModifier

class SelfHealingAgents:
    """
    Factory class for creating and managing agent instances for the Self-Healing Agents system.
    Provides methods to create different agent types with shared LLM service.
    """
    
    def __init__(self, llm_service: Optional[LLMService] = None):
        """
        Initialize the agent factory with an optional LLM service.
        
        Args:
            llm_service: The LLM service to use for all agents. If None, agents will need
                       to be provided with LLM services when created.
        """
        self.llm_service = llm_service
    
    def set_llm_service(self, llm_service: LLMService):
        """
        Set or update the LLM service used by the factory.
        
        Args:
            llm_service: The LLM service to use for all agents.
        """
        self.llm_service = llm_service
    
    def planner_agent(self, name: str = "Planner") -> Planner:
        """
        Create a Planner agent instance.
        
        Args:
            name: Name for the Planner agent.
            
        Returns:
            A configured Planner agent.
            
        Raises:
            ValueError: If no LLM service is available.
        """
        if not self.llm_service:
            raise ValueError("No LLM service available. Set an LLM service before creating agents.")
        
        return Planner(
            name=name,
            llm_service=self.llm_service,
            system_prompt=PLANNER_SYSTEM_PROMPT
        )
    
    def executor_agent(self, name: str = "Executor") -> Executor:
        """
        Create an Executor agent instance.
        
        Args:
            name: Name for the Executor agent.
            
        Returns:
            A configured Executor agent.
            
        Raises:
            ValueError: If no LLM service is available.
        """
        if not self.llm_service:
            raise ValueError("No LLM service available. Set an LLM service before creating agents.")
        
        return Executor(
            name=name,
            llm_service=self.llm_service,
            system_prompt=EXECUTOR_SYSTEM_PROMPT_V1
        )
    
    def critic_agent(self, name: str = "Critic") -> Critic:
        """
        Create a Critic agent instance.
        
        Args:
            name: Name for the Critic agent.
            
        Returns:
            A configured Critic agent.
            
        Raises:
            ValueError: If no LLM service is available.
        """
        if not self.llm_service:
            raise ValueError("No LLM service available. Set an LLM service before creating agents.")
        
        return Critic(
            name=name,
            llm_service=self.llm_service,
            system_prompt=CRITIC_SYSTEM_PROMPT
        )
    
    def prompt_modifier_agent(self, name: str = "PromptModifier") -> PromptModifier:
        """
        Create a PromptModifier agent instance.
        
        Args:
            name: Name for the PromptModifier agent.
            
        Returns:
            A configured PromptModifier agent.
            
        Raises:
            ValueError: If no LLM service is available.
        """
        if not self.llm_service:
            raise ValueError("No LLM service available. Set an LLM service before creating agents.")
        
        return PromptModifier(
            name=name,
            llm_service=self.llm_service
        )
