"""
Manages the history of prompt evolution for self-healing agents.
Task 3.3 from PRD_PHASE2.md.
"""
import uuid
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime

from self_healing_agents.error_types import AgentType

@dataclass
class PromptHistoryEntry:
    """Represents a single entry in the prompt evolution history."""
    prompt_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_prompt_id: Optional[str] = None
    agent_type: AgentType
    prompt_text: str
    score: float
    critic_report_summary: Dict[str, Any] # e.g., status, key error messages, test results summary
    evolution_operator_used: Optional[str] = None
    task_id: str
    iteration_number: int # Main self-healing iteration
    evo_iteration_number: int # Internal EvoPrompt iteration within PromptModifier
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Could also include:
    # - llm_call_details (tokens, latency, model_name)
    # - full_critic_report_id (if reports are stored separately)

class PromptHistoryTracker:
    """Tracks and provides access to the history of prompt evolution for a given task."""
    
    def __init__(self, task_id: str):
        """
        Initializes the tracker for a specific task.

        Args:
            task_id: The unique identifier for the task whose prompt history is being tracked.
        """
        self.task_id: str = task_id
        self._history: Dict[str, PromptHistoryEntry] = {} # Store entries by prompt_id for quick lookup
        self._entries_by_agent: Dict[AgentType, List[str]] = {
            agent_type: [] for agent_type in AgentType
        }
        # For lineage, we can trace back using parent_prompt_id

    def add_entry(self, entry: PromptHistoryEntry) -> None:
        """
        Adds a new prompt history entry.

        Args:
            entry: The PromptHistoryEntry to add.
        
        Raises:
            ValueError: If the entry's task_id does not match the tracker's task_id,
                        or if a prompt with the same ID already exists.
        """
        if entry.task_id != self.task_id:
            raise ValueError(f"Entry task_id '{entry.task_id}' does not match Tracker's task_id '{self.task_id}'.")
        if entry.prompt_id in self._history:
            raise ValueError(f"Prompt with ID '{entry.prompt_id}' already exists in history.")
            
        self._history[entry.prompt_id] = entry
        if entry.agent_type not in self._entries_by_agent:
            self._entries_by_agent[entry.agent_type] = []
        self._entries_by_agent[entry.agent_type].append(entry.prompt_id)

    def get_prompt_by_id(self, prompt_id: str) -> Optional[PromptHistoryEntry]:
        """
        Retrieves a specific prompt history entry by its ID.

        Args:
            prompt_id: The ID of the prompt entry to retrieve.

        Returns:
            The PromptHistoryEntry if found, otherwise None.
        """
        return self._history.get(prompt_id)

    def get_prompts_for_agent(self, agent_type: AgentType, sort_by_timestamp: bool = True) -> List[PromptHistoryEntry]:
        """
        Retrieves all prompt history entries for a specific agent type, optionally sorted.

        Args:
            agent_type: The AgentType for which to retrieve prompts.
            sort_by_timestamp: If True (default), sorts entries by timestamp (ascending).

        Returns:
            A list of PromptHistoryEntry objects.
        """
        prompt_ids = self._entries_by_agent.get(agent_type, [])
        entries = [self._history[pid] for pid in prompt_ids if pid in self._history]
        
        if sort_by_timestamp:
            entries.sort(key=lambda e: e.timestamp)
        return entries

    def get_lineage(self, prompt_id: str) -> List[PromptHistoryEntry]:
        """
        Retrieves the lineage (from the given prompt back to its ultimate ancestor) for a prompt.

        Args:
            prompt_id: The ID of the prompt whose lineage is to be retrieved.

        Returns:
            A list of PromptHistoryEntry objects, starting with the given prompt_id
            and going up to its oldest ancestor. Returns an empty list if prompt_id is not found.
        """
        lineage: List[PromptHistoryEntry] = []
        current_entry = self.get_prompt_by_id(prompt_id)
        
        while current_entry:
            lineage.append(current_entry)
            if current_entry.parent_prompt_id:
                current_entry = self.get_prompt_by_id(current_entry.parent_prompt_id)
            else:
                break # No more parents
        
        return lineage # The list is naturally in reverse chronological order (child to parent)

    def get_full_history(self, sort_by_timestamp: bool = True) -> List[PromptHistoryEntry]:
        """
        Retrieves all entries in the history, optionally sorted by timestamp.

        Args:
            sort_by_timestamp: If True (default), sorts entries by timestamp (ascending).

        Returns:
            A list of all PromptHistoryEntry objects.
        """
        entries = list(self._history.values())
        if sort_by_timestamp:
            entries.sort(key=lambda e: e.timestamp)
        return entries

    # Future methods for Task 3.3:
    # - def track_error_to_fix(self, failing_prompt_id: str, fixing_prompt_id: str, error_details: Dict): ...
    # - def get_successful_evolutions_for_operator(self, operator_name: str) -> List[Tuple[PromptHistoryEntry, PromptHistoryEntry]]: ...
    # - Methods for EvolutionAnalytics (these might be better in a separate class that consumes PromptHistoryTracker) 