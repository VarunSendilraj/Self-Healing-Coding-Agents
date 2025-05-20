from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Any, Dict # Any for now, will be more specific later
import logging # Added for logging
import uuid # Add this import

from self_healing_agents.schemas import (
    CriticReport,
    CRITIC_STATUS_SUCCESS,
    CRITIC_STATUS_FAILURE_SYNTAX,
    CRITIC_STATUS_FAILURE_RUNTIME,
    CRITIC_STATUS_FAILURE_LOGIC
)

from self_healing_agents.llm_service import LLMService, LLMServiceError
from self_healing_agents.error_types import AgentType # Added
from self_healing_agents.prompts import ( # Added
    EXECUTOR_PROMPT_EVOLUTION_SYSTEM_PROMPT,
    PLANNER_PROMPT_EVOLUTION_SYSTEM_PROMPT
)
# Import new evolver classes and selector
from self_healing_agents.evolution_operators import (
    PlannerPromptEvolver,
    ExecutorPromptEvolver,
    ContextAwareOperatorSelector,
    AllOperators # For type hinting if needed
)
# Import PromptHistoryTracker and PromptHistoryEntry
from self_healing_agents.history import PromptHistoryTracker, PromptHistoryEntry # Added

# Placeholder for Orchestrator functions if needed for sub-loops (Task 3.6)
# from self_healing_agents.orchestrator import ... 

MAX_POPULATION_SIZE_N = 3 # As per PRD (e.g., N=2 or 3)
EVO_PROMPT_ITERATIONS_T = 1 # As per PRD (e.g., T=1 or T=2)

# Configure logging
logger = logging.getLogger(__name__) # Added

@dataclass
class PromptInfo:
    prompt: str
    score: float
    prompt_id: str = field(default_factory=lambda: str(uuid.uuid4())) # Added prompt_id
    # Optional: age/iteration_created to help with tie-breaking or other strategies
    iteration_created: int = 0 
    # Optional: store the critic report that led to this score for more detailed evolution
    critic_report: Optional[Dict[str, Any]] = None # Changed to Dict to align with Critic's output
    parent_prompt_id: Optional[str] = None # Added to track lineage within PromptModifier before logging to history

class PromptModifier:
    """
    Manages the EvoPrompt process to refine prompts for Planner or Executor agents.
    This class is stateful per task instance and manages populations for different agent types.
    """
    def __init__(self, 
                 llm_service: LLMService, 
                 task_id: str, 
                 max_population_size: int = MAX_POPULATION_SIZE_N):
        """
        Initializes the PromptModifier.

        Args:
            llm_service: An instance of LLMService to use for prompt evolution.
            task_id: A unique identifier for the current task instance.
            max_population_size: The maximum number of prompts to maintain in each agent's population.
        """
        self.llm_service = llm_service
        self.task_id = task_id
        self.max_population_size = max_population_size
        self.populations: Dict[AgentType, List[PromptInfo]] = {
            agent_type: [] for agent_type in AgentType
        }
        self.current_evo_iteration = 0 
        self.main_system_healing_attempts = 0 
        
        # Initialize evolvers and selector (Task 3.2)
        self.planner_evolver = PlannerPromptEvolver(llm_service)
        self.executor_evolver = ExecutorPromptEvolver(llm_service)
        self.operator_selector = ContextAwareOperatorSelector(self.planner_evolver, self.executor_evolver)

        # Initialize PromptHistoryTracker (Task 3.3)
        self.history_tracker = PromptHistoryTracker(task_id=self.task_id) # Added
        
        logger.info(f"[PromptModifier - {self.task_id}] Initialized for multi-agent prompt evolution with operator and history support.")

    def initialize_population_for_agent(self, 
                                      agent_type: AgentType, 
                                      initial_prompt: str, 
                                      initial_score: float, 
                                      initial_critic_report_dict: Dict[str, Any]):
        """Initializes the prompt population for a specific agent type and logs to history."""
        if not isinstance(agent_type, AgentType):
            raise TypeError("agent_type must be an instance of AgentType Enum")
        
        logger.info(f"[PromptModifier - {self.task_id}] Initializing population for {agent_type.name} with prompt (score: {initial_score:.2f}).")
        
        # Create PromptInfo with a new ID, no parent for initial prompt
        prompt_info = PromptInfo(
            prompt=initial_prompt, 
            score=initial_score, 
            iteration_created=0, # Or self.main_system_healing_attempts which is 0 here
            critic_report=initial_critic_report_dict,
            parent_prompt_id=None # Initial prompt has no parent
        )
        self._add_to_population(agent_type, prompt_info)

        # Log to history tracker (Task 3.3)
        history_entry = PromptHistoryEntry(
            prompt_id=prompt_info.prompt_id,
            parent_prompt_id=None,
            agent_type=agent_type,
            prompt_text=prompt_info.prompt,
            score=prompt_info.score,
            critic_report_summary=self._summarize_critic_report(prompt_info.critic_report),
            evolution_operator_used=None, # No operator for initial prompt
            task_id=self.task_id,
            iteration_number=0, # Corresponds to initialization before first healing attempt
            evo_iteration_number=0 # Not an evo iteration
        )
        self.history_tracker.add_entry(history_entry)
        logger.debug(f"    Logged initial prompt {prompt_info.prompt_id} for {agent_type.name} to history.")

    def _get_population_for_agent(self, agent_type: AgentType) -> List[PromptInfo]:
        """Retrieves the prompt population for the given agent type, ensuring it exists."""
        if agent_type not in self.populations:
            logger.warning(f"[PromptModifier - {self.task_id}] Population for {agent_type.name} not found, initializing.")
            self.populations[agent_type] = []
        return self.populations[agent_type]

    def _add_to_population(self, agent_type: AgentType, prompt_info: PromptInfo):
        """Adds a prompt to the specified agent's population, sorts, and prunes."""
        population = self._get_population_for_agent(agent_type)
        population.append(prompt_info)
        population.sort(key=lambda pi: (pi.score, pi.iteration_created), reverse=True)
        if len(population) > self.max_population_size:
            self.populations[agent_type] = population[:self.max_population_size]
        logger.debug(f"[PromptModifier - {self.task_id}] Added prompt to {agent_type.name} population. New size: {len(self.populations[agent_type])}. Best score: {self.populations[agent_type][0].score if self.populations[agent_type] else 'N/A'}")

    def get_current_population_for_agent(self, agent_type: AgentType) -> List[PromptInfo]:
        """Returns the current prompt population for the specified agent type."""
        return self._get_population_for_agent(agent_type)

    def select_parents(self, agent_type: AgentType, failing_prompt_info: PromptInfo) -> List[PromptInfo]:
        """
        Selects parent prompts for evolution for the specified agent type.
        """
        population = self._get_population_for_agent(agent_type)
        selected_parents_map: Dict[str, PromptInfo] = {}

        if not population and not failing_prompt_info:
            logger.warning(f"[PromptModifier - {self.task_id}] Cannot select parents for {agent_type.name}: no population and no failing prompt info.")
            return []
        
        if failing_prompt_info:
             selected_parents_map[failing_prompt_info.prompt] = failing_prompt_info
        
        if population: # If there's an existing population for this agent type
            historically_best_prompt_in_population = population[0] 
            if historically_best_prompt_in_population.prompt not in selected_parents_map:
                selected_parents_map[historically_best_prompt_in_population.prompt] = historically_best_prompt_in_population
        
        if not selected_parents_map and failing_prompt_info : # Fallback if population was empty but we have the failing one
             selected_parents_map[failing_prompt_info.prompt] = failing_prompt_info

        logger.info(f"[PromptModifier - {self.task_id}] Selected {len(selected_parents_map)} parent(s) for {agent_type.name} evolution.")
        return list(selected_parents_map.values())

    def _get_evolution_system_prompt(self, agent_type: AgentType, original_prompt: str, feedback_str: str) -> str:
        """Returns the specialized system prompt for LLM-based evolution based on agent type.
           DEPRECATED in favor of operator-specific prompts. Kept for reference or potential fallback.
        """
        logger.warning("_get_evolution_system_prompt is deprecated. Using operator-specific evolution.")
        if agent_type == AgentType.EXECUTOR:
            return EXECUTOR_PROMPT_EVOLUTION_SYSTEM_PROMPT.format(original_prompt=original_prompt, feedback=feedback_str)
        elif agent_type == AgentType.PLANNER:
            return PLANNER_PROMPT_EVOLUTION_SYSTEM_PROMPT.format(original_prompt=original_prompt, feedback=feedback_str)
        else:
            logger.error(f"[PromptModifier - {self.task_id}] Unknown agent type for evolution: {agent_type}. Defaulting to Executor prompt.")
            return EXECUTOR_PROMPT_EVOLUTION_SYSTEM_PROMPT.format(original_prompt=original_prompt, feedback=feedback_str)

    def _llm_evolve_prompt_candidates(self, agent_type: AgentType, parents: List[PromptInfo], critic_report_dict: Dict[str, Any]) -> List[Tuple[str, Optional[str], Optional[str]]]:
        """
        Generates new candidate prompts using operator-based evolution for the specified agent type.
        Returns a list of tuples: (evolved_prompt_text, parent_id, selected_operator_name)
        """
        if not parents:
            logger.warning(f"[PromptModifier - {self.task_id}] No parents selected for {agent_type.name} evolution. Cannot evolve.")
            return []

        evolved_prompt_details: List[Tuple[str, Optional[str], Optional[str]]] = [] # (text, parent_id, operator)
        primary_parent_info = parents[0] # Assuming the first parent is the most relevant for feedback and base prompt
        original_prompt = primary_parent_info.prompt
        parent_id_for_history = primary_parent_info.prompt_id # Capture parent ID for history
        
        # Construct feedback string from critic_report_dict
        # This could be more sophisticated, e.g. concatenating multiple pieces of info from the report.
        feedback_str = critic_report_dict.get('feedback', critic_report_dict.get('summary', 'No detailed feedback provided.'))
        if not feedback_str:
             feedback_str = "General issues were observed."

        # 1. Select an operator (Task 3.2)
        selected_operator_name = self.operator_selector.select_operator(agent_type, critic_report_dict)

        if not selected_operator_name:
            logger.error(f"[PromptModifier - {self.task_id}] No operator selected for {agent_type.name}. Cannot evolve prompt.")
            return []

        # 2. Apply the selected operator using the appropriate evolver (Task 3.2)
        evolved_prompt_text: Optional[str] = None
        logger.info(f"[PromptModifier - {self.task_id}] Attempting to evolve {agent_type.name} prompt using operator: '{selected_operator_name}'. Parent ID: {parent_id_for_history}, Parent score: {primary_parent_info.score:.2f}. Parent: \"{original_prompt[:70]}...\"")

        try:
            if agent_type == AgentType.PLANNER:
                evolved_prompt_text = self.planner_evolver.evolve(selected_operator_name, original_prompt, feedback_str)
            elif agent_type == AgentType.EXECUTOR:
                evolved_prompt_text = self.executor_evolver.evolve(selected_operator_name, original_prompt, feedback_str)
            else:
                logger.error(f"[PromptModifier - {self.task_id}] Unknown agent type {agent_type} for operator-based evolution.")

            if evolved_prompt_text:
                evolved_prompt_details.append((evolved_prompt_text, parent_id_for_history, selected_operator_name))
                logger.info(f"    Successfully generated evolved {agent_type.name} prompt via operator '{selected_operator_name}': \"{evolved_prompt_text[:70]}...\"")
            else:
                logger.warning(f"    Operator '{selected_operator_name}' for {agent_type.name} did not return an evolved prompt.")
        
        except Exception as e: # Catching potential errors from evolver.evolve or selector
            logger.error(f"    Unexpected error during operator-based evolution for {agent_type.name} with operator '{selected_operator_name}': {e}", exc_info=True)
            
        return evolved_prompt_details

    def evaluate_candidate_prompt_and_create_info(self, 
                                               candidate_prompt_text: str, 
                                               parent_prompt_id: Optional[str], # Added parent_prompt_id
                                               evolution_operator_used: Optional[str], # Added operator used
                                               evaluation_score: float, 
                                               evaluation_critic_report_dict: Dict[str, Any],
                                               current_iteration_number: int,
                                               current_evo_iteration_number: int) -> PromptInfo: # Added evo iteration
        """Packages evaluation results into a PromptInfo object and logs to history."""
        logger.debug(f"[PromptModifier - {self.task_id}] Packaging evaluation for candidate prompt (score: {evaluation_score:.2f}): {candidate_prompt_text[:50]}...")
        
        # Create PromptInfo for internal use (e.g., population management)
        prompt_info = PromptInfo(
            prompt=candidate_prompt_text,
            score=evaluation_score,
            iteration_created=current_iteration_number, # main system healing attempt number
            critic_report=evaluation_critic_report_dict,
            parent_prompt_id=parent_prompt_id # Store parent ID in PromptInfo as well
        )

        # Log to history tracker (Task 3.3)
        # Determine agent_type from context, assuming it's known where this is called
        # This is a bit tricky as this method doesn't inherently know agent_type.
        # For now, we'll assume the caller (run_self_healing_iteration) passes it or we derive it if needed.
        # The history entry needs agent_type. Let's assume the critic_report might hint or it's passed implicitly.
        # For now, placeholder for agent_type - this needs to be resolved by caller.
        # The caller `run_self_healing_iteration` knows the agent_type. This method should take agent_type.
        # Let's refactor this method to take agent_type.

        # This method will be called from run_self_healing_iteration, which knows agent_to_modify.
        # So, this method should ideally receive agent_type.
        # For now, I will defer adding agent_type to this method's signature and handle it in run_self_healing_iteration.

        return prompt_info

    def _summarize_critic_report(self, critic_report: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Creates a concise summary of the critic report for history logging."""
        if not critic_report:
            return {"status": "unavailable", "reason": "No report provided"}
        
        summary = {
            "score": critic_report.get('score'),
            "overall_status": critic_report.get('overall_status'),
            "feedback_snippet": critic_report.get('feedback', '')[:100] + "..." if critic_report.get('feedback') else None,
            "error_type": critic_report.get('error_type'),
            "suggested_agent": critic_report.get('suggested_agent')
        }
        # Remove None values for cleaner logs if desired
        return {k: v for k, v in summary.items() if v is not None}

    def run_self_healing_iteration(self, 
                                   agent_to_modify: AgentType,
                                   failing_prompt_initial_text: str,
                                   failing_prompt_initial_score: float,
                                   failing_prompt_critic_report_dict: Dict[str, Any],
                                   orchestrator_callback_evaluate_candidate: callable) -> Optional[str]:
        """
        Runs one full iteration of the constrained EvoPrompt process (T internal iterations)
        for the specified agent type.

        Args:
            agent_to_modify: The type of agent whose prompt is being modified (PLANNER or EXECUTOR).
            failing_prompt_initial_text: The text of the prompt that just failed.
            failing_prompt_initial_score: The score of the output from the failing prompt.
            failing_prompt_critic_report_dict: The critic report (as a dict) for the failing prompt's output.
            orchestrator_callback_evaluate_candidate: A callable function.
                Expected signature: evaluate(agent_type_to_eval: AgentType, candidate_prompt_text: str) -> Dict[str, Any] (CriticReport as dict)
                This function is responsible for taking a candidate prompt string,
                giving it to the appropriate agent (Planner or Executor), getting its output, 
                evaluating that output (likely involving the full chain ending in a Critic report),
                and returning the full CriticReport as a dictionary.

        Returns:
            The best prompt found after T EvoPrompt iterations for the specified agent, 
            or None if no better prompt is found or an error occurs.
        """
        if not isinstance(agent_to_modify, AgentType):
            logger.error(f"[PromptModifier - {self.task_id}] Invalid agent_to_modify: {agent_to_modify}. Must be AgentType Enum.")
            return None

        self.main_system_healing_attempts += 1
        current_population = self._get_population_for_agent(agent_to_modify)
        
        # Construct PromptInfo for the failing prompt to be used in selection and potentially added to population
        # This initial failing prompt might already be in history if it was an output of a previous iteration.
        # However, if it's the very first prompt of the system run, it gets added by initialize_population_for_agent.
        # We should ensure it has a prompt_id and is correctly logged if it's a *newly evaluated* failing prompt.
        # Let's assume initialize_population_for_agent has already logged it if it's the truly initial one.
        # If this is a subsequent failure of an *evolved* prompt, that evolved prompt would have been logged when it was created.
        # So, current_failing_prompt_info is primarily for population management here.
        current_failing_prompt_info = PromptInfo(
            prompt=failing_prompt_initial_text,
            score=failing_prompt_initial_score,
            # prompt_id will be new if this is a fresh object, or could be passed in if it refers to an existing one.
            # For simplicity, let's assume failing_prompt_initial_text + score + report defines its state for population,
            # but history tracking relies on distinct PromptInfo objects generated at each evolution/evaluation step.
            iteration_created=self.main_system_healing_attempts, 
            critic_report=failing_prompt_critic_report_dict,
            # parent_prompt_id is tricky here if this is a re-evaluation. For now, assume it's None for a *triggering* failure.
            parent_prompt_id=None 
        )
        self._add_to_population(agent_to_modify, current_failing_prompt_info)

        logger.info(f"[PromptModifier - {self.task_id}] Starting self-healing attempt #{self.main_system_healing_attempts} for {agent_to_modify.name} prompt (ID: {current_failing_prompt_info.prompt_id}, initial score: {current_failing_prompt_info.score:.2f}). Current {agent_to_modify.name} population: {len(current_population)} prompts.")

        best_prompt_overall_text: Optional[str] = None
        best_score_overall = failing_prompt_initial_score # Initialize with the score of the prompt that triggered healing

        # Find the current best prompt in this agent's population to compare against
        if current_population: # current_population is already sorted
            best_prompt_overall_text = current_population[0].prompt
            best_score_overall = current_population[0].score
        else: # If population was empty, the failing prompt is the current best
             best_prompt_overall_text = failing_prompt_initial_text
        
        logger.info(f"[PromptModifier - {self.task_id}] Initial best score for {agent_to_modify.name} before EvoPrompt: {best_score_overall:.2f}")

        for t in range(EVO_PROMPT_ITERATIONS_T):
            self.current_evo_iteration = t + 1
            logger.info(f"  EvoPrompt Iteration t={self.current_evo_iteration}/{EVO_PROMPT_ITERATIONS_T} for {agent_to_modify.name}")

            # 1. Select Parents (Task 3.4)
            # The failing_prompt_info for selection should be the one from *this specific healing trigger*
            parents = self.select_parents(agent_to_modify, current_failing_prompt_info)
            if not parents:
                logger.warning(f"    No parents selected for {agent_to_modify.name} in iteration {t+1}. Skipping evolution.")
                continue

            # 2. Evolve Candidate Prompts (Task 3.5)
            # Use the critic_report_dict from the *current failing prompt* to guide evolution.
            # _llm_evolve_prompt_candidates now returns list of (text, parent_id, operator_name)
            evolved_details_list = self._llm_evolve_prompt_candidates(agent_to_modify, parents, failing_prompt_critic_report_dict)
            if not evolved_details_list:
                logger.warning(f"    No new candidate prompts evolved for {agent_to_modify.name} in iteration {t+1}.")
                continue
            
            # 3. Evaluate Candidate Prompts (Task 3.6)
            for candidate_prompt_text, parent_id_for_history, operator_used_for_history in evolved_details_list:
                logger.info(f"    Evaluating candidate {agent_to_modify.name} prompt: \"{candidate_prompt_text[:70]}...\"")
                try:
                    # This callback runs the agent with the new prompt and gets a critic report
                    evaluation_critic_report_dict = orchestrator_callback_evaluate_candidate(
                        agent_to_modify, # Pass agent type to the callback
                        candidate_prompt_text
                    )
                    
                    if not evaluation_critic_report_dict or 'score' not in evaluation_critic_report_dict:
                        logger.error(f"    Evaluation callback for {agent_to_modify.name} candidate prompt did not return a valid report with a score.")
                        continue

                    candidate_score = evaluation_critic_report_dict['score']
                    logger.info(f"    Candidate {agent_to_modify.name} prompt scored: {candidate_score:.2f}")

                    # Package results into PromptInfo
                    # This creates a new PromptInfo object with its own new prompt_id.
                    candidate_prompt_info = self.evaluate_candidate_prompt_and_create_info(
                        candidate_prompt_text=candidate_prompt_text,
                        parent_prompt_id=parent_id_for_history, # Pass parent id from evolved details
                        evolution_operator_used=operator_used_for_history, # Pass operator from evolved details
                        evaluation_score=candidate_score,
                        evaluation_critic_report_dict=evaluation_critic_report_dict,
                        current_iteration_number=self.main_system_healing_attempts,
                        current_evo_iteration_number=self.current_evo_iteration 
                    )
                    
                    # Log evaluated candidate to history tracker (Task 3.3)
                    history_entry = PromptHistoryEntry(
                        prompt_id=candidate_prompt_info.prompt_id, # Use ID from the newly created PromptInfo
                        parent_prompt_id=parent_id_for_history,
                        agent_type=agent_to_modify,
                        prompt_text=candidate_prompt_info.prompt,
                        score=candidate_prompt_info.score,
                        critic_report_summary=self._summarize_critic_report(candidate_prompt_info.critic_report),
                        evolution_operator_used=operator_used_for_history,
                        task_id=self.task_id,
                        iteration_number=self.main_system_healing_attempts,
                        evo_iteration_number=self.current_evo_iteration
                    )
                    self.history_tracker.add_entry(history_entry)
                    logger.debug(f"    Logged evolved candidate {candidate_prompt_info.prompt_id} for {agent_to_modify.name} to history (Parent: {parent_id_for_history}, Operator: {operator_used_for_history}).")

                    # 4. Update Population (Task 3.7)
                    self._add_to_population(agent_to_modify, candidate_prompt_info)

                    # Update overall best if this candidate is better
                    if candidate_score > best_score_overall:
                        best_score_overall = candidate_score
                        best_prompt_overall_text = candidate_prompt_text
                        logger.info(f"    New best {agent_to_modify.name} prompt found with score {best_score_overall:.2f}: \"{best_prompt_overall_text[:70]}...\"")
                
                except Exception as e:
                    logger.error(f"    Error during evaluation of {agent_to_modify.name} candidate prompt '{candidate_prompt_text[:50]}...': {e}", exc_info=True)
                    continue # Skip to the next candidate or iteration

        # 5. Return Best Prompt from Population for this agent type (Task 3.8)
        final_population_for_agent = self._get_population_for_agent(agent_to_modify)
        if final_population_for_agent:
            # The population is sorted, so the first element is the best.
            best_from_population_after_evo = final_population_for_agent[0]
            # We return the text of the prompt that has the highest score found *during this entire method call*
            # which is tracked by best_prompt_overall_text and best_score_overall.
            # It's possible the best in population is from a previous healing attempt if no improvement was made now.
            # The PRD asks for "the prompt with the highest score in the current runtime population".

            # If the best score found during these T iterations is better than what was initially failing...
            if best_score_overall > failing_prompt_initial_score :
                 logger.info(f"[PromptModifier - {self.task_id}] Self-healing for {agent_to_modify.name} completed. Best prompt score: {best_score_overall:.2f}. (Original failing score was {failing_prompt_initial_score:.2f})")
                 return best_prompt_overall_text # Return the best one found in this run
            else:
                 logger.info(f"[PromptModifier - {self.task_id}] Self-healing for {agent_to_modify.name} completed. No improvement found over initial failing score {failing_prompt_initial_score:.2f}. Best in pop score: {best_from_population_after_evo.score:.2f}. Returning best from population.")
                 # If no improvement in this specific call, but the population might hold something better from the past.
                 # We should return the absolute best from the population if it's better than current failing.
                 if best_from_population_after_evo.score > failing_prompt_initial_score:
                     return best_from_population_after_evo.prompt
                 else: # Stick with original or no improvement
                     return None # Or return failing_prompt_initial_text if we must return something? PRD implies return best from pop. If pop best isn't better, it means no improvement.
        
        logger.warning(f"[PromptModifier - {self.task_id}] No prompts in {agent_to_modify.name} population after self-healing attempt. Returning None.")
        return None

    def get_best_prompt_for_agent(self, agent_type: AgentType) -> Optional[PromptInfo]:
        """
        Retrieves the current best PromptInfo object (prompt text + score + report) 
        from the population for the specified agent type.
        Returns None if the population for that agent type is empty.
        """
        population = self._get_population_for_agent(agent_type)
        if population:
            return population[0] # Population is sorted by score descending
        return None


# Placeholder for Orchestrator if PromptModifier needs to call it directly
# (Not recommended, prefer callbacks)
# class OrchestratorPlaceholder:
    # ... existing code ...

    # ... rest of the existing methods ... 