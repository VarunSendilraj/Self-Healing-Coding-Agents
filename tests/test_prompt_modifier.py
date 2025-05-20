import unittest
from unittest.mock import MagicMock, patch
import dataclasses

from self_healing_agents.prompt_modifier import PromptModifier, PromptInfo, MAX_POPULATION_SIZE_N, EVO_PROMPT_ITERATIONS_T
from self_healing_agents.schemas import (
    CriticReport, 
    CRITIC_STATUS_FAILURE_SYNTAX, 
    CRITIC_STATUS_SUCCESS,
    CRITIC_STATUS_FAILURE_LOGIC,
    CRITIC_STATUS_FAILURE_RUNTIME
)
from self_healing_agents.error_types import AgentType
from self_healing_agents.prompts import (
    EXECUTOR_PROMPT_EVOLUTION_SYSTEM_PROMPT,
    PLANNER_PROMPT_EVOLUTION_SYSTEM_PROMPT
)
from self_healing_agents.llm_service import LLMService
from self_healing_agents.evolution_operators import PlannerPromptEvolver, ExecutorPromptEvolver, ContextAwareOperatorSelector

class TestPromptModifierStructure(unittest.TestCase):

    def setUp(self):
        self.initial_prompt_text = "Initial executor prompt for testing."
        self.initial_score = 0.3
        self.task_id = "test_task_001"
        self.mock_llm_service = MagicMock(spec=LLMService)
        self.initial_critic_report_obj = CriticReport(
            overall_status=CRITIC_STATUS_FAILURE_SYNTAX,
            score=self.initial_score, 
            feedback="Initial syntax error",
            raw_code=self.initial_prompt_text
        )
        self.prompt_modifier = PromptModifier(
            llm_service=self.mock_llm_service,
            task_id=self.task_id
        )
        self.prompt_modifier.initialize_population_for_agent(
            AgentType.EXECUTOR,
            self.initial_prompt_text,
            self.initial_score,
            dataclasses.asdict(self.initial_critic_report_obj)
        )
        self.agent_type_under_test = AgentType.EXECUTOR
        self.prompt_modifier.planner_evolver = MagicMock(spec=PlannerPromptEvolver)
        self.prompt_modifier.executor_evolver = MagicMock(spec=ExecutorPromptEvolver)
        self.prompt_modifier.operator_selector = MagicMock(spec=ContextAwareOperatorSelector)

    def test_initialization(self):
        self.assertEqual(self.prompt_modifier.task_id, self.task_id)
        self.assertEqual(self.prompt_modifier.max_population_size, MAX_POPULATION_SIZE_N)
        
        executor_population = self.prompt_modifier.populations[self.agent_type_under_test]
        self.assertEqual(len(executor_population), 1)
        self.assertEqual(executor_population[0].prompt, self.initial_prompt_text)
        self.assertEqual(executor_population[0].score, self.initial_score)
        self.assertEqual(executor_population[0].critic_report, dataclasses.asdict(self.initial_critic_report_obj))
        self.assertEqual(self.prompt_modifier.current_evo_iteration, 0)
        self.assertEqual(self.prompt_modifier.main_system_healing_attempts, 0)
        self.assertEqual(len(self.prompt_modifier.populations[AgentType.PLANNER]), 0)

    def test_initialization_with_no_initial_prompt_via_method(self):
        pm = PromptModifier(llm_service=self.mock_llm_service, task_id="no_initial")
        self.assertEqual(len(pm.populations[AgentType.EXECUTOR]), 0)
        self.assertEqual(len(pm.populations[AgentType.PLANNER]), 0)
        
    def test_add_to_population_method_works_for_specific_agent(self):
        new_prompt_text = "A new executor prompt."
        new_score = 0.5
        new_critic_report_obj = CriticReport(overall_status=CRITIC_STATUS_SUCCESS, score=new_score, feedback="Improved slightly")
        new_prompt_info = PromptInfo(prompt=new_prompt_text, score=new_score, iteration_created=1, critic_report=dataclasses.asdict(new_critic_report_obj))
        
        self.prompt_modifier._add_to_population(self.agent_type_under_test, new_prompt_info)
        
        executor_population = self.prompt_modifier.populations[self.agent_type_under_test]
        self.assertEqual(len(executor_population), 2)
        self.assertEqual(executor_population[0].score, new_score)
        self.assertEqual(executor_population[1].score, self.initial_score)
        self.assertEqual(len(self.prompt_modifier.populations[AgentType.PLANNER]), 0)

    def test_get_current_population_for_agent_method(self):
        population = self.prompt_modifier.get_current_population_for_agent(self.agent_type_under_test)
        self.assertIsInstance(population, list)
        self.assertEqual(len(population), 1)
        self.assertIsInstance(population[0], PromptInfo)

        planner_population = self.prompt_modifier.get_current_population_for_agent(AgentType.PLANNER)
        self.assertEqual(len(planner_population), 0)

    def test_select_parents_method_for_agent(self):
        executor_population = self.prompt_modifier.populations[self.agent_type_under_test]
        failing_info_arg = executor_population[0] 
        parents = self.prompt_modifier.select_parents(self.agent_type_under_test, failing_info_arg)
        self.assertIsInstance(parents, list)
        self.assertEqual(len(parents), 1) 
        self.assertEqual(parents[0].prompt, self.initial_prompt_text)

    def test_llm_evolve_prompt_candidates_method_exists_and_returns_list(self): 
        self.assertTrue(hasattr(self.prompt_modifier, '_llm_evolve_prompt_candidates'))
        failing_info = self.prompt_modifier.populations[self.agent_type_under_test][0]
        parents = [failing_info]
        critic_report_dict = failing_info.critic_report
        
        # Setup mocks for the new operator-based evolution path
        selected_operator_name = "error_specific_enhancement" # Example for executor
        expected_evolved_text = "Evolved prompt via operator"
        self.prompt_modifier.operator_selector.select_operator.return_value = selected_operator_name
        if self.agent_type_under_test == AgentType.EXECUTOR:
            self.prompt_modifier.executor_evolver.evolve.return_value = expected_evolved_text
        elif self.agent_type_under_test == AgentType.PLANNER: # Though current setup is EXECUTOR
            self.prompt_modifier.planner_evolver.evolve.return_value = expected_evolved_text

        evolved_prompts = self.prompt_modifier._llm_evolve_prompt_candidates(self.agent_type_under_test, parents, critic_report_dict)
        self.assertIsInstance(evolved_prompts, list)
        self.assertEqual(len(evolved_prompts), 1)
        self.assertIsInstance(evolved_prompts[0], str)
        self.assertEqual(evolved_prompts[0], expected_evolved_text)

    def test_evaluate_candidate_prompt_and_create_info(self):
        candidate_prompt_str = "A candidate prompt string."
        mock_critic_report_dict = {"score": 0.7, "feedback": "Good candidate", "overall_status": CRITIC_STATUS_SUCCESS}
        iteration_num = 1
        
        evaluated_info = self.prompt_modifier.evaluate_candidate_prompt_and_create_info(
            candidate_prompt_text=candidate_prompt_str, 
            evaluation_score=mock_critic_report_dict['score'],
            evaluation_critic_report_dict=mock_critic_report_dict,
            current_iteration_number=iteration_num
        )
        self.assertIsInstance(evaluated_info, PromptInfo)
        self.assertEqual(evaluated_info.prompt, candidate_prompt_str)
        self.assertEqual(evaluated_info.score, 0.7)
        self.assertEqual(evaluated_info.critic_report, mock_critic_report_dict)
        self.assertEqual(evaluated_info.iteration_created, iteration_num)

    def test_run_self_healing_iteration_for_agent(self): 
        failing_prompt_text = self.initial_prompt_text
        failing_score = self.initial_score
        failing_critic_dict = dataclasses.asdict(self.initial_critic_report_obj)

        evolved_prompt_text = f"Evolved by operator from: {failing_prompt_text}"
        evolved_score = 0.8 # Higher than failing_score
        evolved_critic_report_dict_from_callback = {"overall_status": CRITIC_STATUS_SUCCESS, "score": evolved_score, "feedback": "Evolved and successful via callback"}
        
        mock_orchestrator_callback = MagicMock(return_value=evolved_critic_report_dict_from_callback)
    
        # Mock the selector to return a specific operator
        selected_operator = "error_specific_enhancement" # Example for executor
        self.prompt_modifier.operator_selector.select_operator.return_value = selected_operator
        
        # Mock the executor_evolver.evolve method to return the desired evolved prompt
        self.prompt_modifier.executor_evolver.evolve.return_value = evolved_prompt_text

        best_prompt = self.prompt_modifier.run_self_healing_iteration(
            agent_to_modify=self.agent_type_under_test,
            failing_prompt_initial_text=failing_prompt_text,
            failing_prompt_initial_score=failing_score,
            failing_prompt_critic_report_dict=failing_critic_dict,
            orchestrator_callback_evaluate_candidate=mock_orchestrator_callback
        )
        
        self.assertEqual(best_prompt, evolved_prompt_text) 
        self.assertEqual(self.prompt_modifier.main_system_healing_attempts, 1)
        
        # Verify selector and evolver calls
        self.prompt_modifier.operator_selector.select_operator.assert_called_once_with(self.agent_type_under_test, failing_critic_dict)
        if self.agent_type_under_test == AgentType.EXECUTOR:
            self.prompt_modifier.executor_evolver.evolve.assert_called_once_with(
                selected_operator, 
                failing_prompt_text, 
                failing_critic_dict.get('feedback', failing_critic_dict.get('summary'))
            )
            self.prompt_modifier.planner_evolver.evolve.assert_not_called()
        # Add similar check for planner if testing that path

        mock_orchestrator_callback.assert_called_once_with(self.agent_type_under_test, evolved_prompt_text)
        final_population = self.prompt_modifier.populations[self.agent_type_under_test]
        self.assertTrue(any(p.prompt == evolved_prompt_text and p.score == evolved_score for p in final_population))

class TestPromptPopulationManagement(unittest.TestCase):
    def setUp(self):
        self.task_id = "test_pop_mgmt_002"
        self.max_pop_size = MAX_POPULATION_SIZE_N
        self.mock_llm_service = MagicMock(spec=LLMService)
        self.prompt_modifier = PromptModifier(
            llm_service=self.mock_llm_service,
            task_id=self.task_id,
            max_population_size=self.max_pop_size
        )
        self.executor_agent_type = AgentType.EXECUTOR
        self.planner_agent_type = AgentType.PLANNER
        self.prompt_modifier.planner_evolver = MagicMock(spec=PlannerPromptEvolver)
        self.prompt_modifier.executor_evolver = MagicMock(spec=ExecutorPromptEvolver)
        self.prompt_modifier.operator_selector = MagicMock(spec=ContextAwareOperatorSelector)

    def test_initialize_population_for_agent_adds_prompt(self):
        prompt1_text = "Initial executor failing prompt"
        prompt1_score = 0.2
        prompt1_report_obj = CriticReport(overall_status=CRITIC_STATUS_FAILURE_RUNTIME, score=prompt1_score, feedback="failed")
        prompt1_report_dict = dataclasses.asdict(prompt1_report_obj)
        
        self.prompt_modifier.initialize_population_for_agent(
            self.executor_agent_type, prompt1_text, prompt1_score, prompt1_report_dict
        )
        executor_pop = self.prompt_modifier.populations[self.executor_agent_type]
        self.assertEqual(len(executor_pop), 1)
        self.assertEqual(executor_pop[0].prompt, prompt1_text)
        self.assertEqual(executor_pop[0].score, prompt1_score)
        self.assertEqual(executor_pop[0].critic_report, prompt1_report_dict)
        self.assertEqual(len(self.prompt_modifier.populations[self.planner_agent_type]), 0)

    def test_run_self_healing_iteration_populates_for_correct_agent(self):
        pm = PromptModifier(llm_service=self.mock_llm_service, task_id=self.task_id, max_population_size=self.max_pop_size)
        # Correctly mock the attributes on the `pm` instance created within this test
        pm.planner_evolver = MagicMock(spec=PlannerPromptEvolver)
        pm.executor_evolver = MagicMock(spec=ExecutorPromptEvolver)
        pm.operator_selector = MagicMock(spec=ContextAwareOperatorSelector)

        self.assertEqual(len(pm.populations[self.executor_agent_type]), 0)
        self.assertEqual(len(pm.populations[self.planner_agent_type]), 0)

        failing_exec_text = "Executor failing prompt"
        failing_exec_score = 0.1
        failing_exec_report_obj = CriticReport(overall_status=CRITIC_STATUS_FAILURE_SYNTAX, score=failing_exec_score, feedback="exec syntax err")
        failing_exec_report_dict = dataclasses.asdict(failing_exec_report_obj)
        
        evolved_exec_text = "Evolved Executor prompt"
        evolved_exec_score = 0.8
        evolved_exec_report_dict = {"overall_status": CRITIC_STATUS_SUCCESS, "score": evolved_exec_score, "feedback": "Evolved exec success"}
        mock_callback_exec = MagicMock(return_value=evolved_exec_report_dict)

        # Mock selector and evolver for executor part
        pm.operator_selector.select_operator.return_value = "error_specific_enhancement"
        pm.executor_evolver.evolve.return_value = evolved_exec_text

        pm.run_self_healing_iteration(
            agent_to_modify=self.executor_agent_type,
            failing_prompt_initial_text=failing_exec_text,
            failing_prompt_initial_score=failing_exec_score,
            failing_prompt_critic_report_dict=failing_exec_report_dict,
            orchestrator_callback_evaluate_candidate=mock_callback_exec
        )
        
        executor_pop = pm.populations[self.executor_agent_type]
        self.assertGreaterEqual(len(executor_pop), 1)
        self.assertTrue(any(p.prompt == evolved_exec_text and p.score == evolved_exec_score for p in executor_pop))
        self.assertEqual(len(pm.populations[self.planner_agent_type]), 0)

        failing_plan_text = "Planner failing prompt"
        failing_plan_score = 0.15
        failing_plan_report_obj = CriticReport(overall_status=CRITIC_STATUS_FAILURE_LOGIC, score=failing_plan_score, feedback="plan logic err")
        failing_plan_report_dict = dataclasses.asdict(failing_plan_report_obj)
        
        evolved_plan_text = "Evolved Planner prompt"
        evolved_plan_score = 0.7
        evolved_plan_report_dict = {"overall_status": CRITIC_STATUS_SUCCESS, "score": evolved_plan_score, "feedback": "Evolved plan success"}
        mock_callback_plan = MagicMock(return_value=evolved_plan_report_dict)

        # Mock selector and evolver for planner part
        pm.operator_selector.select_operator.return_value = "enhance_decomposition"
        pm.planner_evolver.evolve.return_value = evolved_plan_text

        pm.run_self_healing_iteration(
            agent_to_modify=self.planner_agent_type,
            failing_prompt_initial_text=failing_plan_text,
            failing_prompt_initial_score=failing_plan_score,
            failing_prompt_critic_report_dict=failing_plan_report_dict,
            orchestrator_callback_evaluate_candidate=mock_callback_plan
        )
        
        planner_pop = pm.populations[self.planner_agent_type]
        self.assertGreaterEqual(len(planner_pop), 1)
        self.assertTrue(any(p.prompt == evolved_plan_text and p.score == evolved_plan_score for p in planner_pop))
        self.assertGreaterEqual(len(pm.populations[self.executor_agent_type]), 1) 

    def test_add_prompts_maintains_size_and_sort_order_per_agent(self):
        pm = self.prompt_modifier
        pm.populations[self.executor_agent_type] = []
        pm.populations[self.planner_agent_type] = []

        pm.initialize_population_for_agent(self.executor_agent_type, "Exec P1", 0.5, {})
        pm._add_to_population(self.executor_agent_type, PromptInfo("Exec P2", 0.7, 1, {}))
        pm._add_to_population(self.executor_agent_type, PromptInfo("Exec P3", 0.3, 2, {}))
        pm._add_to_population(self.executor_agent_type, PromptInfo("Exec P4", 0.9, 3, {}))

        exec_pop = pm.populations[self.executor_agent_type]
        self.assertEqual(len(exec_pop), min(4, pm.max_population_size))
        if exec_pop:
            self.assertEqual(exec_pop[0].prompt, "Exec P4")
            self.assertEqual(exec_pop[0].score, 0.9)
            if len(exec_pop) > 1:
                self.assertEqual(exec_pop[1].prompt, "Exec P2")
                self.assertEqual(exec_pop[1].score, 0.7)
            if len(exec_pop) > 2:
                self.assertEqual(exec_pop[2].prompt, "Exec P1")
                self.assertEqual(exec_pop[2].score, 0.5)
        
        pm.initialize_population_for_agent(self.planner_agent_type, "Plan P1", 0.6, {})
        pm._add_to_population(self.planner_agent_type, PromptInfo("Plan P2", 0.8, 1, {}))
        
        plan_pop = pm.populations[self.planner_agent_type]
        self.assertEqual(len(plan_pop), 2)
        self.assertEqual(plan_pop[0].prompt, "Plan P2")
        self.assertEqual(plan_pop[0].score, 0.8)

class TestParentSelectionStrategy(unittest.TestCase):
    def setUp(self):
        self.task_id = "test_parent_select_003"
        self.mock_llm_service = MagicMock(spec=LLMService)
        self.prompt_modifier = PromptModifier(llm_service=self.mock_llm_service, task_id=self.task_id)
        self.executor_agent_type = AgentType.EXECUTOR
        self.prompt_modifier.planner_evolver = MagicMock(spec=PlannerPromptEvolver)
        self.prompt_modifier.executor_evolver = MagicMock(spec=ExecutorPromptEvolver)
        self.prompt_modifier.operator_selector = MagicMock(spec=ContextAwareOperatorSelector)

        self.failing_exec_prompt_info = PromptInfo("Failing exec prompt", 0.1, 0, {"feedback": "exec failed"})
        self.good_exec_prompt_info = PromptInfo("Good exec prompt", 0.9, 0, {"feedback": "exec good"})

    def test_select_parents_first_attempt_population_has_only_failing(self):
        self.prompt_modifier.initialize_population_for_agent(
            self.executor_agent_type, 
            self.failing_exec_prompt_info.prompt, 
            self.failing_exec_prompt_info.score, 
            self.failing_exec_prompt_info.critic_report
        )
        parents = self.prompt_modifier.select_parents(self.executor_agent_type, self.failing_exec_prompt_info)
        self.assertEqual(len(parents), 1)
        self.assertEqual(parents[0].prompt, self.failing_exec_prompt_info.prompt)
    
    def test_select_parents_subsequent_attempt_failing_different_from_historical_best(self):
        self.prompt_modifier.initialize_population_for_agent(
            self.executor_agent_type, 
            self.good_exec_prompt_info.prompt, 
            self.good_exec_prompt_info.score, 
            self.good_exec_prompt_info.critic_report
        )
        current_failing_prompt = PromptInfo("Newer failing prompt", 0.05, 1, {"feedback": "exec failed again"})
        self.prompt_modifier._add_to_population(self.executor_agent_type, current_failing_prompt)
        
        parents = self.prompt_modifier.select_parents(self.executor_agent_type, current_failing_prompt)
        self.assertEqual(len(parents), 2)
        prompts_in_parents = {p.prompt for p in parents}
        self.assertIn(current_failing_prompt.prompt, prompts_in_parents)
        self.assertIn(self.good_exec_prompt_info.prompt, prompts_in_parents)

class TestBestPromptSelectionFromRunIteration(unittest.TestCase):
    def setUp(self):
        self.task_id = "test_best_prompt_005"
        self.mock_llm_service = MagicMock(spec=LLMService)
        self.prompt_modifier = PromptModifier(llm_service=self.mock_llm_service, task_id=self.task_id)
        self.agent_type = AgentType.EXECUTOR
        self.prompt_modifier.planner_evolver = MagicMock(spec=PlannerPromptEvolver)
        self.prompt_modifier.executor_evolver = MagicMock(spec=ExecutorPromptEvolver)
        self.prompt_modifier.operator_selector = MagicMock(spec=ContextAwareOperatorSelector)

        self.initial_failing_text = "Initial failing prompt"
        self.initial_failing_score = 0.1
        self.initial_failing_report_dict = {"overall_status":"FAILURE_RUNTIME", "score":self.initial_failing_score, "feedback":"Initial fail"}
        
        # Initialize population with this failing prompt for some tests
        self.prompt_modifier.initialize_population_for_agent(
            self.agent_type,
            self.initial_failing_text,
            self.initial_failing_score,
            self.initial_failing_report_dict
        )

    def run_iteration_with_mocks(self, evolved_prompt_text, evolved_score, evolved_feedback="Evolved success"):
        evolved_critic_report = {"overall_status": CRITIC_STATUS_SUCCESS if evolved_score > self.initial_failing_score else CRITIC_STATUS_FAILURE_LOGIC, 
                                 "score": evolved_score, "feedback": evolved_feedback}
        mock_orchestrator_callback = MagicMock(return_value=evolved_critic_report)

        # Mock the selector and the appropriate evolver's evolve method
        selected_operator_name = "add_defensive_programming" # Example, can be varied if needed per test
        self.prompt_modifier.operator_selector.select_operator.return_value = selected_operator_name
        
        if self.agent_type == AgentType.EXECUTOR:
            self.prompt_modifier.executor_evolver.evolve.return_value = evolved_prompt_text
            self.prompt_modifier.planner_evolver.evolve.reset_mock() # Ensure planner isn't called
        elif self.agent_type == AgentType.PLANNER:
            self.prompt_modifier.planner_evolver.evolve.return_value = evolved_prompt_text
            self.prompt_modifier.executor_evolver.evolve.reset_mock() # Ensure executor isn't called

        return self.prompt_modifier.run_self_healing_iteration(
            agent_to_modify=self.agent_type,
            failing_prompt_initial_text=self.initial_failing_text,
            failing_prompt_initial_score=self.initial_failing_score,
            failing_prompt_critic_report_dict=self.initial_failing_report_dict,
            orchestrator_callback_evaluate_candidate=mock_orchestrator_callback
        )

    def test_returns_improved_prompt_if_evolution_succeeds(self):
        evolved_text = "Improved prompt"
        evolved_score = 0.9 # Higher than initial_failing_score (0.1)
        best_prompt = self.run_iteration_with_mocks(evolved_text, evolved_score)
        self.assertEqual(best_prompt, evolved_text)

    def test_returns_none_if_evolution_does_not_improve_and_no_better_in_pop(self):
        # Ensure population only contains the initial failing prompt or is empty before this call for a clean test
        self.prompt_modifier.populations[self.agent_type] = []
        self.prompt_modifier.initialize_population_for_agent(
            self.agent_type, self.initial_failing_text, self.initial_failing_score, self.initial_failing_report_dict
        )
        
        evolved_text = "Slightly evolved but still bad prompt"
        evolved_score = 0.05 # Lower than initial_failing_score (0.1)
        best_prompt = self.run_iteration_with_mocks(evolved_text, evolved_score)
        self.assertIsNone(best_prompt)

    def test_returns_best_from_population_if_evolution_fails_but_pop_has_better_historical(self):
        # Setup: Populate with a very good historical prompt first
        historical_best_text = "Historically great prompt"
        historical_best_score = 0.95
        historical_best_report = {"score": historical_best_score, "feedback": "Historically great", "overall_status": CRITIC_STATUS_SUCCESS} # Added overall_status
        # Ensure a clean slate for this specific agent type's population before initializing for this test case
        self.prompt_modifier.populations[self.agent_type] = []
        self.prompt_modifier.initialize_population_for_agent(self.agent_type, historical_best_text, historical_best_score, historical_best_report)
        
        # Now, run_self_healing_iteration is triggered by self.initial_failing_text (score 0.1)
        # The evolution within this run_self_healing_iteration also fails to improve much
        evolved_text_current_run = "Evolved but not great"
        evolved_score_current_run = 0.15 # Better than self.initial_failing_score (0.1) but worse than historical_best_score (0.95)
        
        best_prompt = self.run_iteration_with_mocks(evolved_text_current_run, evolved_score_current_run)
        
        # Corrected assertion: 
        # Initial best_score_overall becomes 0.95 (from historical_best_text).
        # Evolution to 0.15 does not change best_score_overall.
        # Final check: best_score_overall (0.95) > failing_prompt_initial_score (0.1) -> True.
        # Returns best_prompt_overall_text, which is historical_best_text.
        self.assertEqual(best_prompt, historical_best_text)

    def test_returns_historical_best_if_current_evolution_is_worse_than_failing_and_historical_is_better_than_failing(self):
        # Setup: Populate with a good historical prompt
        historical_best_text = "Historically good prompt for this scenario"
        historical_best_score = 0.7
        historical_best_report = {"score": historical_best_score, "feedback": "Historically good"}
        self.prompt_modifier.populations[self.agent_type] = [] # Clear population first
        self.prompt_modifier.initialize_population_for_agent(self.agent_type, historical_best_text, historical_best_score, historical_best_report)

        # Current failing prompt (passed to run_self_healing_iteration)
        current_failing_text = "Current failing prompt"
        current_failing_score = 0.2
        current_failing_report = {"score": current_failing_score, "feedback": "Current fail"}

        # Evolution within this run is even worse
        evolved_text_current_run = "Evolved very badly"
        evolved_score_current_run = 0.05 # Worse than current_failing_score (0.2)

        evolved_critic_report_callback = {"overall_status": CRITIC_STATUS_FAILURE_LOGIC, "score": evolved_score_current_run, "feedback": "very bad evolution"}
        mock_orchestrator_callback = MagicMock(return_value=evolved_critic_report_callback)

        with patch.object(self.prompt_modifier, '_llm_evolve_prompt_candidates', return_value=[evolved_text_current_run]):
            best_prompt = self.prompt_modifier.run_self_healing_iteration(
                agent_to_modify=self.agent_type,
                failing_prompt_initial_text=current_failing_text,
                failing_prompt_initial_score=current_failing_score,
                failing_prompt_critic_report_dict=current_failing_report,
                orchestrator_callback_evaluate_candidate=mock_orchestrator_callback
            )
        # Logic: best_score_overall (0.05) is NOT > current_failing_score (0.2). -> else branch
        # best_from_population_after_evo (historical_best_text, score 0.7) IS > current_failing_score (0.2)
        # So, it should return historical_best_text.
        self.assertEqual(best_prompt, historical_best_text)

    def test_returns_none_if_population_becomes_empty_and_no_improvement(self):
        self.prompt_modifier.populations[self.agent_type] = [] # Start with empty population
        # initial_failing_score is 0.1
        evolved_text = "Bad evolved prompt"
        evolved_score = 0.05 # Worse than initial
        # After this run, if max_population_size is small, and only bad prompts are generated,
        # the population might remain empty if initial_failing_text is not re-added or is pruned.
        # The current logic of run_self_healing_iteration adds the current_failing_prompt_info to the population at the start.
        # So the population will have at least that one.
        # If evolved_score (0.05) < initial_failing_score (0.1), and best_from_pop is initial_failing_prompt (0.1) which is not > initial_failing_score (0.1) -> returns None
        best_prompt = self.run_iteration_with_mocks(evolved_text, evolved_score)
        self.assertIsNone(best_prompt)

if __name__ == '__main__':
    unittest.main() 