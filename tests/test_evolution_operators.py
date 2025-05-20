"""
Unit tests for the evolution operator classes and selector.
"""
import unittest
from unittest.mock import MagicMock, patch

from self_healing_agents.llm_service import LLMService
from self_healing_agents.error_types import AgentType
from self_healing_agents.evolution_operators import (
    PlannerPromptEvolver,
    ExecutorPromptEvolver,
    ContextAwareOperatorSelector,
    BasePromptEvolver # For testing _apply_operator_prompt if needed directly
)
from self_healing_agents.prompts import (
    ENHANCE_DECOMPOSITION_OPERATOR_PROMPT,
    STRENGTHEN_CONSTRAINTS_OPERATOR_PROMPT,
    ADD_EXAMPLES_OPERATOR_PROMPT,
    ERROR_SPECIFIC_ENHANCEMENT_OPERATOR_PROMPT,
    ADD_DEFENSIVE_PROGRAMMING_OPERATOR_PROMPT,
    ALGORITHM_SUGGESTION_OPERATOR_PROMPT
)

class TestBasePromptEvolver(unittest.TestCase):
    def setUp(self):
        self.mock_llm_service = MagicMock(spec=LLMService)
        # Create a concrete dummy subclass for testing BasePromptEvolver's concrete methods
        class DummyEvolver(BasePromptEvolver):
            def get_available_operators(self):
                return ["dummy_op"]
            def evolve(self, operator_name, original_prompt, feedback_str):
                if operator_name == "dummy_op":
                    return self._apply_operator_prompt("Dummy template: {original_prompt} - {feedback}", original_prompt, feedback_str)
                return None
        self.evolver = DummyEvolver(self.mock_llm_service, AgentType.EXECUTOR) # AgentType doesn't matter much for this test

    def test_apply_operator_prompt_success(self):
        self.mock_llm_service.invoke.return_value = "Evolved prompt text"
        original_prompt = "Original"
        feedback = "Feedback"
        evolved = self.evolver.evolve("dummy_op", original_prompt, feedback)

        self.assertEqual(evolved, "Evolved prompt text")
        self.mock_llm_service.invoke.assert_called_once_with(
            [{"role": "system", "content": f"Dummy template: {original_prompt} - {feedback}"}],
            expect_json=False
        )

    def test_apply_operator_prompt_llm_returns_none(self):
        self.mock_llm_service.invoke.return_value = None
        evolved = self.evolver.evolve("dummy_op", "Original", "Feedback")
        self.assertIsNone(evolved)

    def test_apply_operator_prompt_llm_returns_empty_string(self):
        self.mock_llm_service.invoke.return_value = "  "
        evolved = self.evolver.evolve("dummy_op", "Original", "Feedback")
        self.assertIsNone(evolved)

class TestPlannerPromptEvolver(unittest.TestCase):
    def setUp(self):
        self.mock_llm_service = MagicMock(spec=LLMService)
        self.planner_evolver = PlannerPromptEvolver(self.mock_llm_service)
        self.original_prompt = "Original planner prompt"
        self.feedback = "Planner feedback details"

    def test_get_available_operators(self):
        ops = self.planner_evolver.get_available_operators()
        self.assertIn("enhance_decomposition", ops)
        self.assertIn("strengthen_constraints", ops)
        self.assertIn("add_examples", ops)
        self.assertEqual(len(ops), 3)

    @patch.object(PlannerPromptEvolver, '_apply_operator_prompt')
    def test_enhance_decomposition(self, mock_apply_op):
        mock_apply_op.return_value = "Evolved by enhance_decomposition"
        result = self.planner_evolver.enhance_decomposition(self.original_prompt, self.feedback)
        self.assertEqual(result, "Evolved by enhance_decomposition")
        mock_apply_op.assert_called_once_with(ENHANCE_DECOMPOSITION_OPERATOR_PROMPT, self.original_prompt, self.feedback)

    @patch.object(PlannerPromptEvolver, '_apply_operator_prompt')
    def test_strengthen_constraints(self, mock_apply_op):
        mock_apply_op.return_value = "Evolved by strengthen_constraints"
        result = self.planner_evolver.strengthen_constraints(self.original_prompt, self.feedback)
        self.assertEqual(result, "Evolved by strengthen_constraints")
        mock_apply_op.assert_called_once_with(STRENGTHEN_CONSTRAINTS_OPERATOR_PROMPT, self.original_prompt, self.feedback)

    @patch.object(PlannerPromptEvolver, '_apply_operator_prompt')
    def test_add_examples(self, mock_apply_op):
        mock_apply_op.return_value = "Evolved by add_examples"
        result = self.planner_evolver.add_examples(self.original_prompt, self.feedback)
        self.assertEqual(result, "Evolved by add_examples")
        mock_apply_op.assert_called_once_with(ADD_EXAMPLES_OPERATOR_PROMPT, self.original_prompt, self.feedback)

    def test_evolve_unknown_operator(self):
        result = self.planner_evolver.evolve("unknown_planner_op", self.original_prompt, self.feedback)
        self.assertIsNone(result)

class TestExecutorPromptEvolver(unittest.TestCase):
    def setUp(self):
        self.mock_llm_service = MagicMock(spec=LLMService)
        self.executor_evolver = ExecutorPromptEvolver(self.mock_llm_service)
        self.original_prompt = "Original executor prompt"
        self.feedback = "Executor feedback details"

    def test_get_available_operators(self):
        ops = self.executor_evolver.get_available_operators()
        self.assertIn("error_specific_enhancement", ops)
        self.assertIn("add_defensive_programming", ops)
        self.assertIn("algorithm_suggestion", ops)
        self.assertEqual(len(ops), 3)

    @patch.object(ExecutorPromptEvolver, '_apply_operator_prompt')
    def test_error_specific_enhancement(self, mock_apply_op):
        mock_apply_op.return_value = "Evolved by error_specific_enhancement"
        result = self.executor_evolver.error_specific_enhancement(self.original_prompt, self.feedback)
        self.assertEqual(result, "Evolved by error_specific_enhancement")
        mock_apply_op.assert_called_once_with(ERROR_SPECIFIC_ENHANCEMENT_OPERATOR_PROMPT, self.original_prompt, self.feedback)

    @patch.object(ExecutorPromptEvolver, '_apply_operator_prompt')
    def test_add_defensive_programming(self, mock_apply_op):
        mock_apply_op.return_value = "Evolved by add_defensive_programming"
        result = self.executor_evolver.add_defensive_programming(self.original_prompt, self.feedback)
        self.assertEqual(result, "Evolved by add_defensive_programming")
        mock_apply_op.assert_called_once_with(ADD_DEFENSIVE_PROGRAMMING_OPERATOR_PROMPT, self.original_prompt, self.feedback)

    @patch.object(ExecutorPromptEvolver, '_apply_operator_prompt')
    def test_algorithm_suggestion(self, mock_apply_op):
        mock_apply_op.return_value = "Evolved by algorithm_suggestion"
        result = self.executor_evolver.algorithm_suggestion(self.original_prompt, self.feedback)
        self.assertEqual(result, "Evolved by algorithm_suggestion")
        mock_apply_op.assert_called_once_with(ALGORITHM_SUGGESTION_OPERATOR_PROMPT, self.original_prompt, self.feedback)
    
    def test_evolve_unknown_operator(self):
        result = self.executor_evolver.evolve("unknown_executor_op", self.original_prompt, self.feedback)
        self.assertIsNone(result)

class TestContextAwareOperatorSelector(unittest.TestCase):
    def setUp(self):
        self.mock_llm_service = MagicMock(spec=LLMService)
        self.planner_evolver = PlannerPromptEvolver(self.mock_llm_service)
        self.executor_evolver = ExecutorPromptEvolver(self.mock_llm_service)
        self.selector = ContextAwareOperatorSelector(self.planner_evolver, self.executor_evolver)
        self.critic_report_dict = {"feedback": "Some feedback"} # Dummy report

    def test_select_planner_operator_round_robin(self):
        planner_ops = self.planner_evolver.get_available_operators()
        num_ops = len(planner_ops)
        
        selected_ops = []
        for _ in range(num_ops * 2 + 1): # Cycle through more than twice
            selected_ops.append(self.selector.select_operator(AgentType.PLANNER, self.critic_report_dict))
        
        self.assertEqual(selected_ops[0], planner_ops[0])
        self.assertEqual(selected_ops[1], planner_ops[1])
        self.assertEqual(selected_ops[2], planner_ops[2 % num_ops]) # Handles if num_ops < 3
        self.assertEqual(selected_ops[num_ops], planner_ops[0]) # Should wrap around
        if num_ops > 1: # Check next item in wrapped sequence
             self.assertEqual(selected_ops[num_ops+1], planner_ops[1])

    def test_select_executor_operator_round_robin(self):
        executor_ops = self.executor_evolver.get_available_operators()
        num_ops = len(executor_ops)

        selected_ops = []
        for _ in range(num_ops * 2 + 1):
            selected_ops.append(self.selector.select_operator(AgentType.EXECUTOR, self.critic_report_dict))

        self.assertEqual(selected_ops[0], executor_ops[0])
        self.assertEqual(selected_ops[1], executor_ops[1])
        self.assertEqual(selected_ops[2], executor_ops[2 % num_ops])
        self.assertEqual(selected_ops[num_ops], executor_ops[0])
        if num_ops > 1:
            self.assertEqual(selected_ops[num_ops+1], executor_ops[1])

    def test_select_operator_unknown_agent_type(self):
        class UnknownAgentType:
            pass
        selected_op = self.selector.select_operator(UnknownAgentType(), self.critic_report_dict)
        self.assertIsNone(selected_op)

    def test_select_operator_no_available_operators(self):
        # Mock get_available_operators to return empty list
        with patch.object(self.planner_evolver, 'get_available_operators', return_value=[]):
            selector_no_ops = ContextAwareOperatorSelector(self.planner_evolver, self.executor_evolver)
            selected_op = selector_no_ops.select_operator(AgentType.PLANNER, self.critic_report_dict)
            self.assertIsNone(selected_op)

if __name__ == '__main__':
    unittest.main() 