import pytest
from unittest.mock import MagicMock, call, patch # Import call for checking multiple calls and patch

from self_healing_agents.llm_service import LLMService
from self_healing_agents.agents import Planner, Executor, Critic # Updated import
from self_healing_agents.prompts import PLANNER_SYSTEM_PROMPT, EXECUTOR_SYSTEM_PROMPT_V1, CRITIC_SYSTEM_PROMPT

@pytest.fixture
def mock_llm_service():
    """Fixture to create a MagicMock for LLMService."""
    service = MagicMock(spec=LLMService)
    return service

def test_basic_sequential_workflow(mock_llm_service):
    """
    Tests the basic sequential workflow: Planner -> Executor -> Critic.
    Mocks LLMService responses to verify data flow and agent interaction.
    """
    # 1. Setup: User request and expected LLM responses
    user_request = "Create a Python function to add two numbers."
    
    # expected_plan should be a dictionary, as Planner.run expects to return a Dict
    # and Executor.run expects a Dict with a 'steps' key.
    expected_plan_dict = {"steps": "1. Define a function `add_numbers(a, b)`. 2. It should return a + b."}
    expected_code = "def add_numbers(a, b):\n    return a + b"
    # expected_evaluation is for the old critic LLM call, not directly used by placeholder

    # Configure the mock LLMService to return different values for each call
    mock_llm_service.invoke.side_effect = [
        expected_plan_dict,     # First call (Planner) - now a dictionary
        expected_code,          # Second call (Executor)
        # Critic call is removed as it uses placeholder logic
    ]

    # 2. Initialize Agents with the mocked LLMService
    # Use updated class names and pass 'name' argument
    planner = Planner(name="TestPlanner", llm_service=mock_llm_service, system_prompt=PLANNER_SYSTEM_PROMPT)
    executor = Executor(name="TestExecutor", llm_service=mock_llm_service, system_prompt=EXECUTOR_SYSTEM_PROMPT_V1)
    critic = Critic(name="TestCritic", llm_service=mock_llm_service, system_prompt=CRITIC_SYSTEM_PROMPT)

    # 3. Execute Workflow
    # Planner
    # Update method calls to .run() and arguments according to new agent definitions
    actual_plan = planner.run(user_request=user_request)
    assert actual_plan == expected_plan_dict # Assert against the dictionary

    # Executor
    actual_code = executor.run(plan=actual_plan, original_request=user_request)
    assert actual_code == expected_code

    # Critic
    # For Critic.run, the placeholder logic is now in evaluate_code, 
    # and run orchestrates this. The mock setup needs to align.
    # The test_workflow.py currently tests the *older* Critic behavior where it directly calls LLM.
    # For Task 1.5, the Critic.run calls evaluate_code which is placeholder.
    # We need to adjust what the mock_llm_service.invoke.side_effect returns for the critic,
    # or acknowledge that this specific test will change significantly once the critic uses LLM again.
    # For now, let's assume the critic.run() bypasses LLM for placeholder and returns the direct eval.
    # The test for Task 1.4 tested the LLM path for critic. 
    # This test needs to be reconciled with the new Critic structure for Task 1.5 (placeholder logic)
    # We will assume the critic's run method (for Task 1.5 placeholder) returns the evaluation directly.
    # The current mock_llm_service.invoke.side_effect[2] is for an LLM call that Critic no longer makes in placeholder mode.

    # Let's adjust the test for the placeholder critic behavior for Task 1.5.
    # The Critic.run method will call self.evaluate_code(generated_code, task_description)
    # which is now the placeholder. So, the mock_llm_service won't be called by Critic.run directly.

    # Re-configuring side_effect for Planner and Executor only for this test as Critic is placeholder.
    mock_llm_service.invoke.side_effect = [
        expected_plan_dict,     # First call (Planner) - now a dictionary
        expected_code,          # Second call (Executor)
        # Critic call is removed as it uses placeholder logic
    ]
    mock_llm_service.reset_mock() # Reset call count and side_effect index
    mock_llm_service.invoke.side_effect = [
        expected_plan_dict, # Use the dictionary here as well
        expected_code,
    ]

    # Planner
    actual_plan = planner.run(user_request=user_request)
    # Executor
    actual_code = executor.run(plan=actual_plan, original_request=user_request)
    
    # Critic (uses placeholder logic, does not call LLM via invoke in this phase)
    # The critic.run method now has a signature: (self, generated_code: str, task_description: str, plan: Dict)
    # It will internally call self.evaluate_code which is the placeholder.
    # The previous expected_evaluation was for an LLM call.
    # The placeholder will return a fixed structure.
    
    # Let's define what the placeholder critic would return for a success scenario:
    expected_placeholder_critic_success_report = {
        "status": "SUCCESS_EXECUTION",
        "score": 1.0,
        "error_details": None,
        "test_results": [],
        "summary": "Code executed successfully. Test cases generated.",
        "execution_stdout": "",
        "execution_stderr": ""
    }

    # For this specific test, mock _generate_test_cases to prevent LLM call from Critic
    with patch.object(critic, '_generate_test_cases', return_value=[]) as mock_critic_generate_tests:
        actual_evaluation = critic.run(generated_code=actual_code, task_description=user_request, plan=actual_plan)
    
    assert actual_evaluation == expected_placeholder_critic_success_report

    # 4. Assert LLMService.invoke calls (should now be 2: Planner, Executor)
    assert mock_llm_service.invoke.call_count == 2

    # Define expected user messages for each agent based on how they format their prompts
    # IMPORTANT: Use \\n to match the repr() output of the actual mock call argument string
    expected_executor_user_prompt = (
        f"Original User Request:\\n{user_request}\\n\\n"
        f"Execution Plan:\\n{expected_plan_dict['steps']}\\n\\n"
        f"Please generate Python code to accomplish this. Ensure you only output the raw Python code, without any markdown formatting or explanations."
    )

    # Check details of each call to llm_service.invoke individually
    all_actual_calls = mock_llm_service.invoke.call_args_list

    # Expected Planner call details
    expected_planner_messages = [
        {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
        {"role": "user", "content": user_request}
    ]
    # Assert Planner call
    planner_actual_call_args, planner_actual_call_kwargs = all_actual_calls[0]
    assert planner_actual_call_args[0] == expected_planner_messages
    assert planner_actual_call_kwargs == {"expect_json": True}

    # Expected Executor call details
    expected_executor_messages = [
        {"role": "system", "content": EXECUTOR_SYSTEM_PROMPT_V1},
        {"role": "user", "content": expected_executor_user_prompt}
    ]
    # Assert Executor call
    executor_actual_call_args, executor_actual_call_kwargs = all_actual_calls[1]
    
    print(f"DEBUG: Actual executor content: {repr(executor_actual_call_args[0][1]['content'])}")
    print(f"DEBUG: Expected executor content: {repr(expected_executor_user_prompt)}")

    assert executor_actual_call_args[0] == expected_executor_messages
    assert executor_actual_call_kwargs == {"expect_json": False}

    # Verify the `expect_json` argument for the calls specifically (already covered by above)
    # args_list = mock_llm_service.invoke.call_args_list
    # assert args_list[0][0][0] == expected_calls[0].args[0] # Planner messages
    # assert args_list[0][1] == {"expect_json": True} # Planner kwargs (expect_json=True)
    
    # assert args_list[1][0][0] == expected_calls[1].args[0] # Executor messages
    # assert args_list[1][1] == {"expect_json": False} # Executor kwargs (expect_json=False)

    # Remove checks for the third (Critic) call as it's no longer made to LLMService in this phase
    # assert args_list[2][0][0] == expected_calls[2].args[0] # Critic messages
    # assert args_list[2][1] == {"expect_json": True}     # Critic kwargs (expect_json=True) 