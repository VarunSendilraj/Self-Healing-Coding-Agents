from crewai import Task, Agent

class SelfHealingTasks:
    def plan_task(self, agent: Agent, user_request: str) -> Task:
        return Task(
            description=f"Analyze the following user request and create a high-level plan and code structure to address it. User request: {user_request}",
            expected_output="A textual description of the plan and code structure.",
            agent=agent
        )

    def execute_code_task(self, agent: Agent, plan: str, dynamic_prompt_instructions: str) -> Task:
        # The dynamic_prompt_instructions will come from the PromptModifier in later stages.
        # For initial setup, it can be a placeholder.
        return Task(
            description=f"Generate Python code based on the following plan and instructions:\n\nPlan:\n{plan}\n\nExecution Instructions:\n{dynamic_prompt_instructions}",
            expected_output="A string containing only the generated Python code.",
            agent=agent
        )

    def critique_task(self, agent: Agent, generated_code: str, original_request: str) -> Task:
        return Task(
            description=f"Evaluate the following Python code. Original request: '{original_request}'. Code:\n```python\n{generated_code}\n```\nExecute the code, check for errors, and if possible, generate 1-2 simple test cases based on the original request to verify its logic. Provide a structured report including errors, test results, and a quantifiable score (0.0-1.0).",
            expected_output="A JSON object or string containing the structured feedback: {'status': 'SUCCESS'|'FAILURE_SYNTAX'|'FAILURE_RUNTIME'|'FAILURE_LOGIC', 'score': float, 'error_details': str_or_null, 'test_results': list_of_tests_or_null, 'summary': str}}.",
            agent=agent
            # output_json_schema might be useful here later
        ) 