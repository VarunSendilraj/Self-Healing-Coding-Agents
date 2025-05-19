from crewai import Crew, Process
# To ensure the script can find other modules in the same package, especially when run directly,
# we might need to adjust Python's path if 'self_healing_agents' is not in a standard location.
# However, for typical package structures and execution via `python -m`, this should resolve.
from self_healing_agents.agents import SelfHealingAgents
from self_healing_agents.tasks import SelfHealingTasks
import sys
import os
# Initialize LLM (e.g., using ChatGroq or ChatOpenAI)
# from langchain_groq import ChatGroq
# llm = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768", api_key="YOUR_GROQ_API_KEY")
# For now, let's assume an LLM is configured. CrewAI will use a default if none is explicitly set for agents/crew.



class Orchestrator:
    def __init__(self, user_request: str):
        self.user_request = user_request
        self.agents_factory = SelfHealingAgents()
        self.tasks_factory = SelfHealingTasks()

    def run_initial_flow(self):
        planner = self.agents_factory.planner_agent()
        executor = self.agents_factory.executor_agent()
        critic = self.agents_factory.critic_agent()

        # Initial dynamic prompt for the executor (placeholder for now)
        initial_executor_prompt = "You are a meticulous Python programmer. Write code according to the spec. Output only code."

        task_plan = self.tasks_factory.plan_task(planner, self.user_request)
        
        # In CrewAI, the output of a task can be used as input for the next task
        # by referencing it in the description or by using the context capabilities.
        # The `context` parameter for a Task tells CrewAI that this task depends on the output of other tasks.

        task_execute = self.tasks_factory.execute_code_task(
            agent=executor,
            # The plan will be injected by CrewAI from the output of task_plan
            # if task_plan is listed in the context of task_execute.
            # The description string will use a placeholder like {context[task_name]} or similar.
            # For this version, we are explicitly formatting the description here, but this
            # might be better handled by making the task_execute description more generic
            # and letting CrewAI inject the context.
            # Let's adjust the task description to expect context.
            plan="{outputs[task_plan]}", # Placeholder expecting output from a task named 'task_plan'
            dynamic_prompt_instructions=initial_executor_prompt
        )
        task_execute.context = [task_plan] # Specifies task_plan as context for task_execute

        task_critique = self.tasks_factory.critique_task(
            agent=critic,
            generated_code="{outputs[task_execute]}", # Placeholder for code from task_execute
            original_request=self.user_request
        )
        task_critique.context = [task_execute] # Specifies task_execute as context

        crew = Crew(
            agents=[planner, executor, critic],
            tasks=[task_plan, task_execute, task_critique],
            process=Process.sequential,
            verbose=True
        )

        print("Kicking off the initial flow (conceptual - no LLM call)...")
        # result = crew.kickoff(inputs={'user_request': self.user_request})
        # print("\nCrew execution finished. Result:")
        # print(result)

        # For demonstration without LLM, print structure
        print(f"User Request: {self.user_request}")
        print("--- Agents ---")
        print(f"Planner: {planner.role}")
        print(f"Executor: {executor.role}")
        print(f"Critic: {critic.role}")
        print("--- Tasks (with dynamic prompt injection for Executor) ---")
        print(f"1. Plan Task Description: {task_plan.description}")
        # To show the dynamic prompt correctly, we access the description from the task object
        # which has `initial_executor_prompt` embedded within its `dynamic_prompt_instructions` argument.
        print(f"2. Execute Code Task (will use output from Plan Task):")
        print(f"   Description (template): {self.tasks_factory.execute_code_task(executor, plan='[Plan from Planner]', dynamic_prompt_instructions=initial_executor_prompt).description}")
        print(f"   This task expects plan from: {task_plan.expected_output}")
        print(f"3. Critique Task (will use output from Execute Code Task):")
        print(f"   Description (template): {self.tasks_factory.critique_task(critic, generated_code='[Code from Executor]', original_request=self.user_request).description}")
        print(f"   This task expects code from: {task_execute.expected_output}")
        
        return "Conceptual flow demonstrated. Check print statements for structure."

if __name__ == '__main__':
    request = "Create a Python function that calculates the factorial of a number."
    orchestrator = Orchestrator(user_request=request)
    orchestrator.run_initial_flow()
