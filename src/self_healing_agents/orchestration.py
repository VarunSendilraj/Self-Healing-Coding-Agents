from crewai import Crew, Process
# Using relative imports to make it work when running from the project root
from .agent_factory import SelfHealingAgents
from .tasks import SelfHealingTasks
from .agents import STATUS_SUCCESS, STATUS_LOGICAL_ERROR, STATUS_CRITICAL_RUNTIME_ERROR, STATUS_CRITICAL_SYNTAX_ERROR, NO_TESTS_FOUND
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
    
    def run_enhanced_flow(self):
        """
        Runs the enhanced flow with the direct error-fixing attempt step before engaging the Self-Healing Module.
        This implements a single attempt where the agent system tries to append the error message
        to the executor to directly fix the error before falling back to the Self-Healing Module.
        """
        # Create agent instances
        planner = self.agents_factory.planner_agent()
        executor = self.agents_factory.executor_agent()
        critic = self.agents_factory.critic_agent()
        prompt_modifier = self.agents_factory.prompt_modifier_agent()
        
        # Initial system prompt for the executor
        initial_executor_prompt = "You are a meticulous Python programmer. Write code according to the spec. Output only code."
        
        # Step 1: Planning phase - Generate a plan for the task
        print("\n=== Step 1: Planning Phase ===")
        plan_result = planner.run(self.user_request)
        print(f"Plan generated:\n{plan_result}")
        
        # Step 2: Initial execution phase - Generate code based on the plan
        print("\n=== Step 2: Initial Execution Phase ===")
        initial_code = executor.run(plan_result, self.user_request)
        print(f"Initial code generated:\n{initial_code}")
        
        # Step 3: Initial critique phase - Evaluate the generated code
        print("\n=== Step 3: Initial Critique Phase ===")
        critique_result = critic.run(initial_code, self.user_request, plan_result)
        
        # Add detailed logging of the critique result
        print("DIAGNOSTIC: Initial critique result keys: ", critique_result.keys() if isinstance(critique_result, dict) else "Not a dict")
        print("DIAGNOSTIC: Initial critique overall_status: ", critique_result.get('overall_status', 'NOT_FOUND') if isinstance(critique_result, dict) else "Not a dict")
        print("DIAGNOSTIC: Initial critique score: ", critique_result.get('quantitative_score', 'NOT_FOUND') if isinstance(critique_result, dict) else "Not a dict")
        print("DIAGNOSTIC: Initial critique test results: ", len(critique_result.get('test_results', [])) if isinstance(critique_result, dict) else "Not a dict")
        print("DIAGNOSTIC: Initial critique failed_test_details: ", len(critique_result.get('failed_test_details', [])) if isinstance(critique_result, dict) else "Not a dict")
        
        # Extract the overall status from the critique
        if isinstance(critique_result, dict):
            # Check for field name variations (either 'overall_status' or 'status')
            if 'overall_status' in critique_result:
                overall_status = critique_result.get('overall_status')
            else:
                overall_status = critique_result.get('status', 'UNKNOWN_ERROR')
                
            # Check for field name variations (either 'quantitative_score' or 'score')
            if 'quantitative_score' in critique_result:
                score = critique_result.get('quantitative_score')
            else:
                score = critique_result.get('score', 0.0)
        else:
            # Handle unexpected critique result format
            print("Error: Unexpected format for critique result. Cannot proceed with self-healing.")
            return {"error": "Unexpected critique result format", "final_code": initial_code}
            
        # Check if the code was successful
        if overall_status == STATUS_SUCCESS:
            print("Initial code is successful! No need for error fixing or self-healing.")
            return {"status": "SUCCESS", "final_code": initial_code, "critique": critique_result}
            
        # Store best code and score so far
        best_code = initial_code
        best_score = score
        
        # Step 4: Direct Error-Fixing Attempt (new step added to the flow)
        print("\n=== Step 4: Direct Error-Fixing Attempt ===")
        print(f"Code has issues. Overall status: {overall_status}, Score: {score}")
        print("Attempting direct error fix using the error report...")
        
        # Call the direct fix method on the executor
        fixed_code = executor.direct_fix_attempt(
            original_code=initial_code,
            error_report=critique_result,
            task_description=self.user_request,
            plan=plan_result
        )
        
        # Print detailed info about the code being sent for evaluation
        print("DIAGNOSTIC: Direct fix code being evaluated:\n", fixed_code[:200], "..." if len(fixed_code) > 200 else "")
        
        # Re-evaluate the directly fixed code
        print("Re-evaluating the directly fixed code...")
        direct_fix_critique = critic.run(fixed_code, self.user_request, plan_result)
        
        # Add detailed logging of the direct fix critique result
        print("DIAGNOSTIC: Direct fix critique type:", type(direct_fix_critique))
        print("DIAGNOSTIC: Direct fix critique result keys: ", direct_fix_critique.keys() if isinstance(direct_fix_critique, dict) else "Not a dict")
        print("DIAGNOSTIC: Direct fix critique overall_status: ", direct_fix_critique.get('overall_status', 'NOT_FOUND') if isinstance(direct_fix_critique, dict) else "Not a dict")
        print("DIAGNOSTIC: Direct fix critique score: ", direct_fix_critique.get('quantitative_score', 'NOT_FOUND') if isinstance(direct_fix_critique, dict) else "Not a dict")
        print("DIAGNOSTIC: Direct fix critique test results: ", len(direct_fix_critique.get('test_results', [])) if isinstance(direct_fix_critique, dict) else "Not a dict")
        print("DIAGNOSTIC: Direct fix critique failed_test_details: ", len(direct_fix_critique.get('failed_test_details', [])) if isinstance(direct_fix_critique, dict) else "Not a dict")
        
        # Extract the overall status from the direct fix critique
        if isinstance(direct_fix_critique, dict):
            # Check for field name variations and use the correct ones
            # The Critic might return either 'overall_status' or 'status'
            if 'overall_status' in direct_fix_critique:
                direct_fix_status = direct_fix_critique.get('overall_status')
            else:
                direct_fix_status = direct_fix_critique.get('status', 'UNKNOWN_ERROR')
                
            # The Critic might return either 'quantitative_score' or 'score'
            if 'quantitative_score' in direct_fix_critique:
                direct_fix_score = direct_fix_critique.get('quantitative_score')
            else:
                direct_fix_score = direct_fix_critique.get('score', 0.0)
                
            print(f"Direct fix result - Status: {direct_fix_status}, Score: {direct_fix_score}")
            
            # If the direct fix was successful, return the fixed code
            if direct_fix_status in ['SUCCESS', 'PARTIAL_SUCCESS']:
                print("\n=== Step 5: Self-Healing Module (Skipped) ===")
                print("Direct error-fixing was successful. Skipping Self-Healing Module.")
                return {
                    "status": "COMPLETED",
                    "source": "DIRECT_FIX",
                    "score": direct_fix_score,
                    "final_code": fixed_code
                }
            # Update best code and score if direct fix improved the code
            if direct_fix_score > best_score:
                best_code = fixed_code
                best_score = direct_fix_score
                print(f"Direct fix improved the code (Score: {best_score}), but still has issues.")
        else:
            # Handle unexpected critique result format
            print("Error: Unexpected format for direct fix critique result.")
        
        # Step 5: Self-Healing Module (only if direct fix didn't fully succeed)
        print("\n=== Step 5: Self-Healing Module ===")
        print("Direct error-fixing attempt was not fully successful. Engaging Self-Healing Module...")
        
        # Call to the prompt modifier to generate optimized prompts
        print("Generating optimized prompts based on critique feedback...")
        optimized_prompt = prompt_modifier.run(
            initial_prompt=initial_executor_prompt,
            feedback=direct_fix_critique,  # Use the critique from the direct fix attempt
            code=best_code,                # Use the best code so far (which might be from direct fix)
            task_description=self.user_request
        )
        
        # Update the executor with the optimized prompt
        print("Updating executor with optimized prompt...")
        executor.set_prompt(optimized_prompt)
        
        # Re-run executor with the optimized prompt
        print("Regenerating code with optimized prompt...")
        healed_code = executor.run(plan_result, self.user_request)
        
        # Final evaluation of the healed code
        print("Evaluating the healed code...")
        healed_critique = critic.run(healed_code, self.user_request, plan_result)
        
        if isinstance(healed_critique, dict):
            healed_status = healed_critique.get('overall_status', 'UNKNOWN_ERROR')
            healed_score = healed_critique.get('quantitative_score', 0.0)
            
            # Compare with the best score so far (which might be from direct fix)
            if healed_score > best_score:
                best_code = healed_code
                best_score = healed_score
                best_source = "SELF_HEALING"
            else:
                best_source = "DIRECT_FIX" if best_code == fixed_code else "INITIAL"
        else:
            # Handle unexpected critique result format
            best_source = "DIRECT_FIX" if best_code == fixed_code else "INITIAL"
            
        # Return the final result with the best code
        return {
            "status": "COMPLETED",
            "final_code": best_code,
            "best_source": best_source,
            "best_score": best_score,
            "initial_critique": critique_result,
            "direct_fix_critique": direct_fix_critique,
            "self_healing_critique": healed_critique if 'healed_critique' in locals() else None
        }

if __name__ == '__main__':
    request = "Create a Python function that calculates the factorial of a number."
    orchestrator = Orchestrator(user_request=request)
    orchestrator.run_initial_flow()
