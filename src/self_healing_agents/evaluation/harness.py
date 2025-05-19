import sys
import os
# Ensure the src directory is in the Python path for relative imports to work correctly
# when running the module directly or with python -m
# Get the absolute path to the directory containing the current file (evaluation)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the path to the src directory (evaluation -> self_healing_agents -> src)
src_dir = os.path.dirname(os.path.dirname(current_dir)) 
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import logging
import re # Added for stripping ANSI codes
from typing import Any, Dict, List, Tuple, Optional
import datetime

# Actual agent and LLM service imports
from self_healing_agents.agents import Planner, Executor, Critic
from self_healing_agents.llm_service import LLMService, LLMServiceError
from self_healing_agents.prompts import DEFAULT_EXECUTOR_SYSTEM_PROMPT
# Import actual PromptModifier and related schemas
from self_healing_agents.prompt_modifier import PromptModifier as ActualPromptModifier, PromptInfo
from self_healing_agents.schemas import CriticReport as CriticReportDataclass, CRITIC_STATUS_FAILURE_EVALUATION

# --- Console Coloring Utility ---
class TermColors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    @staticmethod
    def color_text(text: str, color: str) -> str:
        return f"{color}{text}{TermColors.ENDC}"

# --- Logging Setup ---
LOG_FILE_NAME = "evaluation_harness.log"

# Function to strip ANSI escape codes
def strip_ansi_codes(text: str) -> str:
    return re.sub(r'\\x1b\\[[0-9;]*[mK]', '', text)

# Custom filter to strip ANSI codes
class StripAnsiFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = strip_ansi_codes(record.getMessage())
        record.args = () # Clear args as msg is now pre-formatted
        return True

# Remove existing handlers if any to prevent duplicate logs on re-runs in some environments
root_logger = logging.getLogger()
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)

# Configure basic console logging (will be more controlled later)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__) # Logger for this module

# File Handler - overwrites file on each run
file_handler = logging.FileHandler(LOG_FILE_NAME, mode='w')
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
file_handler.setLevel(logging.INFO)
file_handler.addFilter(StripAnsiFilter()) # Add the filter here
logging.getLogger().addHandler(file_handler) # Add to root logger to capture all logs

# --- Configuration ---
MAX_MAIN_LOOP_ITERATIONS = 3
CRITIC_SUCCESS_THRESHOLD = 0.9

# --- Placeholder PromptModifier ---
# class PromptModifier:
#    ... (placeholder class removed) ...

# --- Evaluation Harness Logic ---

def run_single_task(
    task_definition: Dict[str, Any],
    planner: Planner,
    executor: Executor,
    critic: Critic,
    llm_service_instance: LLMService # Pass LLM service for PromptModifier
) -> Dict[str, Any]:
    task_id = task_definition["id"]
    task_description = task_definition["description"]
    initial_prompt = task_definition["initial_executor_prompt"]

    logger.info(f"--- Starting Task: {TermColors.color_text(task_id, TermColors.HEADER)} ---")
    logger.info(f"Description: {TermColors.color_text(task_description, TermColors.CYAN)}")
    
    executor.set_prompt(initial_prompt)
    
    prompt_modifier_for_task: Optional[ActualPromptModifier] = None # Instance for the real PromptModifier

    current_executor_prompt = initial_prompt
    task_run_log: Dict[str, Any] = {
        "task_id": task_id,
        "description": task_description,
        "initial_prompt": initial_prompt,
        "iterations_data": [],
        "final_status": "UNKNOWN",
        "final_code": None,
        "final_score": 0.0,
        "total_iterations": 0
    }

    # Define the orchestrator callback function here, so it has access to executor, critic, etc.
    def _orchestrator_callback_evaluate_candidate(candidate_prompt_text: str, current_task_description: str) -> CriticReportDataclass:
        logger.info(f"    [Harness Callback] Evaluating candidate prompt via Executor & Critic: \"{candidate_prompt_text[:70]}...\"")
        temp_executor_generated_code = ""
        temp_critic_report_dict: Dict[str, Any]
        try:
            # 1. Set new prompt for Executor (temporarily for this candidate)
            #    The actual executor instance's prompt is updated.
            #    We assume executor.run() uses its internal current prompt.
            #    If not, we might need a way to pass the candidate_prompt_text directly to executor.run()
            original_executor_prompt_before_candidate_eval = executor.system_prompt
            executor.set_prompt(candidate_prompt_text)

            # 2. Run Executor (re-using the last plan from the main loop)
            #    The 'plan' variable should be accessible from the outer scope of run_single_task
            last_plan = task_run_log["iterations_data"][-1]["planner_output"] if task_run_log["iterations_data"] and "planner_output" in task_run_log["iterations_data"][-1] else {"plan_steps": ["No previous plan available for candidate evaluation."]}
            
            logger.info(f"      [Harness Callback] Running Executor with candidate prompt.")
            temp_executor_generated_code = executor.run(plan=last_plan, original_request=current_task_description)
            logger.info(f"      [Harness Callback] Executor generated code for candidate.")

            # 3. Run Critic
            logger.info(f"      [Harness Callback] Running Critic on new code.")
            temp_critic_report_dict = critic.run(generated_code=temp_executor_generated_code, task_description=current_task_description, plan=last_plan)
            
            # Restore original executor prompt after evaluation of this candidate
            executor.set_prompt(original_executor_prompt_before_candidate_eval)

        except Exception as e:
            logger.error(f"      [Harness Callback] EXCEPTION during candidate evaluation: {e}")
            # Restore original executor prompt in case of error too
            if 'original_executor_prompt_before_candidate_eval' in locals():
                 executor.set_prompt(original_executor_prompt_before_candidate_eval)
            
            temp_critic_report_dict = {
                "status": CRITIC_STATUS_FAILURE_EVALUATION,
                "score": 0.0,
                "summary": f"Harness callback failed to evaluate candidate prompt: {e}",
                "error_details": str(e),
                "test_results": [],
                "generated_code_for_report": temp_executor_generated_code # include the code if it was generated
            }
        
        # Ensure the dictionary is converted to the CriticReportDataclass
        # The Critic already returns a Dict. The PromptInfo expects a CriticReport dataclass.
        # We need to make sure CriticReportDataclass can be initialized from this dict.
        # For now, let's assume critic.run() returns a dict that IS the CriticReportDataclass structure
        # or we convert it carefully.
        # If critic.run already returns the dataclass, this is fine. If it returns a dict, conversion is needed.
        # Let's assume critic.run returns a dict that matches the CriticReportDataclass fields.
        try:
            return CriticReportDataclass(**temp_critic_report_dict)
        except TypeError as te: # If ** unpacking fails due to mismatched keys
            logger.error(f"  [Harness Callback] Type error converting dict to CriticReportDataclass: {te}. Dict was: {temp_critic_report_dict}")
            # Fallback if conversion fails
            return CriticReportDataclass(
                status=CRITIC_STATUS_FAILURE_EVALUATION,
                score=0.0,
                summary=f"Harness callback: CriticReport conversion failed. {te}",
                error_details=str(te),
                test_results=[],
                generated_code_for_report=temp_executor_generated_code
            )


    for i in range(MAX_MAIN_LOOP_ITERATIONS):
        task_run_log["total_iterations"] = i + 1
        
        iteration_header = f"[Harness] Task '{TermColors.color_text(task_id, TermColors.BLUE)}', Main Loop Iteration: {TermColors.color_text(str(i + 1), TermColors.BOLD)}/{MAX_MAIN_LOOP_ITERATIONS}"
        logger.info(iteration_header)
        
        iteration_data: Dict[str, Any] = {"main_iteration_num": i + 1}
        current_plan_for_iteration: Optional[Dict] = None # To store the plan for this iteration

        # 1. Planner Agent
        logger.info(TermColors.color_text("  1. Planner Agent:", TermColors.HEADER))
        try:
            plan = planner.run(user_request=task_description)
            current_plan_for_iteration = plan # Store for potential use by callback
            iteration_data["planner_output"] = plan
            logger.info(f"    {TermColors.color_text('Planner Output:', TermColors.GREEN)} {plan}")
            if isinstance(plan, dict) and plan.get("error"):
                error_msg = f"Planner returned an error: {plan.get('error')}. Aborting task."
                logger.error(TermColors.color_text(f"    ERROR: {error_msg}", TermColors.FAIL))
                task_run_log["final_status"] = "FAILURE_PLANNER"
                iteration_data["planner_error"] = plan.get('error')
                task_run_log["iterations_data"].append(iteration_data)
                break
        except LLMServiceError as e:
            error_msg = f"LLMServiceError during Planner execution: {e}. Aborting task."
            logger.error(TermColors.color_text(f"    ERROR: {error_msg}", TermColors.FAIL))
            task_run_log["final_status"] = "FAILURE_LLM_SERVICE_PLANNER"
            iteration_data["planner_error"] = str(e)
            task_run_log["iterations_data"].append(iteration_data)
            break
        except Exception as e:
            error_msg = f"Unexpected error during Planner execution: {e}. Aborting task."
            logger.error(TermColors.color_text(f"    ERROR: {error_msg}", TermColors.FAIL))
            task_run_log["final_status"] = "FAILURE_UNEXPECTED_PLANNER"
            iteration_data["planner_error"] = str(e)
            task_run_log["iterations_data"].append(iteration_data)
            break


        # 2. Executor Agent
        logger.info(TermColors.color_text("  2. Executor Agent:", TermColors.HEADER))
        logger.info(f"    {TermColors.color_text('Using Executor Prompt:', TermColors.CYAN)} {executor.system_prompt[:100]}...") # Log the actual prompt being used by executor
        try:
            generated_code = executor.run(plan=current_plan_for_iteration, original_request=task_description)
            iteration_data["executor_prompt_used"] = executor.system_prompt # Log the prompt that was actually used
            iteration_data["generated_code"] = generated_code
            logger.info(f"    {TermColors.color_text('Generated Code:', TermColors.GREEN)}\n{generated_code}")

            if generated_code.startswith("# Error generating code:"):
                error_msg = f"Executor returned an error string: {generated_code}"
                logger.error(TermColors.color_text(f"    ERROR: {error_msg}", TermColors.FAIL))
                task_run_log["final_status"] = "FAILURE_EXECUTOR"
                iteration_data["executor_error"] = generated_code
        except LLMServiceError as e:
            error_msg = f"LLMServiceError during Executor execution: {e}."
            logger.error(TermColors.color_text(f"    ERROR: {error_msg}", TermColors.FAIL))
            iteration_data["executor_error"] = str(e)
            if i < MAX_MAIN_LOOP_ITERATIONS - 1:
                logger.info(TermColors.color_text("    Attempting self-heal after Executor LLM error.", TermColors.WARNING))
            else:
                task_run_log["final_status"] = "FAILURE_LLM_SERVICE_EXECUTOR"
            generated_code = f"# LLM Service Error during generation: {e}" # Allow critic to see this
        except Exception as e:
            error_msg = f"Unexpected error during Executor execution: {e}. Aborting task."
            logger.error(TermColors.color_text(f"    ERROR: {error_msg}", TermColors.FAIL))
            task_run_log["final_status"] = "FAILURE_UNEXPECTED_EXECUTOR"
            iteration_data["executor_error"] = str(e)
            task_run_log["iterations_data"].append(iteration_data)
            break


        # 3. Critic Agent
        logger.info(TermColors.color_text("  3. Critic Agent:", TermColors.HEADER))
        critic_report_dict: Dict[str, Any] # To store the raw dictionary from critic
        try:
            critic_report_dict = critic.run(generated_code=generated_code, task_description=task_description, plan=current_plan_for_iteration)
            iteration_data["critic_report"] = critic_report_dict # Store the dict
            current_score = critic_report_dict.get("score", 0.0)
            logger.info(f"    {TermColors.color_text('Critic Report:', TermColors.GREEN)} "
                        f"Status: {TermColors.color_text(critic_report_dict.get('status','N/A'), TermColors.CYAN if critic_report_dict.get('status') == 'SUCCESS_EXECUTION' else TermColors.WARNING)}, "
                        f"Score: {TermColors.color_text(str(current_score), TermColors.BOLD)}, "
                        f"Summary: {critic_report_dict.get('summary', 'N/A')}")
        except LLMServiceError as e:
            error_msg = f"LLMServiceError during Critic execution (e.g. test generation): {e}."
            logger.error(TermColors.color_text(f"    ERROR: {error_msg}", TermColors.FAIL))
            critic_report_dict = {"status": "FAILURE_LLM_SERVICE_CRITIC", "score": 0.0, "summary": f"Critic LLM failed: {e}", "error_details": {"type": "LLMServiceError", "message": str(e)}, "test_results": []}
            iteration_data["critic_report"] = critic_report_dict
            current_score = 0.0
        except Exception as e:
            error_msg = f"Unexpected error during Critic execution: {e}."
            logger.error(TermColors.color_text(f"    ERROR: {error_msg}", TermColors.FAIL))
            critic_report_dict = {"status": "FAILURE_UNEXPECTED_CRITIC", "score": 0.0, "summary": f"Critic unexpected error: {e}", "error_details": {"type": "Exception", "message": str(e)}, "test_results": []}
            iteration_data["critic_report"] = critic_report_dict
            current_score = 0.0


        task_run_log["iterations_data"].append(iteration_data) # Append data for this main iteration
        task_run_log["final_code"] = generated_code
        task_run_log["final_score"] = current_score

        # 4. Check for Success or Failure & Decide to Self-Heal
        logger.info(TermColors.color_text("  4. Evaluation & Self-Healing Decision:", TermColors.HEADER))
        critic_status = critic_report_dict.get("status", "UNKNOWN_STATUS")

        if critic_status == "SUCCESS_EXECUTION" and current_score >= CRITIC_SUCCESS_THRESHOLD:
            success_msg = f"Task '{TermColors.color_text(task_id, TermColors.GREEN)}' PASSED (score {current_score}) in iteration {i + 1}."
            logger.info(TermColors.color_text(f"    {success_msg}", TermColors.GREEN))
            task_run_log["final_status"] = "SUCCESS"
            break
        elif critic_status.startswith("FAILURE_") or current_score < CRITIC_SUCCESS_THRESHOLD:
            fail_msg = f"Task '{TermColors.color_text(task_id, TermColors.FAIL)}' FAILED (score {current_score}) in iteration {i + 1}. Critic: {critic_status}"
            logger.warning(TermColors.color_text(f"    {fail_msg}", TermColors.WARNING))
            
            if i < MAX_MAIN_LOOP_ITERATIONS - 1:
                logger.info(TermColors.color_text(f"    Triggering Prompt Modifier for task '{task_id}'.", TermColors.CYAN))
                
                if prompt_modifier_for_task is None:
                    logger.info("      Instantiating PromptModifier for this task.")
                    
                    # Convert dict from critic to the dataclass PromptModifier expects
                    try:
                        critic_report_obj = CriticReportDataclass(**critic_report_dict)
                    except TypeError as te:
                        logger.error(f"    Could not convert critic_report_dict to CriticReportDataclass: {te}. Using placeholder critic report for PromptInfo.")
                        critic_report_obj = CriticReportDataclass(status="FAILURE_CONVERSION", score=0.0, summary=f"Dict to dataclass conversion failed: {te}")
                    
                    prompt_modifier_for_task = ActualPromptModifier(
                        llm_service=llm_service_instance,
                        task_id=task_id,
                        initial_prompt=executor.system_prompt,  # Pass current executor prompt as initial prompt
                        initial_score=current_score,  # Pass current score from critic
                        initial_critic_report=critic_report_obj  # Pass created CriticReportDataclass object
                    )
                
                # Prepare Failing PromptInfo
                failing_prompt_info = PromptInfo(
                    prompt=executor.system_prompt, # Use the prompt that was actually used by executor
                    score=current_score,
                    critic_report=critic_report_obj, # Pass the dataclass object
                    iteration_created=prompt_modifier_for_task.main_system_healing_attempts # reflects how many times PM has been invoked for this task
                )
                
                # Call the actual PromptModifier's run_self_healing_iteration
                evolved_prompt = prompt_modifier_for_task.run_self_healing_iteration(
                    failing_prompt_info=failing_prompt_info,
                    original_task_description=task_description,
                    orchestrator_callback_evaluate_candidate=_orchestrator_callback_evaluate_candidate
                )
                iteration_data["prompt_modifier_evo_details"] = {"population_before_run": [pinfo.__dict__ for pinfo in prompt_modifier_for_task.get_current_population()]}


                if evolved_prompt and evolved_prompt != executor.system_prompt:
                    current_executor_prompt = evolved_prompt # This variable tracks the latest prompt
                    executor.set_prompt(evolved_prompt) # Update the executor instance
                    logger.info(TermColors.color_text(f"    Executor prompt updated by PromptModifier: {evolved_prompt[:100]}...", TermColors.CYAN))
                    iteration_data["prompt_modifier_output_prompt"] = evolved_prompt
                elif evolved_prompt and evolved_prompt == executor.system_prompt:
                    logger.warning(TermColors.color_text("    PromptModifier returned the same prompt. No change to Executor prompt.", TermColors.WARNING))
                    iteration_data["prompt_modifier_output_prompt"] = evolved_prompt # Log it anyway
                else: # evolved_prompt is None
                    logger.warning(TermColors.color_text("    PromptModifier did not return an improved prompt or an error occurred. Executor prompt unchanged.", TermColors.WARNING))
                    iteration_data["prompt_modifier_output_prompt"] = None


            else: # Max iterations reached
                max_iter_fail_msg = f"Task '{TermColors.color_text(task_id, TermColors.FAIL)}' FAILED after max iterations ({MAX_MAIN_LOOP_ITERATIONS}). Final Critic: {critic_status}, Score: {current_score}."
                logger.error(TermColors.color_text(f"    {max_iter_fail_msg}", TermColors.FAIL))
                task_run_log["final_status"] = "FAILURE_MAX_ITERATIONS"
                break

        else: # Ambiguous status
            ambiguous_msg = f"Task '{TermColors.color_text(task_id, TermColors.WARNING)}' ambiguous status ('{critic_status}') but high score ({current_score}) in iteration {i + 1}. Considering SUCCESS."
            logger.info(TermColors.color_text(f"    {ambiguous_msg}", TermColors.WARNING))
            task_run_log["final_status"] = "SUCCESS_AMBIGUOUS" # Or perhaps should retry? For now, matches old logic.
            break
        logger.info("-" * 70)

    if task_run_log["final_status"] == "UNKNOWN":
        unknown_status_msg = f"Task '{TermColors.color_text(task_id, TermColors.FAIL)}' completed {MAX_MAIN_LOOP_ITERATIONS} iterations, undetermined status. Score: {task_run_log['final_score']}."
        logger.error(TermColors.color_text(unknown_status_msg, TermColors.FAIL))
        task_run_log["final_status"] = "FAILURE_MAX_ITERATIONS" # Default to failure if loop finishes without explicit success/fail

    final_summary_msg = f"--- Finished Task: {TermColors.color_text(task_id, TermColors.HEADER)} --- Status: {TermColors.color_text(task_run_log['final_status'], TermColors.GREEN if 'SUCCESS' in task_run_log['final_status'] else TermColors.FAIL)}, Score: {TermColors.color_text(str(task_run_log['final_score']), TermColors.BOLD)}"
    logger.info(final_summary_msg)
    
    # This print is for the final console summary block, distinct from logger
    print(f"\n{TermColors.color_text('Summary for Task:', TermColors.HEADER)} {TermColors.color_text(task_id, TermColors.BLUE)}")
    print(f"  {TermColors.color_text('Final Status:', TermColors.BOLD)} {TermColors.color_text(task_run_log['final_status'], TermColors.GREEN if 'SUCCESS' in task_run_log['final_status'] else TermColors.FAIL)}")
    print(f"  {TermColors.color_text('Final Score:', TermColors.BOLD)} {task_run_log['final_score']}")
    print(f"  {TermColors.color_text('Total Iterations:', TermColors.BOLD)} {task_run_log['total_iterations']}")
    print(f"  {TermColors.color_text('Final Code:', TermColors.BOLD)}\n{task_run_log['final_code']}\n")
    return task_run_log

def main_evaluation_harness():
    # --- Welcome Message ---
    print(TermColors.color_text("=" * 80, TermColors.HEADER))
    print(TermColors.color_text("SELF-HEALING AGENTIC AI - INTERACTIVE TASK RUNNER (LIVE LLM)", TermColors.HEADER + TermColors.BOLD))
    print(TermColors.color_text("=" * 80, TermColors.HEADER))
    print(TermColors.color_text("Using default LLM: Deepseek (deepseek-chat)", TermColors.CYAN))
    print(TermColors.color_text("Using default initial Executor prompt.", TermColors.CYAN))
    print(TermColors.color_text("API key for Deepseek (DEEPSEEK_API_KEY) must be set in environment variables.", TermColors.WARNING))
    print(TermColors.color_text(f"Logging detailed output to: {LOG_FILE_NAME}", TermColors.CYAN))
    print(TermColors.color_text("-" * 80, TermColors.HEADER))

    llm_service_instance = None
    try:
        llm_provider = "deepseek" # Consider making configurable
        llm_model_name = "deepseek-chat" # Consider making configurable
        llm_service_instance = LLMService(provider=llm_provider, model_name=llm_model_name)
        logger.info(f"LLMService initialized for provider: {llm_provider}, model: {llm_model_name}")
        logger.info("Attempting a simple test LLM call to verify setup...")
        test_response = llm_service_instance.invoke([{"role": "user", "content": "Hello!"}])
        logger.info(f"Test LLM call successful. Response snippet: {test_response[:50]}...")
        print(TermColors.color_text("LLM Service connected successfully.", TermColors.GREEN))
    except LLMServiceError as e:
        error_msg = f"Failed to initialize or test LLMService: {e}"
        logger.error(error_msg) 
        print(TermColors.color_text(f"ERROR: {error_msg}", TermColors.FAIL))
        return
    except Exception as e:
        error_msg = f"An unexpected error occurred during LLM setup: {e}"
        logger.error(error_msg)
        print(TermColors.color_text(f"ERROR: {error_msg}", TermColors.FAIL))
        return
    
    print(TermColors.color_text("-" * 80, TermColors.HEADER))
    
    planner = Planner(name="HarnessPlanner", llm_service=llm_service_instance)
    executor = Executor(name="HarnessExecutor", llm_service=llm_service_instance, system_prompt=DEFAULT_EXECUTOR_SYSTEM_PROMPT)
    critic = Critic(name="HarnessCritic", llm_service=llm_service_instance)
    # PromptModifier is now instantiated per task inside run_single_task if needed

    # --- Get Custom Task Input ---
    task_description = input(f"{TermColors.color_text('Enter Task Description', TermColors.BOLD)} (e.g., 'Write a Python function to add two numbers'): \n> ").strip()
    
    task_id_safe_part = "".join(filter(str.isalnum, task_description.lower().replace(" ", "_")[:30]))
    if not task_id_safe_part: task_id_safe_part = "custom" # Fallback if description is all special chars
    task_id = f"interactive_task_{task_id_safe_part}_{datetime.datetime.now().strftime('%H%M%S')}"
    
    print(TermColors.color_text("-" * 80, TermColors.HEADER))

    if not task_description:
        logger.error("Task description cannot be empty. Exiting.")
        print(TermColors.color_text("ERROR: Task description is required. Please try again.", TermColors.FAIL))
        return

    custom_task_def = {
        "id": task_id,
        "description": task_description,
        "initial_executor_prompt": DEFAULT_EXECUTOR_SYSTEM_PROMPT # This is the base prompt
    }
    logger.info(f"Preparing interactive task: {custom_task_def['id']}")
    logger.info(f"Task Description: {custom_task_def['description']}")
    logger.info(f"Initial Executor Prompt: {custom_task_def['initial_executor_prompt']}")

    print(f"\n{TermColors.color_text('Running task:', TermColors.BOLD)} '{TermColors.color_text(task_description, TermColors.CYAN)}' "
          f"with initial prompt: '{TermColors.color_text(custom_task_def['initial_executor_prompt'][:50] + '...', TermColors.CYAN)}'\n")
    
    result = run_single_task(
        custom_task_def,
        planner,
        executor,
        critic,
        llm_service_instance # Pass the llm_service to run_single_task
    )

    logger.info("=== Interactive Task Execution Finished ===")
    print(f"\n{TermColors.color_text('=' * 30 + ' INTERACTIVE TASK SUMMARY ' + '=' * 30, TermColors.HEADER + TermColors.BOLD)}")
    if result:
        print(f"  {TermColors.color_text('Task ID          :', TermColors.BOLD)} {result.get('task_id')}")
        print(f"  {TermColors.color_text('Task Description :', TermColors.BOLD)} {result.get('description')}")
        print(f"  {TermColors.color_text('Initial Prompt   :', TermColors.BOLD)} {result.get('initial_prompt')}")
        final_status = result.get('final_status')
        status_color = TermColors.GREEN if "SUCCESS" in final_status else TermColors.FAIL
        print(f"  {TermColors.color_text('Final Status     :', TermColors.BOLD)} {TermColors.color_text(final_status, status_color)}")
        print(f"  {TermColors.color_text('Final Score      :', TermColors.BOLD)} {result.get('final_score')}")
        print(f"  {TermColors.color_text('Total Iterations :', TermColors.BOLD)} {result.get('total_iterations')}")
        print(f"  {TermColors.color_text('Final Code Generated:', TermColors.BOLD)}\n{result.get('final_code')}")
        
        print(f"\n  {TermColors.color_text('Iteration Details:', TermColors.BOLD)}")
        for i, iter_data in enumerate(result.get("iterations_data", [])):
            print(f"    {TermColors.color_text(f'Iteration {i+1}:', TermColors.UNDERLINE)}")
            print(f"      {TermColors.color_text('Executor Prompt:', TermColors.CYAN)} {iter_data.get('executor_prompt_used')}")
            critic_rep = iter_data.get('critic_report', {})
            crit_status_color = TermColors.GREEN if critic_rep.get('status') == 'SUCCESS_EXECUTION' else TermColors.WARNING
            print(f"      {TermColors.color_text('Critic:', TermColors.CYAN)} "
                  f"Status: {TermColors.color_text(critic_rep.get('status'), crit_status_color)}, "
                  f"Score: {TermColors.color_text(str(critic_rep.get('score')), TermColors.BOLD)}, "
                  f"Summary: {critic_rep.get('summary')}")
            
            generated_specs = critic_rep.get('generated_test_specifications')
            if generated_specs:
                print(f"      {TermColors.color_text('Generated Test Specifications:', TermColors.CYAN + TermColors.UNDERLINE)}")
                for idx, spec in enumerate(generated_specs):
                    print(f"        {TermColors.color_text(f'Test Spec {idx + 1}:', TermColors.CYAN)}")
                    print(f"          Name: {spec.get('test_case_name')}")
                    print(f"          Inputs: {spec.get('inputs')}")
                    print(f"          Expected Output: {spec.get('expected_output')}")

            if iter_data.get('prompt_modifier_output_prompt'):
                print(f"      {TermColors.color_text('Prompt Modifier Output:', TermColors.CYAN)} {iter_data.get('prompt_modifier_output_prompt')}")
            if iter_data.get('prompt_modifier_evo_details'):
                 print(f"      {TermColors.color_text('Prompt Modifier Evo Details:', TermColors.CYAN)} {iter_data.get('prompt_modifier_evo_details')}")
    else:
        print(TermColors.color_text("  Interactive task did not produce a result structure.", TermColors.FAIL))
    print(TermColors.color_text("=" * (60 + len(" INTERACTIVE TASK SUMMARY ")) + "\n", TermColors.HEADER + TermColors.BOLD))
    
    # Append prompt evolution summary to log file
    if result:
        _append_prompt_evolution_summary_to_log(result, LOG_FILE_NAME)

    print(TermColors.color_text(f"Full logs available in: {LOG_FILE_NAME}", TermColors.CYAN))

def _append_prompt_evolution_summary_to_log(task_result: Dict[str, Any], log_file_path: str):
    """Appends a summary of prompt evolution to the specified log file."""
    summary_lines = ["\n\n", "="*30 + " PROMPT EVOLUTION SUMMARY " + "="*30 + "\n"]
    
    task_id = task_result.get("task_id", "N/A")
    initial_prompt = task_result.get("initial_prompt", "N/A")
    
    summary_lines.append(f"Task ID: {task_id}\n")
    summary_lines.append(f"Initial Executor Prompt: {initial_prompt}\n")
    summary_lines.append("-" * 70 + "\n")

    iterations_data = task_result.get("iterations_data", [])
    if not iterations_data:
        summary_lines.append("No iteration data found for prompt evolution summary.\n")
    
    for i, iter_data in enumerate(iterations_data):
        summary_lines.append(f"Main Loop Iteration {iter_data.get('main_iteration_num', i + 1)}:\n")
        
        executor_prompt_used = iter_data.get("executor_prompt_used", "N/A")
        summary_lines.append(f"  Executor Prompt Used:\n{executor_prompt_used}\n") # Log full prompt
        
        generated_code_this_iter = iter_data.get("generated_code", "N/A")
        summary_lines.append(f"  Executor Generated Code (for above prompt):\n{generated_code_this_iter}\n") # Log full code for this iteration

        critic_report = iter_data.get("critic_report", {})
        critic_score = critic_report.get("score", "N/A")
        critic_status = critic_report.get("status", "N/A")
        summary_lines.append(f"  Critic Score: {critic_score}, Status: {critic_status}\n")
        
        # Add detailed test case information
        summary_lines.append(f"  Critic Test Details:\n")
        test_results = critic_report.get("test_results", [])
        if test_results:
            summary_lines.append(f"    Number of Test Cases: {len(test_results)}\n")
            for idx, test_result in enumerate(test_results):
                summary_lines.append(f"    Test Case {idx+1}:\n")
                summary_lines.append(f"      Name: {test_result.get('test_case_name', 'N/A')}\n")
                summary_lines.append(f"      Status: {test_result.get('status', 'N/A')}\n")
                summary_lines.append(f"      Inputs: {test_result.get('inputs', {})}\n")
                summary_lines.append(f"      Expected Output: {test_result.get('expected_output', 'N/A')}\n")
                summary_lines.append(f"      Actual Output: {test_result.get('actual_output', 'N/A')}\n")
                summary_lines.append(f"      Error Message: {test_result.get('error_message', 'N/A')}\n")
        else:
            summary_lines.append(f"    No test results available.\n")
        
        # Add test specifications
        test_specs = critic_report.get("generated_test_specifications", [])
        if test_specs:
            summary_lines.append(f"    Generated Test Specifications:\n")
            for idx, spec in enumerate(test_specs):
                summary_lines.append(f"      Specification {idx+1}:\n")
                summary_lines.append(f"        Name: {spec.get('test_case_name', 'N/A')}\n")
                summary_lines.append(f"        Inputs: {spec.get('inputs', {})}\n")
                summary_lines.append(f"        Expected Output: {spec.get('expected_output', 'N/A')}\n")
        
        # Include function_to_test information
        function_to_test = critic_report.get("function_to_test", "N/A")
        summary_lines.append(f"    Function Being Tested: {function_to_test}\n")
        
        # Include critic summary
        critic_summary = critic_report.get("summary", "N/A")
        summary_lines.append(f"    Critic Summary: {critic_summary}\n")

        pm_output_prompt = iter_data.get("prompt_modifier_output_prompt")
        if pm_output_prompt is not None:
            summary_lines.append(f"  Prompt Modifier - Evolved Prompt Suggestion:\n{pm_output_prompt}\n") # Log full suggested prompt
        
        pm_evo_details = iter_data.get("prompt_modifier_evo_details")
        if pm_evo_details:
            summary_lines.append("  Prompt Modifier - Evolution Details:\n")
            population_before = pm_evo_details.get("population_before_run", [])
            if population_before:
                summary_lines.append("    Population before this healing iteration (prompts and scores):\n")
                for idx, p_info_dict in enumerate(population_before):
                    prompt_text = p_info_dict.get('prompt', 'N/A')
                    score = p_info_dict.get('score', 'N/A')
                    iteration_created = p_info_dict.get('iteration_created', 'N/A')
                    # Log full prompt from population, ensure score is formatted if it's a float
                    score_display = f"{score:.3f}" if isinstance(score, float) else score
                    summary_lines.append(f"      - Prompt (created iter {iteration_created}): Score {score_display} | \"{prompt_text}\"\n") # Log full prompt text
            else:
                summary_lines.append("    No population details recorded for this iteration.\n")
        
        summary_lines.append("-" * 70 + "\n")

    summary_lines.append("="*30 + " END PROMPT EVOLUTION SUMMARY " + "="*30 + "\n")
    
    try:
        with open(log_file_path, 'a') as f:
            f.write("".join(summary_lines))
        logger.info(f"Prompt evolution summary appended to {log_file_path}")
    except IOError as e:
        logger.error(f"Failed to append prompt evolution summary to log file {log_file_path}: {e}")


if __name__ == "__main__":
    main_evaluation_harness() 