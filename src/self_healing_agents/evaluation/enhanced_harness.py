"""
Enhanced Evaluation Harness for Self-Healing Agents with Direct Fix Capability

This module extends the standard evaluation harness with direct error-fixing
capabilities before engaging the full Self-Healing Module.
"""

import sys
import os
# Ensure the src directory is in the Python path for relative imports to work correctly
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(os.path.dirname(current_dir)) 
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import logging
import re
from typing import Any, Dict, List, Tuple, Optional
import datetime

# Agent and service imports
from self_healing_agents.agents import Planner, Executor, Critic
from self_healing_agents.llm_service import LLMService, LLMServiceError
from self_healing_agents.prompts import DEFAULT_EXECUTOR_SYSTEM_PROMPT, ULTRA_BUGGY_PROMPT
from self_healing_agents.schemas import CriticReport as CriticReportDataclass, CRITIC_STATUS_FAILURE_EVALUATION
from self_healing_agents.schemas import CRITIC_STATUS_SUCCESS

# --- Console Coloring Utility ---
class TermColors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    YELLOW = '\033[93m'  # Same as WARNING
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    @staticmethod
    def color_text(text: str, color: str) -> str:
        return f"{color}{text}{TermColors.ENDC}"

# --- Logging Setup ---
LOG_FILE_NAME = "enhanced_evaluation_harness.log"

# Function to strip ANSI escape codes
def strip_ansi_codes(text: str) -> str:
    return re.sub(r'\\x1b\\[[0-9;]*[mK]', '', text)

# Custom filter to strip ANSI codes
class StripAnsiFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = strip_ansi_codes(record.getMessage())
        record.args = () # Clear args as msg is now pre-formatted
        return True

# Remove existing handlers to prevent duplicate logs
root_logger = logging.getLogger()
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

# File Handler
file_handler = logging.FileHandler(LOG_FILE_NAME, mode='w')
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
file_handler.setLevel(logging.INFO)
file_handler.addFilter(StripAnsiFilter())
logging.getLogger().addHandler(file_handler)

# --- Configuration ---
MAX_MAIN_LOOP_ITERATIONS = 3
CRITIC_SUCCESS_THRESHOLD = 0.9

# --- Enhanced Evaluation Harness Logic ---

def run_single_task(
    task_definition: Dict[str, Any],
    planner: Planner,
    executor: Executor,
    critic: Critic,
    llm_service_instance: LLMService
) -> Dict[str, Any]:
    """
    Run a single task with enhanced evaluation that includes direct fix attempts
    before engaging the Self-Healing Module.
    
    Args:
        task_definition: Dictionary containing task information
        planner: Planner agent instance
        executor: Executor agent instance
        critic: Critic agent instance
        llm_service_instance: LLM service for PromptModifier
        
    Returns:
        Dictionary containing task results
    """
    task_id = task_definition["id"]
    task_description = task_definition["description"]
    initial_prompt = task_definition["initial_executor_prompt"]

    logger.info(f"--- Starting Task: {TermColors.color_text(task_id, TermColors.HEADER)} ---")
    logger.info(f"Description: {TermColors.color_text(task_description, TermColors.CYAN)}")
    
    # Log the initial prompt being set
    logger.info(f"🔧 SETTING INITIAL EXECUTOR PROMPT: {initial_prompt[:100]}...")
    executor.set_prompt(initial_prompt)
    logger.info(f"✅ EXECUTOR PROMPT SET TO: {executor.system_prompt[:100]}...")
    
    prompt_modifier_for_task = None  # Removed ActualPromptModifier dependency

    current_executor_prompt = initial_prompt
    task_run_log: Dict[str, Any] = {
        "task_id": task_id,
        "description": task_description,
        "initial_prompt": initial_prompt,
        "iterations_data": [],
        "final_status": "UNKNOWN",
        "final_code": None,
        "final_score": 0.0,
        "total_iterations": 0,
        "direct_fix_attempts": 0,
        "direct_fix_successes": 0
    }

    # Define the orchestrator callback function for evaluating candidate prompts
    def _orchestrator_callback_evaluate_candidate(candidate_prompt_text: str, current_task_description: str) -> CriticReportDataclass:
        logger.info(f"    [Harness Callback] Evaluating candidate prompt via Executor & Critic: \"{candidate_prompt_text[:70]}...\"")
        temp_executor_generated_code = ""
        temp_critic_report_dict: Dict[str, Any]
        try:
            # 1. Set new prompt for Executor (temporarily for this candidate)
            original_executor_prompt_before_candidate_eval = executor.system_prompt
            executor.set_prompt(candidate_prompt_text)

            # 2. Run Executor with the last plan from the main loop
            last_plan = task_run_log["iterations_data"][-1]["planner_output"] if task_run_log["iterations_data"] and "planner_output" in task_run_log["iterations_data"][-1] else {"plan_steps": ["No previous plan available for candidate evaluation."]}
            
            logger.info(f"      [Harness Callback] Running Executor with candidate prompt.")
            temp_executor_generated_code = executor.run(plan=last_plan, original_request=current_task_description)
            logger.info(f"      [Harness Callback] Executor generated code for candidate.")

            # 3. Run Critic
            logger.info(f"      [Harness Callback] Running Critic on new code.")
            temp_critic_report_dict = critic.run(generated_code=temp_executor_generated_code, task_description=current_task_description, plan=last_plan)
            
            # Restore original executor prompt after evaluation
            executor.set_prompt(original_executor_prompt_before_candidate_eval)

        except Exception as e:
            logger.error(f"      [Harness Callback] EXCEPTION during candidate evaluation: {e}")
            # Restore original executor prompt in case of error
            if 'original_executor_prompt_before_candidate_eval' in locals():
                 executor.set_prompt(original_executor_prompt_before_candidate_eval)
            
            temp_critic_report_dict = {
                "status": CRITIC_STATUS_FAILURE_EVALUATION,
                "score": 0.0,
                "summary": f"Harness callback failed to evaluate candidate prompt: {e}",
                "error_details": str(e),
                "test_results": [],
                "generated_code_for_report": temp_executor_generated_code
            }
        
        # Convert dict to CriticReportDataclass
        try:
            return CriticReportDataclass(**temp_critic_report_dict)
        except TypeError as te:
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


    # ----- PHASE 1: Initial Run with Direct Fix -----
    logger.info(f"\n=== PHASE 1: Initial Run with Direct Fix for Task '{TermColors.color_text(task_id, TermColors.BLUE)}' ===")
    
    # Start with iteration count at 0 (will be incremented in self-healing phase if needed)
    task_run_log["total_iterations"] = 0
    
    # Create data structure for initial direct fix attempt
    initial_run_data: Dict[str, Any] = {
        "phase": "DIRECT_FIX",
        "direct_fix_attempted": False,
        "direct_fix_successful": False
    }
    
    # Variables to track the best solution throughout all phases
    best_code = None
    best_score = 0.0
    best_source = "UNKNOWN"
    
    # Flag to determine if we need self-healing
    need_self_healing = True
    
    # ----- STEP 1: Planning -----
    logger.info(TermColors.color_text("  1. Planning:", TermColors.HEADER))
    try:
        plan = planner.run(user_request=task_description)
        initial_run_data["planner_output"] = plan
        logger.info(f"    {TermColors.color_text('Planner Output:', TermColors.GREEN)}\n{plan}")
        
        if isinstance(plan, dict) and plan.get("error"):
            error_msg = f"Planner returned an error: {plan.get('error')}. Aborting task."
            logger.error(TermColors.color_text(f"    ERROR: {error_msg}", TermColors.FAIL))
            task_run_log["final_status"] = "FAILURE_PLANNER"
            initial_run_data["planner_error"] = plan.get('error')
            task_run_log["iterations_data"].append(initial_run_data)
            return task_run_log
    except LLMServiceError as e:
        error_msg = f"LLMServiceError during Planner execution: {e}. Aborting task."
        logger.error(TermColors.color_text(f"    ERROR: {error_msg}", TermColors.FAIL))
        task_run_log["final_status"] = "FAILURE_LLM_SERVICE"
        initial_run_data["planner_error"] = str(e)
        task_run_log["iterations_data"].append(initial_run_data)
        return task_run_log
        
    # ----- STEP 2: Execution -----
    logger.info(TermColors.color_text("  2. Execution:", TermColors.HEADER))
    try:
        initial_code = executor.run(plan=plan, original_request=task_description)
        initial_run_data["executor_initial_code"] = initial_code
        logger.info(f"    {TermColors.color_text('Executor Output (initial code):', TermColors.GREEN)}\n{initial_code}")
        
        if isinstance(initial_code, dict) and initial_code.get("error"):
            error_msg = f"Executor returned an error: {initial_code.get('error')}. Aborting task."
            logger.error(TermColors.color_text(f"    ERROR: {error_msg}", TermColors.FAIL))
            task_run_log["final_status"] = "FAILURE_EXECUTOR"
            initial_run_data["executor_error"] = initial_code.get('error')
            task_run_log["iterations_data"].append(initial_run_data)
            return task_run_log
    except LLMServiceError as e:
        error_msg = f"LLMServiceError during Executor execution: {e}. Aborting task."
        logger.error(TermColors.color_text(f"    ERROR: {error_msg}", TermColors.FAIL))
        task_run_log["final_status"] = "FAILURE_LLM_SERVICE"
        initial_run_data["executor_error"] = str(e)
        task_run_log["iterations_data"].append(initial_run_data)
        return task_run_log

    # ----- STEP 3: Initial Evaluation -----
    logger.info(TermColors.color_text("  3. Initial Evaluation:", TermColors.HEADER))
    try:
        critique_result = critic.run(generated_code=initial_code, task_description=task_description, plan=plan)
        initial_run_data["critic_output"] = critique_result
        
        # Diagnostic logging of critique structure
        logger.info(f"    Critique Keys: {critique_result.keys() if isinstance(critique_result, dict) else 'Not a dict'}")
        
        # Handle field name variations (either 'status' or 'overall_status')
        if isinstance(critique_result, dict):
            if 'overall_status' in critique_result:
                overall_status = critique_result['overall_status']
            else:
                overall_status = critique_result.get('status', 'UNKNOWN_ERROR')
                
            # Handle field name variations (either 'quantitative_score' or 'score')
            if 'quantitative_score' in critique_result:
                score = critique_result['quantitative_score']
            else:
                score = critique_result.get('score', 0.0)
                
            initial_run_data["critic_status"] = overall_status
            initial_run_data["critic_score"] = score
            
            logger.info(f"    {TermColors.color_text('Full Initial Critique:', TermColors.CYAN)}\n{critique_result}")

            # Update best code tracking
            best_code = initial_code
            best_score = score
            best_source = "INITIAL"
            
            logger.info(f"    {TermColors.color_text('Critic Status:', TermColors.GREEN)} {overall_status}")
            logger.info(f"    {TermColors.color_text('Critic Score:', TermColors.GREEN)} {score}")
            
            # If code is already successful, no need for direct fix or self-healing
            if overall_status == CRITIC_STATUS_SUCCESS or score >= CRITIC_SUCCESS_THRESHOLD:
                logger.info(TermColors.color_text("  Initial code passed evaluation! No need for direct fix or self-healing.", TermColors.GREEN))
                task_run_log["final_status"] = "SUCCESS"
                task_run_log["final_code"] = initial_code
                task_run_log["final_score"] = score
                task_run_log["final_source"] = "INITIAL"
                need_self_healing = False
                task_run_log["iterations_data"].append(initial_run_data)
                
                # Return early to avoid unnecessary self-healing
                return task_run_log
            else:
                # Initial code has issues, proceed to direct fix attempt
                logger.info(TermColors.color_text("  4. Direct Fix Attempt:", TermColors.HEADER))
                
                # Track direct fix attempt
                task_run_log["direct_fix_attempts"] = 1
                initial_run_data["direct_fix_attempted"] = True
                
                try:
                    # Call the direct_fix_attempt method on the executor
                    logger.info("    Attempting direct fix of code based on error report...")
                    fixed_code = executor.direct_fix_attempt(
                        original_code=initial_code,
                        error_report=critique_result,
                        task_description=task_description,
                        plan=plan
                    )
                    initial_run_data["direct_fix_code"] = fixed_code
                    logger.info(f"    {TermColors.color_text('Direct Fix Output:', TermColors.GREEN)}\n{fixed_code}")
                    
                    # Re-evaluate the directly fixed code
                    logger.info("    Re-evaluating directly fixed code...")
                    direct_fix_critique = critic.run(
                        generated_code=fixed_code, 
                        task_description=task_description, 
                        plan=plan
                    )
                    initial_run_data["direct_fix_critique"] = direct_fix_critique
                    
                    logger.info(f"    {TermColors.color_text('Full Direct Fix Critique:', TermColors.CYAN)}\n{direct_fix_critique}")

                    # Handle field name variations in direct fix critique
                    if isinstance(direct_fix_critique, dict):
                        if 'overall_status' in direct_fix_critique:
                            direct_fix_status = direct_fix_critique['overall_status']
                        else:
                            direct_fix_status = direct_fix_critique.get('status', 'UNKNOWN_ERROR')
                            
                        if 'quantitative_score' in direct_fix_critique:
                            direct_fix_score = direct_fix_critique['quantitative_score']
                        else:
                            direct_fix_score = direct_fix_critique.get('score', 0.0)
                            
                        initial_run_data["direct_fix_status"] = direct_fix_status
                        initial_run_data["direct_fix_score"] = direct_fix_score
                        
                        logger.info(f"    {TermColors.color_text('Direct Fix Status:', TermColors.GREEN)} {direct_fix_status}")
                        logger.info(f"    {TermColors.color_text('Direct Fix Score:', TermColors.GREEN)} {direct_fix_score}")
                        
                        # If direct fix was successful, we can skip self-healing
                        if direct_fix_status == CRITIC_STATUS_SUCCESS or direct_fix_score >= CRITIC_SUCCESS_THRESHOLD:
                            logger.info(TermColors.color_text("  Direct fix successful! No need for self-healing.", TermColors.GREEN))
                            task_run_log["direct_fix_successes"] = 1
                            initial_run_data["direct_fix_successful"] = True
                            task_run_log["final_status"] = "SUCCESS_DIRECT_FIX"
                            task_run_log["final_code"] = fixed_code
                            task_run_log["final_score"] = direct_fix_score
                            task_run_log["final_source"] = "DIRECT_FIX"
                            task_run_log["iterations_data"].append(initial_run_data)
                            need_self_healing = False
                            
                        # Check if direct fix improved the code but didn't fully fix it
                        if direct_fix_score > score:
                            logger.info(TermColors.color_text("  Direct fix improved the code but not enough for success.", TermColors.YELLOW))
                            best_code = fixed_code
                            best_score = direct_fix_score
                            best_source = "DIRECT_FIX"
                    else:
                        error_msg = "Critic returned invalid critique for direct fix. Continuing with the best code so far."
                        logger.error(TermColors.color_text(f"    ERROR: {error_msg}", TermColors.FAIL))
                        initial_run_data["direct_fix_error"] = "Invalid critique format"
                except LLMServiceError as e:
                    error_msg = f"LLMServiceError during direct fix: {e}. Continuing with best code so far."
                    logger.error(TermColors.color_text(f"    ERROR: {error_msg}", TermColors.FAIL))
                    initial_run_data["direct_fix_error"] = str(e)
                
                # Complete the initial run phase by adding to the log
                task_run_log["iterations_data"].append(initial_run_data)
                
        # ----- PHASE 2: Self-Healing Iterations (if needed) -----
        if need_self_healing:
            logger.info(f"\n=== PHASE 2: Self-Healing Iterations for Task '{TermColors.color_text(task_id, TermColors.BLUE)}' ===")
        
            # Initialize PromptModifier for self-healing phase
            if prompt_modifier_for_task is None:
                logger.info("    Skipping PromptModifier initialization (dependency removed)")
                # Simplified self-healing without PromptModifier
                task_run_log["final_status"] = "COMPLETED_NO_SELF_HEALING"
                task_run_log["final_code"] = best_code
                task_run_log["final_score"] = best_score
                task_run_log["final_source"] = best_source
                return task_run_log
                    
            # Max number of self-healing iterations to perform
            MAX_SELF_HEALING_ITERATIONS = 3
            
            # Start self-healing iterations
            for iteration_num in range(1, MAX_SELF_HEALING_ITERATIONS + 1):
                # Update the total iterations count
                task_run_log["total_iterations"] += 1
                
                # Create iteration data structure
                iteration_data = {
                    "phase": "SELF_HEALING",
                    "iteration_num": iteration_num,
                    "best_code_source_before_iteration": best_source
                }
                
                logger.info(f"\n----- Self-Healing Iteration {iteration_num}/{MAX_SELF_HEALING_ITERATIONS} -----")
                
                # 1. Generate optimized prompt based on critique feedback
                logger.info(TermColors.color_text("  1. Generating optimized prompt based on critique feedback:", TermColors.HEADER))
                
                try:
                    # Use the appropriate feedback and code based on best source so far
                    critique_to_use = None
                    code_to_use = None
                    
                    if best_source == "DIRECT_FIX":
                        # We'll use the critique from the direct fix as our starting point
                        critique_to_use = initial_run_data.get("direct_fix_critique")
                        code_to_use = initial_run_data.get("direct_fix_code")
                    else:  # best_source == "INITIAL"
                        # We'll use the critique from the initial run
                        critique_to_use = initial_run_data.get("critic_output")
                        code_to_use = initial_run_data.get("executor_initial_code")
                    
                    # Generate optimized prompt
                    optimized_prompt = prompt_modifier_for_task.run(
                        initial_prompt=current_executor_prompt,
                        feedback=critique_to_use,
                        code=code_to_use,
                        task_description=task_description
                    )
                    
                    # Update executor with optimized prompt
                    logger.info("    Updating executor with optimized prompt...")
                    executor.set_prompt(optimized_prompt)
                    current_executor_prompt = optimized_prompt
                    iteration_data["optimized_prompt"] = optimized_prompt
                    logger.info(f"    {TermColors.color_text(f'Optimized Prompt for Iteration {iteration_num}:', TermColors.CYAN)}\n{optimized_prompt}")
                    
                    # 2. Regenerate code with the optimized prompt
                    logger.info(TermColors.color_text("  2. Regenerating code with optimized prompt:", TermColors.HEADER))
                    try:
                        optimized_code = executor.run(plan=plan, original_request=task_description)
                        iteration_data["optimized_code"] = optimized_code
                        logger.info(f"    {TermColors.color_text(f'Optimized Code Output (Iteration {iteration_num}):', TermColors.GREEN)}\n{optimized_code}")
                        
                        if isinstance(optimized_code, dict) and optimized_code.get("error"):
                            error_msg = f"Executor returned an error during self-healing: {optimized_code.get('error')}. Using best code so far."
                            logger.error(TermColors.color_text(f"    ERROR: {error_msg}", TermColors.FAIL))
                            iteration_data["executor_error"] = optimized_code.get('error')
                            continue  # Skip to next iteration
                    except LLMServiceError as e:
                        error_msg = f"LLMServiceError during executor run in self-healing: {e}. Using best code so far."
                        logger.error(TermColors.color_text(f"    ERROR: {error_msg}", TermColors.FAIL))
                        iteration_data["executor_error"] = str(e)
                        continue  # Skip to next iteration
                    
                    # 3. Evaluate the optimized code
                    logger.info(TermColors.color_text("  3. Evaluating optimized code:", TermColors.HEADER))
                    try:
                        optimized_critique = critic.run(
                            generated_code=optimized_code,
                            task_description=task_description,
                            plan=plan
                        )
                        iteration_data["optimized_critique"] = optimized_critique
                        
                        logger.info(f"    {TermColors.color_text(f'Full Optimized Critique (Iteration {iteration_num}):', TermColors.CYAN)}\n{optimized_critique}")

                        # Extract status and score from the optimized critique
                        if isinstance(optimized_critique, dict):
                            if 'overall_status' in optimized_critique:
                                optimized_status = optimized_critique['overall_status']
                            else:
                                optimized_status = optimized_critique.get('status', 'UNKNOWN_ERROR')
                                
                            if 'quantitative_score' in optimized_critique:
                                optimized_score = optimized_critique['quantitative_score']
                            else:
                                optimized_score = optimized_critique.get('score', 0.0)
                                        
                            iteration_data["optimized_status"] = optimized_status
                            iteration_data["optimized_score"] = optimized_score
                            
                            logger.info(f"    {TermColors.color_text('Optimized Code Status:', TermColors.GREEN)} {optimized_status}")
                            logger.info(f"    {TermColors.color_text('Optimized Code Score:', TermColors.GREEN)} {optimized_score}")
                            # Update best code tracking if this is better
                            if optimized_score > best_score:
                                logger.info(TermColors.color_text(f"  Found better solution in iteration {iteration_num}!", TermColors.GREEN))
                                best_code = optimized_code
                                best_score = optimized_score
                                best_source = f"SELF_HEALING_ITERATION_{iteration_num}"
                            
                            # Check if we've reached success criteria
                            if optimized_status == CRITIC_STATUS_SUCCESS or optimized_score >= CRITIC_SUCCESS_THRESHOLD:
                                logger.info(TermColors.color_text(f"  Self-healing iteration {iteration_num} successful! No need for more iterations.", TermColors.GREEN))
                                task_run_log["final_status"] = "SUCCESS_SELF_HEALING"
                                task_run_log["final_code"] = optimized_code
                                task_run_log["final_score"] = optimized_score
                                task_run_log["final_source"] = f"SELF_HEALING_ITERATION_{iteration_num}"
                                task_run_log["successful_iteration"] = iteration_num
                                break
                        else:
                            error_msg = "Critic returned invalid critique for optimized code. Continuing with the best code so far."
                            logger.error(TermColors.color_text(f"    ERROR: {error_msg}", TermColors.FAIL))
                            iteration_data["critic_error"] = "Invalid critique format"
                    except LLMServiceError as e:
                        error_msg = f"LLMServiceError during critic evaluation in self-healing: {e}. Using best code so far."
                        logger.error(TermColors.color_text(f"    ERROR: {error_msg}", TermColors.FAIL))
                        iteration_data["critic_error"] = str(e)
                    
                    # Always add iteration data to the log
                    task_run_log["iterations_data"].append(iteration_data)
                    
                except Exception as e_iteration_generic: 
                    logger.error(f"Unhandled exception in self-healing iteration {iteration_num}: {e_iteration_generic}")
                    if 'iteration_data' in locals() and isinstance(iteration_data, dict): 
                        iteration_data["iteration_error"] = f"Unhandled: {str(e_iteration_generic)}"
                        # Avoid appending if L506 already added this iteration_data instance
                        if not task_run_log["iterations_data"] or task_run_log["iterations_data"][-1] != iteration_data:
                            task_run_log["iterations_data"].append(iteration_data)
                    continue # Move to the next self-healing iteration or consider breaking

    except LLMServiceError as e: 
        error_msg = f"LLMServiceError during task execution: {e}. Aborting task."
        logger.error(TermColors.color_text(f"    ERROR: {error_msg}", TermColors.FAIL))
        task_run_log["final_status"] = "FAILURE_LLM_SERVICE"
        if 'iteration_data' in locals() and isinstance(iteration_data, dict):
            iteration_data["task_error_llm"] = str(e) 
        elif 'initial_run_data' in locals() and isinstance(initial_run_data, dict):
             initial_run_data["task_error_llm"] = str(e)
        else:
            task_run_log["task_error_llm"] = str(e) 

    except Exception as e_task_generic: 
        logger.error(TermColors.color_text(f"    ERROR: Unhandled generic exception during task execution: {e_task_generic}", TermColors.FAIL))
        task_run_log["final_status"] = "FAILURE_UNHANDLED_EXCEPTION"
        task_run_log["error_details"] = str(e_task_generic)

    finally: 
        # Ensure essential fields are present in the log, even after errors.
        # Fallback to initial_code/initial_score if best_code/best_score were never set.
        
        # Get current best values, or None if not set
        current_best_code = locals().get('best_code')
        current_best_score = locals().get('best_score')
        current_best_source = locals().get('best_source')

        # Determine ultimate fallbacks (assuming initial_code and initial_score are in scope)
        ultimate_fallback_code = locals().get('initial_code', "ERROR: CODE UNAVAILABLE")
        ultimate_fallback_score = locals().get('initial_score', 0.0)
        ultimate_fallback_source = "INITIAL_AS_FALLBACK"
        if current_best_code is None and ultimate_fallback_code == "ERROR: CODE UNAVAILABLE":
            ultimate_fallback_source = "ERROR_NO_CODE_AVAILABLE"

        # Set final log values if not already set by a success/specific failure path
        if task_run_log.get("final_code") is None:
            task_run_log["final_code"] = current_best_code if current_best_code is not None else ultimate_fallback_code
        
        if task_run_log.get("final_score") is None: # Score could be 0.0 legitimately
            task_run_log["final_score"] = current_best_score if current_best_score is not None else ultimate_fallback_score

        if task_run_log.get("final_source") is None:
            task_run_log["final_source"] = current_best_source if current_best_source is not None else ultimate_fallback_source
        
        if task_run_log.get("final_status") is None:
            if task_run_log.get("error_details") or task_run_log.get("task_error_llm") or "FAILURE" in str(task_run_log.get("final_status")):
                task_run_log["final_status"] = "COMPLETED_WITH_ERRORS"
            elif current_best_code is not None: # If we have some code, but not explicit success/failure
                task_run_log["final_status"] = "COMPLETED_MAX_ITERATIONS_OR_NO_IMPROVEMENT"
            else: # Should ideally not happen if fallbacks are good
                task_run_log["final_status"] = "UNKNOWN_COMPLETION_STATE"

    # Calculate direct fix success rate if any attempts were made
    if task_run_log.get("direct_fix_attempts", 0) > 0:
        success_rate = task_run_log.get("direct_fix_successes", 0) / task_run_log["direct_fix_attempts"] * 100
        task_run_log["direct_fix_success_rate"] = f"{success_rate:.2f}%"
    else:
        task_run_log["direct_fix_success_rate"] = "N/A"
    
    # Print a summary of the task results
    logger.info("\n" + "=" * 80)
    logger.info("RESULTS:")
    logger.info("=" * 80)
    logger.info(f"Status: {task_run_log['final_status']}")
    if "final_source" in task_run_log:
        logger.info(f"Best Source: {task_run_log['final_source']}")
    logger.info(f"Best Score: {task_run_log['final_score']}")
    logger.info(f"Direct Fix Attempts: {task_run_log['direct_fix_attempts']}")
    logger.info(f"Direct Fix Success Rate: {task_run_log['direct_fix_success_rate']}")
    
    if task_run_log["final_code"]:
        logger.info("\nFinal Code:")
        logger.info("-" * 40)
        logger.info(f"```python\n{task_run_log['final_code']}\n```")
        logger.info("-" * 40)
    
    return task_run_log


def _append_summary_to_log(task_results: List[Dict[str, Any]], log_file_path: str) -> None:
    """
    Appends a summary of task results to the specified log file.
    
    Args:
        task_results: List of task result dictionaries from run_single_task
        log_file_path: Path to the log file
    """
    with open(log_file_path, "a") as f:
        f.write("\n\n" + "=" * 100 + "\n")
        f.write(f"ENHANCED EVALUATION SUMMARY (generated at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n")
        f.write("=" * 100 + "\n\n")
        
        # Overall statistics
        total_tasks = len(task_results)
        successful_tasks = sum(1 for r in task_results if r["final_status"] in ["SUCCESS", "SUCCESS_DIRECT_FIX"])
        success_rate = successful_tasks / total_tasks * 100 if total_tasks > 0 else 0
        
        total_direct_fix_attempts = sum(r.get("direct_fix_attempts", 0) for r in task_results)
        total_direct_fix_successes = sum(r.get("direct_fix_successes", 0) for r in task_results)
        direct_fix_success_rate = total_direct_fix_successes / total_direct_fix_attempts * 100 if total_direct_fix_attempts > 0 else 0
        
        # Best solution source counts
        direct_fix_best = sum(1 for r in task_results if r.get("final_source") == "DIRECT_FIX")
        self_healing_best = sum(1 for r in task_results if r.get("final_source") == "SELF_HEALING")
        initial_best = sum(1 for r in task_results if r.get("final_source") == "INITIAL")
        
        f.write(f"Total tasks: {total_tasks}\n")
        f.write(f"Successful tasks: {successful_tasks} ({success_rate:.2f}%)\n")
        f.write(f"Direct fix attempts: {total_direct_fix_attempts}\n")
        f.write(f"Direct fix successes: {total_direct_fix_successes} ({direct_fix_success_rate:.2f}%)\n")
        f.write(f"Tasks where direct fix was best: {direct_fix_best} ({direct_fix_best/total_tasks*100:.2f}%)\n")
        f.write(f"Tasks where self-healing was best: {self_healing_best} ({self_healing_best/total_tasks*100:.2f}%)\n")
        f.write(f"Tasks where initial code was best: {initial_best} ({initial_best/total_tasks*100:.2f}%)\n\n")
        
        # Individual task results
        f.write("INDIVIDUAL TASK RESULTS:\n")
        f.write("-" * 80 + "\n")
        
        for i, result in enumerate(task_results, 1):
            task_id = result["task_id"]
            status = result["final_status"]
            score = result["final_score"]
            source = result.get("final_source", "N/A")
            direct_fix_attempts = result.get("direct_fix_attempts", 0)
            direct_fix_successes = result.get("direct_fix_successes", 0)
            
            f.write(f"Task {i}: {task_id}\n")
            f.write(f"  Status: {status}\n")
            f.write(f"  Score: {score}\n")
            f.write(f"  Best Source: {source}\n")
            f.write(f"  Direct Fix: {direct_fix_successes}/{direct_fix_attempts}\n")
            
            # Log detailed test results for the final successful code of this task
            # This requires accessing the critique from the last successful iteration or direct fix
            
            critique_to_log = None
            if result.get("final_source") == "DIRECT_FIX" and "direct_fix_critique" in result["iterations_data"][0]:
                critique_to_log = result["iterations_data"][0]["direct_fix_critique"]
            elif result.get("final_source") == "SELF_HEALING":
                # Find the last successful self-healing iteration's critique
                for iteration_data in reversed(result.get("iterations_data", [])):
                    if iteration_data.get("phase") == "SELF_HEALING" and iteration_data.get("optimized_status") == CRITIC_STATUS_SUCCESS:
                        critique_to_log = iteration_data.get("optimized_critique")
                        break
            elif result.get("final_source") == "INITIAL" and "critic_output" in result["iterations_data"][0]:
                 critique_to_log = result["iterations_data"][0]["critic_output"]


            if critique_to_log and isinstance(critique_to_log.get("test_results"), list):
                f.write("  Test Case Details (from final successful code):\n")
                
                for test_case_num, test_detail in enumerate(critique_to_log["test_results"], 1):
                    passed = test_detail.get('passed', False)
                    status_text = "PASSED" if passed else "FAILED"
                    inputs = test_detail.get('input', 'N/A')
                    expected = test_detail.get('expected', 'N/A')
                    actual = test_detail.get('actual', 'N/A')
                    error = test_detail.get('error', 'N/A')
                    stdout = test_detail.get('stdout', 'N/A') # Get stdout
                    stderr = test_detail.get('stderr', 'N/A') # Get stderr
                    
                    f.write(f"    Test Case {test_case_num}: {status_text}\n")
                    f.write(f"      Input: {inputs}\n")
                    f.write(f"      Expected: {expected}\n")
                    f.write(f"      Actual: {actual}\n")
                    if error != 'N/A' and error: # Only print if error exists
                        f.write(f"      Error: {error}\n")
                    f.write(f"      Stdout: {stdout}\n") # Log stdout
                    f.write(f"      Stderr: {stderr}\n") # Log stderr
            else:
                f.write("  No detailed test results available for the final code.\n")

            f.write("-" * 80 + "\n")


def main_enhanced_evaluation_harness() -> None:
    """
    Main function to run the enhanced evaluation harness with the
    direct fix capability.
    
    This function initializes the required agents and evaluation task(s),
    then processes each task through the enhanced evaluation flow and
    generates a summary report.
    """
    logger.info(f"Starting Enhanced Evaluation Harness at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Configure LLM service
    provider = os.environ.get("LLM_PROVIDER", "deepseek")
    model_name = os.environ.get("LLM_MODEL", "deepseek-coder")
    logger.info(f"Using LLM provider: {provider}, model: {model_name}")
    
    llm_service = None
    try:
        # Initialize with required provider parameter
        llm_service = LLMService(
            provider=provider,
            model_name=model_name
        )
        logger.info("LLM service initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize LLM service: {e}")
        return
    
    # Initialize agents
    try:
        logger.info("🏗️  CREATING AGENTS...")
        planner = Planner(llm_service=llm_service, name="Planner")
        logger.info(f"✅ PLANNER CREATED: {planner.name}")
        
        executor = Executor(llm_service=llm_service, name="Executor")
        logger.info(f"✅ EXECUTOR CREATED: {executor.name} with initial system_prompt: '{executor.system_prompt[:100]}...'")
        
        critic = Critic(llm_service=llm_service, name="Critic")
        logger.info(f"✅ CRITIC CREATED: {critic.name}")
        
        logger.info("Agents initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize agents: {e}")
        return
    
    # Define evaluation tasks
    # You can add more tasks or load them from a JSON file
    evaluation_tasks = [
        {
            "id": "Wildcard Matching",
            "description": """Given an input string (s) and a pattern (p), implement wildcard pattern matching with support for '?' and '*' where:\n\n'?' Matches any single character.\n'*' Matches any sequence of characters (including the empty sequence).\nThe matching should cover the entire input string (not partial).\n\nExample 1:\nInput: s = \"aa\", p = \"a\"\nOutput: false\nExplanation: \"a\" does not match the entire string \"aa\".\n\nExample 2:\nInput: s = \"aa\", p = \"*\"\nOutput: true\nExplanation: '*' matches any sequence.\n\nExample 3:\nInput: s = \"cb\", p = \"?a\"\nOutput: false\nExplanation: '?' matches 'c', but the second letter is 'a', which does not match 'b'.

            """,
            "initial_executor_prompt": ULTRA_BUGGY_PROMPT
        },
        # Add more tasks as needed
    ]
    
    # Process each task
    all_task_results = []
    for task in evaluation_tasks:
        task_result = run_single_task(
            task_definition=task,
            planner=planner,
            executor=executor,
            critic=critic,
            llm_service_instance=llm_service
        )
        all_task_results.append(task_result)
    
    # Generate summary report
    _append_summary_to_log(all_task_results, LOG_FILE_NAME)
    
    logger.info(f"Enhanced Evaluation Harness completed at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Results saved to {LOG_FILE_NAME}")


if __name__ == "__main__":
    main_enhanced_evaluation_harness()
