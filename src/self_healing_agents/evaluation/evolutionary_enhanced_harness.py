"""
Evolutionary Enhanced Multi-Agent Self-Healing Harness

This module extends the enhanced harness with evolutionary prompt optimization
for systematic self-healing of both planning and execution agents.
"""

import sys
import os
import logging
import re
from typing import Any, Dict, List, Tuple, Optional
import datetime

# Ensure the src directory is in the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(os.path.dirname(current_dir))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Import base classes and services
from self_healing_agents.agents import Planner, Executor, Critic
from self_healing_agents.llm_service import LLMService, LLMServiceError
from self_healing_agents.classifiers import PlanValidator
from self_healing_agents.prompts import DEFAULT_EXECUTOR_SYSTEM_PROMPT, ULTRA_BUGGY_PROMPT, BAD_PLANNER_PROMPT, PLANNER_SYSTEM_PROMPT
from self_healing_agents.schemas import CRITIC_STATUS_SUCCESS, CRITIC_STATUS_FAILURE_EVALUATION
from self_healing_agents.classifiers.llm_failure_classifier import LLMFailureClassifier

# Import evolutionary components
from self_healing_agents.evolution import (
    EvolutionaryPromptOptimizer, 
    EvolutionConfig, 
    PromptGenerationConfig,
    create_planner_optimizer,
    create_executor_optimizer
)

# Import enhanced harness for base functionality and color support
try:
    from .enhanced_harness import TermColors, CRITIC_SUCCESS_THRESHOLD
except ImportError:
    # Handle direct execution
    from enhanced_harness import TermColors, CRITIC_SUCCESS_THRESHOLD

logger = logging.getLogger(__name__)


def run_evolutionary_multi_agent_task(
    task_definition: Dict[str, Any],
    planner: Planner,
    executor: Executor,
    critic: Critic,
    llm_service_instance: LLMService,
    max_healing_iterations: int = 2,  # Reduced since each healing is more powerful
    use_evolutionary_optimization: bool = True
) -> Dict[str, Any]:
    """
    Run a single task with evolutionary prompt optimization for self-healing.
    
    Args:
        task_definition: Dictionary containing task information
        planner: Planner agent instance
        executor: Executor agent instance  
        critic: Critic agent instance
        llm_service_instance: LLM service instance
        max_healing_iterations: Maximum number of self-healing attempts
        use_evolutionary_optimization: Whether to use evolutionary optimization
        
    Returns:
        Dictionary containing comprehensive task results
    """
    task_id = task_definition["id"]
    task_description = task_definition["description"]
    initial_prompt = task_definition["initial_executor_prompt"]

    logger.info(f"--- Starting Evolutionary Multi-Agent Task: {TermColors.color_text(task_id, TermColors.HEADER)} ---")
    logger.info(f"Description: {TermColors.color_text(task_description, TermColors.CYAN)}")
    logger.info(f"🧬 Evolutionary Optimization: {TermColors.color_text('ENABLED' if use_evolutionary_optimization else 'DISABLED', TermColors.GREEN if use_evolutionary_optimization else TermColors.YELLOW)}")
    
    # Initialize enhanced components
    failure_classifier = LLMFailureClassifier(llm_service_instance)
    plan_validator = PlanValidator()
    
    # Initialize evolutionary optimizers if enabled
    planner_optimizer = None
    executor_optimizer = None
    if use_evolutionary_optimization:
        planner_optimizer = create_planner_optimizer(llm_service_instance)
        executor_optimizer = create_executor_optimizer(llm_service_instance)
        logger.info(f"🧬 EVOLUTION: Initialized optimizers (Planner: {planner_optimizer.config.population_size}x{planner_optimizer.config.max_generations}, Executor: {executor_optimizer.config.population_size}x{executor_optimizer.config.max_generations})")
    
    # Set initial executor prompt
    logger.info(f"🔧 SETTING INITIAL EXECUTOR PROMPT: {initial_prompt[:100]}...")
    executor.set_prompt(initial_prompt)
    
    # Initialize comprehensive task log
    task_run_log: Dict[str, Any] = {
        "task_id": task_id,
        "description": task_description,
        "initial_prompt": initial_prompt,
        "workflow_phases": [],
        "final_status": "UNKNOWN",
        "final_code": None,
        "final_plan": None,
        "final_score": 0.0,
        "total_healing_iterations": 0,
        "healing_breakdown": {
            "planner_healings": 0,
            "executor_healings": 0,
            "direct_fix_attempts": 0,
            "evolutionary_optimizations": 0
        },
        "classification_history": [],
        "plan_validation_history": [],
        "evolutionary_results": []
    }
    
    # === PHASE 1: INITIAL PLANNING AND VALIDATION ===
    logger.info(f"\n=== PHASE 1: Initial Planning and Validation for Task '{TermColors.color_text(task_id, TermColors.BLUE)}' ===")
    
    initial_phase_data = {
        "phase": "INITIAL_PLANNING_AND_VALIDATION",
        "plan_validation_attempted": False,
        "plan_validation_passed": False
    }
    
    # Step 1: Planning
    logger.info(TermColors.color_text("  1. Planning:", TermColors.HEADER))
    try:
        plan = planner.run(user_request=task_description)
        initial_phase_data["planner_output"] = plan
        logger.info(f"    {TermColors.color_text('Planner Output:', TermColors.GREEN)}\n{plan}")
        
        if isinstance(plan, dict) and plan.get("error"):
            error_msg = f"Planner returned an error: {plan.get('error')}. Aborting task."
            logger.error(TermColors.color_text(f"    ERROR: {error_msg}", TermColors.FAIL))
            task_run_log["final_status"] = "FAILURE_PLANNER"
            initial_phase_data["planner_error"] = plan.get('error')
            task_run_log["workflow_phases"].append(initial_phase_data)
            return task_run_log
            
    except LLMServiceError as e:
        error_msg = f"LLMServiceError during Planner execution: {e}. Aborting task."
        logger.error(TermColors.color_text(f"    ERROR: {error_msg}", TermColors.FAIL))
        task_run_log["final_status"] = "FAILURE_LLM_SERVICE"
        initial_phase_data["planner_error"] = str(e)
        task_run_log["workflow_phases"].append(initial_phase_data)
        return task_run_log
    
    # Step 2: Plan Validation
    logger.info(TermColors.color_text("  2. Plan Validation:", TermColors.HEADER))
    initial_phase_data["plan_validation_attempted"] = True
    
    plan_validation_result = plan_validator.validate_plan(plan, task_description)
    initial_phase_data["plan_validation_result"] = plan_validation_result
    task_run_log["plan_validation_history"].append(plan_validation_result)
    
    logger.info(f"    {TermColors.color_text('Plan Quality Score:', TermColors.CYAN)} {plan_validation_result['quality_score']:.2f}")
    logger.info(f"    {TermColors.color_text('Plan Valid:', TermColors.GREEN if plan_validation_result['is_valid'] else TermColors.FAIL)} {plan_validation_result['is_valid']}")
    
    if plan_validation_result["issues"]:
        logger.warning(f"    {TermColors.color_text('Plan Issues:', TermColors.WARNING)} {plan_validation_result['issues']}")
    if plan_validation_result["warnings"]:
        logger.warning(f"    {TermColors.color_text('Plan Warnings:', TermColors.WARNING)} {plan_validation_result['warnings']}")
        
    initial_phase_data["plan_validation_passed"] = plan_validation_result["is_valid"]
    
    # === PHASE 2: EXECUTION AND INITIAL EVALUATION ===
    logger.info(f"\n=== PHASE 2: Execution and Initial Evaluation ===")
    
    # Variables to track the best solution
    current_plan = plan
    best_code = None
    best_score = 0.0
    best_source = "UNKNOWN"
    
    # Step 3: Execution
    logger.info(TermColors.color_text("  3. Execution:", TermColors.HEADER))
    try:
        initial_code = executor.run(plan=current_plan, original_request=task_description)
        initial_phase_data["executor_initial_code"] = initial_code
        logger.info(f"    {TermColors.color_text('Executor Output (initial code):', TermColors.GREEN)}\n{initial_code}")
        
        if isinstance(initial_code, dict) and initial_code.get("error"):
            error_msg = f"Executor returned an error: {initial_code.get('error')}. Aborting task."
            logger.error(TermColors.color_text(f"    ERROR: {error_msg}", TermColors.FAIL))
            task_run_log["final_status"] = "FAILURE_EXECUTOR"
            initial_phase_data["executor_error"] = initial_code.get('error')
            task_run_log["workflow_phases"].append(initial_phase_data)
            return task_run_log
            
    except LLMServiceError as e:
        error_msg = f"LLMServiceError during Executor execution: {e}. Aborting task."
        logger.error(TermColors.color_text(f"    ERROR: {error_msg}", TermColors.FAIL))
        task_run_log["final_status"] = "FAILURE_LLM_SERVICE"
        initial_phase_data["executor_error"] = str(e)
        task_run_log["workflow_phases"].append(initial_phase_data)
        return task_run_log

    # Step 4: Initial Evaluation
    logger.info(TermColors.color_text("  4. Initial Evaluation:", TermColors.HEADER))
    try:
        critique_result = critic.run(generated_code=initial_code, task_description=task_description, plan=current_plan)
        initial_phase_data["critic_output"] = critique_result
        
        # Handle field name variations
        if isinstance(critique_result, dict):
            overall_status = critique_result.get('overall_status', critique_result.get('status', 'UNKNOWN_ERROR'))
            score = critique_result.get('quantitative_score', critique_result.get('score', 0.0))
            
            initial_phase_data["critic_status"] = overall_status
            initial_phase_data["critic_score"] = score
            
            logger.info(f"    {TermColors.color_text('Critic Status:', TermColors.GREEN)} {overall_status}")
            logger.info(f"    {TermColors.color_text('Critic Score:', TermColors.GREEN)} {score}")
            
            # Update best solution tracking
            best_code = initial_code
            best_score = score
            best_source = "INITIAL"
            
            # Check if initial solution is already successful
            if overall_status == CRITIC_STATUS_SUCCESS or score >= CRITIC_SUCCESS_THRESHOLD:
                logger.info(TermColors.color_text("  Initial solution passed! No need for healing.", TermColors.GREEN))
                task_run_log["final_status"] = "SUCCESS"
                task_run_log["final_code"] = initial_code
                task_run_log["final_plan"] = current_plan
                task_run_log["final_score"] = score
                task_run_log["workflow_phases"].append(initial_phase_data)
                return task_run_log
                
    except Exception as e:
        error_msg = f"Error during initial evaluation: {e}"
        logger.error(TermColors.color_text(f"    ERROR: {error_msg}", TermColors.FAIL))
        initial_phase_data["critic_error"] = str(e)
        
    task_run_log["workflow_phases"].append(initial_phase_data)
    
    # === PHASE 3: DIRECT FIX ATTEMPT ===
    logger.info(f"\n=== PHASE 3: Direct Fix Attempt ===")
    
    direct_fix_data = {
        "phase": "DIRECT_FIX",
        "direct_fix_attempted": False,
        "direct_fix_successful": False
    }
    
    # Try direct fix as in original system
    logger.info(TermColors.color_text("  Direct Fix Attempt:", TermColors.HEADER))
    task_run_log["healing_breakdown"]["direct_fix_attempts"] = 1
    direct_fix_data["direct_fix_attempted"] = True
    
    try:
        fixed_code = executor.direct_fix_attempt(
            original_code=initial_code,
            error_report=critique_result,
            task_description=task_description,
            plan=current_plan
        )
        direct_fix_data["direct_fix_code"] = fixed_code
        logger.info(f"    {TermColors.color_text('Direct Fix Output:', TermColors.GREEN)}\n{fixed_code}")
        
        # Re-evaluate directly fixed code
        direct_fix_critique = critic.run(
            generated_code=fixed_code,
            task_description=task_description,
            plan=current_plan
        )
        direct_fix_data["direct_fix_critique"] = direct_fix_critique
        
        if isinstance(direct_fix_critique, dict):
            direct_fix_status = direct_fix_critique.get('overall_status', direct_fix_critique.get('status', 'UNKNOWN_ERROR'))
            direct_fix_score = direct_fix_critique.get('quantitative_score', direct_fix_critique.get('score', 0.0))
            
            direct_fix_data["direct_fix_status"] = direct_fix_status
            direct_fix_data["direct_fix_score"] = direct_fix_score
            
            logger.info(f"    {TermColors.color_text('Direct Fix Status:', TermColors.GREEN)} {direct_fix_status}")
            logger.info(f"    {TermColors.color_text('Direct Fix Score:', TermColors.GREEN)} {direct_fix_score}")
            
            # Check if direct fix succeeded
            if direct_fix_status == CRITIC_STATUS_SUCCESS or direct_fix_score >= CRITIC_SUCCESS_THRESHOLD:
                logger.info(TermColors.color_text("  Direct fix successful! No need for healing.", TermColors.GREEN))
                direct_fix_data["direct_fix_successful"] = True
                task_run_log["final_status"] = "SUCCESS_DIRECT_FIX"
                task_run_log["final_code"] = fixed_code
                task_run_log["final_plan"] = current_plan
                task_run_log["final_score"] = direct_fix_score
                task_run_log["workflow_phases"].append(direct_fix_data)
                return task_run_log
                
            # Update best solution if improved
            if direct_fix_score > best_score:
                best_code = fixed_code
                best_score = direct_fix_score
                best_source = "DIRECT_FIX"
                
    except Exception as e:
        logger.error(f"Direct fix failed: {e}")
        direct_fix_data["direct_fix_error"] = str(e)
        
    task_run_log["workflow_phases"].append(direct_fix_data)
    
    # === PHASE 4: EVOLUTIONARY PROMPT OPTIMIZATION ===
    logger.info(f"\n=== PHASE 4: Evolutionary Prompt Optimization ===")
    
    if not use_evolutionary_optimization:
        logger.info(TermColors.color_text("  Evolutionary optimization disabled. Using best solution found.", TermColors.YELLOW))
        task_run_log["final_status"] = "COMPLETED_WITHOUT_EVOLUTION"
        task_run_log["final_code"] = best_code
        task_run_log["final_plan"] = current_plan
        task_run_log["final_score"] = best_score
        return task_run_log
    
    for healing_iteration in range(1, max_healing_iterations + 1):
        task_run_log["total_healing_iterations"] += 1
        
        logger.info(f"\n----- Evolutionary Healing Iteration {healing_iteration}/{max_healing_iterations} -----")
        
        healing_data = {
            "phase": "EVOLUTIONARY_HEALING",
            "iteration_num": healing_iteration,
            "classification_attempted": False,
            "healing_target": None,
            "healing_successful": False,
            "evolutionary_optimization": False
        }
        
        # Step 1: Classify the failure
        logger.info(TermColors.color_text("  1. Failure Classification:", TermColors.HEADER))
        healing_data["classification_attempted"] = True
        
        # Use the best available critique for classification
        critique_for_classification = direct_fix_data.get("direct_fix_critique", initial_phase_data.get("critic_output", {}))
        code_for_classification = best_code or initial_code
        
        classification_result = failure_classifier.classify_failure(
            task_description=task_description,
            plan=current_plan,
            code=code_for_classification,
            error_report=critique_for_classification,
            additional_context={
                "healing_iteration": healing_iteration,
                "best_score": best_score,
                "max_iterations": max_healing_iterations,
                "use_evolutionary": True
            }
        )
        
        healing_data["classification_result"] = classification_result
        task_run_log["classification_history"].append(classification_result)
        
        logger.info(f"    {TermColors.color_text('Failure Type:', TermColors.CYAN)} {classification_result['primary_failure_type']}")
        logger.info(f"    {TermColors.color_text('Confidence:', TermColors.CYAN)} {classification_result['confidence']:.2f}")
        logger.info(f"    {TermColors.color_text('Recommended Target:', TermColors.CYAN)} {classification_result['recommended_healing_target']}")
        
        # Log LLM reasoning
        if classification_result.get("reasoning"):
            logger.info(f"    {TermColors.color_text('LLM Reasoning:', TermColors.YELLOW)}")
            for reason in classification_result["reasoning"]:
                logger.info(f"      - {reason}")
        
        # Step 2: Apply evolutionary prompt optimization
        healing_target = classification_result["recommended_healing_target"]
        healing_data["healing_target"] = healing_target
        
        if healing_target == "PLANNER":
            # === EVOLUTIONARY PLANNER OPTIMIZATION ===
            logger.info(TermColors.color_text("  2. Evolutionary Planner Optimization:", TermColors.HEADER))
            task_run_log["healing_breakdown"]["planner_healings"] += 1
            task_run_log["healing_breakdown"]["evolutionary_optimizations"] += 1
            healing_data["evolutionary_optimization"] = True
            
            try:
                # Get current planner prompt
                current_planner_prompt = planner.system_prompt
                
                # Create failure context for optimization
                failure_context = {
                    "original_task": task_description,
                    "failed_plan": current_plan,
                    "failure_report": critique_for_classification,
                    "classification": classification_result,
                    "iteration": healing_iteration
                }
                
                # 🆕 ENHANCED: Extract specific test failure information for targeting
                specific_test_failures = []
                if isinstance(critique_for_classification, dict) and 'test_results' in critique_for_classification:
                    test_results = critique_for_classification['test_results']
                    if test_results:
                        failed_tests = [t for t in test_results if t.get('status') == 'failed']
                        for test in failed_tests[:3]:  # Include up to 3 failed tests
                            test_failure = {
                                "test_name": test.get('name', 'unknown'),
                                "inputs": test.get('inputs', {}),
                                "expected_output": test.get('expected_output_spec', 'unknown'),
                                "actual_output": test.get('actual_output', 'unknown'),
                                "error_type": test.get('error_type', 'unknown')
                            }
                            specific_test_failures.append(test_failure)
                            
                            # 🎯 SPECIAL HANDLING FOR REGEX PATTERN FAILURES
                            if isinstance(test.get('inputs', {}), dict):
                                inputs = test['inputs']
                                if 's' in inputs and 'p' in inputs and '*' in inputs['p']:
                                    test_failure["regex_pattern_issue"] = {
                                        "string": inputs['s'],
                                        "pattern": inputs['p'],
                                        "contains_star": True,
                                        "likely_issue": "star_operator_handling"
                                    }
                
                # Add specific test failures to failure context
                if specific_test_failures:
                    failure_context["specific_test_failures"] = specific_test_failures
                    logger.info(f"   🎯 ENHANCED: Added {len(specific_test_failures)} specific test failures to context")
                    for i, test_failure in enumerate(specific_test_failures, 1):
                        logger.info(f"      Test {i}: {test_failure['test_name']} - {test_failure['inputs']} -> expected {test_failure['expected_output']}, got {test_failure['actual_output']}")
                        if "regex_pattern_issue" in test_failure:
                            logger.info(f"      🎯 REGEX ISSUE: Pattern '{test_failure['regex_pattern_issue']['pattern']}' with string '{test_failure['regex_pattern_issue']['string']}'")
                else:
                    logger.warning(f"   ⚠️ No specific test failures extracted - evolutionary targeting will be generic")
                
                # 🆕 DIAGNOSTIC LOGGING FOR FAILURE CONTEXT
                logger.info(f"🔍 FAILURE CONTEXT CONSTRUCTION (Planner):")
                logger.info(f"   📋 Original Task Length: {len(task_description)} chars")
                logger.info(f"   📋 Failed Plan Type: {type(current_plan)}")
                logger.info(f"   📋 Failure Report Type: {type(critique_for_classification)}")
                logger.info(f"   📋 Classification Result Type: {type(classification_result)}")
                
                # Log specific details from classification
                if isinstance(classification_result, dict):
                    primary_failure = classification_result.get('primary_failure_type', 'unknown')
                    reasoning = classification_result.get('reasoning', [])
                    logger.info(f"   🎯 Primary Failure Type: {primary_failure}")
                    logger.info(f"   🧠 LLM Reasoning Count: {len(reasoning)} items")
                    if reasoning:
                        logger.info(f"   🧠 First Reasoning: {reasoning[0][:100]}...")
                
                # Log failure report details
                if isinstance(critique_for_classification, dict) and 'test_results' in critique_for_classification:
                    test_results = critique_for_classification['test_results']
                    logger.info(f"   🧪 Test Results Available: {len(test_results) if test_results else 0} tests")
                    if test_results:
                        failed_tests = [t for t in test_results if t.get('status') == 'failed']
                        logger.info(f"   🧪 Failed Tests: {len(failed_tests)}")
                        if failed_tests:
                            logger.info(f"   🧪 First Failed Test: {failed_tests[0].get('name', 'unknown')}")
                else:
                    logger.warning(f"   ⚠️ No test_results attribute in critique_for_classification")
                
                logger.info(f"   ✅ Failure context constructed with {len(failure_context)} keys")
                
                # 🎯 SET UP TASK-SPECIFIC FITNESS EVALUATION
                logger.info(f"🎯 CONFIGURING: Task-specific fitness evaluation using actual multi-agent pipeline")
                planner_optimizer.fitness_evaluator.set_task_specific_context(
                    task_description=task_description,
                    planner_agent=planner,
                    executor_agent=executor, 
                    critic_agent=critic
                )
                
                # Run evolutionary optimization
                logger.info(f"    🧬 EVOLVING: Optimizing planner prompt through {planner_optimizer.config.max_generations} generations")
                evolution_results = planner_optimizer.optimize_prompt(
                    base_prompt=current_planner_prompt,
                    agent_type="PLANNER",
                    failure_context=failure_context
                )
                
                healing_data["evolution_results"] = {
                    "best_fitness": evolution_results.best_fitness,
                    "generation_count": evolution_results.generation_count,
                    "evaluation_count": evolution_results.evaluation_count,
                    "execution_time": evolution_results.execution_time,
                    "termination_reason": evolution_results.termination_reason,
                    "improvement": evolution_results.best_fitness - (evolution_results.convergence_history[0] if evolution_results.convergence_history else 0)
                }
                task_run_log["evolutionary_results"].append(healing_data["evolution_results"])
                
                logger.info(f"    ✨ EVOLUTION COMPLETE: Best fitness {evolution_results.best_fitness:.3f} after {evolution_results.generation_count} generations")
                
                # Update planner with evolved prompt
                optimized_prompt = evolution_results.best_prompt
                planner.system_prompt = optimized_prompt
                healing_data["optimized_prompt"] = optimized_prompt
                
                # 🆕 LOG THE EVOLVED PROMPT
                logger.info(f"    ✨ EVOLVED PLANNER PROMPT ({len(optimized_prompt)} chars):")
                logger.info(f"    {'─' * 60}")
                logger.info(f"    {optimized_prompt}{'...' if len(optimized_prompt) > 300 else ''}")
                logger.info(f"    {'─' * 60}")
                
                # Re-plan with optimized prompt
                logger.info(TermColors.color_text("  3. Re-planning with Evolved Prompt:", TermColors.HEADER))
                improved_plan = planner.run(user_request=task_description)
                healing_data["improved_plan"] = improved_plan
                
                # Update current plan
                current_plan = improved_plan
                
                # Re-execute with improved plan
                logger.info(TermColors.color_text("  4. Re-execution with Improved Plan:", TermColors.HEADER))
                improved_code = executor.run(plan=current_plan, original_request=task_description)
                healing_data["improved_code"] = improved_code
                
                # Evaluate the result
                improved_critique = critic.run(
                    generated_code=improved_code,
                    task_description=task_description,
                    plan=current_plan
                )
                healing_data["improved_critique"] = improved_critique
                
                if isinstance(improved_critique, dict):
                    improved_status = improved_critique.get('overall_status', improved_critique.get('status', 'UNKNOWN'))
                    improved_score = improved_critique.get('quantitative_score', improved_critique.get('score', 0.0))
                    
                    healing_data["improved_status"] = improved_status
                    healing_data["improved_score"] = improved_score
                    
                    logger.info(f"    {TermColors.color_text('Improved Status:', TermColors.GREEN)} {improved_status}")
                    logger.info(f"    {TermColors.color_text('Improved Score:', TermColors.GREEN)} {improved_score}")
                    
                    # 🚨 CRITICAL DIAGNOSTIC: Show the disconnect!
                    if healing_data.get("evolution_results"):
                        evo_fitness = healing_data["evolution_results"]["best_fitness"]
                        logger.warning(f"    🚨 CRITICAL DISCONNECT DETECTED:")
                        logger.warning(f"    🚨 Evolutionary Fitness: {evo_fitness:.3f} (from simple validation tasks)")
                        logger.warning(f"    🚨 Actual Task Score:    {improved_score:.3f} (from critic on real regex task)")
                        logger.warning(f"    🚨 Difference: {evo_fitness - improved_score:+.3f}")
                        logger.warning(f"    🚨 FITNESS EVALUATOR IS BROKEN - HIGH FITNESS ≠ TASK SUCCESS!")
                    
                    # Check for success
                    if improved_status == CRITIC_STATUS_SUCCESS or improved_score >= CRITIC_SUCCESS_THRESHOLD:
                        healing_data["healing_successful"] = True
                        task_run_log["final_status"] = "SUCCESS_EVOLUTIONARY_PLANNER"
                        task_run_log["final_code"] = improved_code
                        task_run_log["final_plan"] = current_plan
                        task_run_log["final_score"] = improved_score
                        task_run_log["workflow_phases"].append(healing_data)
                        return task_run_log
                        
                    # Update best solution if improved
                    if improved_score > best_score:
                        best_code = improved_code
                        best_score = improved_score
                        best_source = f"EVOLUTIONARY_PLANNER_ITERATION_{healing_iteration}"
                        
            except Exception as e:
                logger.error(f"Evolutionary planner optimization failed: {e}")
                healing_data["evolution_error"] = str(e)
                
        else:
            # === EVOLUTIONARY EXECUTOR OPTIMIZATION ===
            logger.info(TermColors.color_text("  2. Evolutionary Executor Optimization:", TermColors.HEADER))
            task_run_log["healing_breakdown"]["executor_healings"] += 1
            task_run_log["healing_breakdown"]["evolutionary_optimizations"] += 1
            healing_data["evolutionary_optimization"] = True
            
            try:
                # Get current executor prompt
                current_executor_prompt = executor.system_prompt
                
                # Create failure context for optimization
                failure_context = {
                    "original_task": task_description,
                    "plan": current_plan,
                    "failed_code": code_for_classification,
                    "failure_report": critique_for_classification,
                    "classification": classification_result,
                    "iteration": healing_iteration
                }
                
                # 🆕 ENHANCED: Extract specific test failure information for targeting
                specific_test_failures = []
                if isinstance(critique_for_classification, dict) and 'test_results' in critique_for_classification:
                    test_results = critique_for_classification['test_results']
                    if test_results:
                        failed_tests = [t for t in test_results if t.get('status') == 'failed']
                        for test in failed_tests[:3]:  # Include up to 3 failed tests
                            test_failure = {
                                "test_name": test.get('name', 'unknown'),
                                "inputs": test.get('inputs', {}),
                                "expected_output": test.get('expected_output_spec', 'unknown'),
                                "actual_output": test.get('actual_output', 'unknown'),
                                "error_type": test.get('error_type', 'unknown')
                            }
                            specific_test_failures.append(test_failure)
                            
                            # 🎯 SPECIAL HANDLING FOR REGEX PATTERN FAILURES
                            if isinstance(test.get('inputs', {}), dict):
                                inputs = test['inputs']
                                if 's' in inputs and 'p' in inputs and '*' in inputs['p']:
                                    test_failure["regex_pattern_issue"] = {
                                        "string": inputs['s'],
                                        "pattern": inputs['p'],
                                        "contains_star": True,
                                        "likely_issue": "star_operator_handling"
                                    }
                
                # Add specific test failures to failure context
                if specific_test_failures:
                    failure_context["specific_test_failures"] = specific_test_failures
                    logger.info(f"   🎯 ENHANCED: Added {len(specific_test_failures)} specific test failures to context")
                    for i, test_failure in enumerate(specific_test_failures, 1):
                        logger.info(f"      Test {i}: {test_failure['test_name']} - {test_failure['inputs']} -> expected {test_failure['expected_output']}, got {test_failure['actual_output']}")
                        if "regex_pattern_issue" in test_failure:
                            logger.info(f"      🎯 REGEX ISSUE: Pattern '{test_failure['regex_pattern_issue']['pattern']}' with string '{test_failure['regex_pattern_issue']['string']}'")
                else:
                    logger.warning(f"   ⚠️ No specific test failures extracted - evolutionary targeting will be generic")
                
                # 🆕 DIAGNOSTIC LOGGING FOR FAILURE CONTEXT
                logger.info(f"🔍 FAILURE CONTEXT CONSTRUCTION (Executor):")
                logger.info(f"   📋 Original Task Length: {len(task_description)} chars")
                logger.info(f"   📋 Plan Type: {type(current_plan)}")
                logger.info(f"   📋 Failed Code Length: {len(code_for_classification) if isinstance(code_for_classification, str) else 'N/A'} chars")
                logger.info(f"   📋 Failure Report Type: {type(critique_for_classification)}")
                logger.info(f"   📋 Classification Result Type: {type(classification_result)}")
                logger.info(f"   📋 Specific Test Failures: {len(specific_test_failures)} included")
                
                # Log specific details from classification
                if isinstance(classification_result, dict):
                    primary_failure = classification_result.get('primary_failure_type', 'unknown')
                    reasoning = classification_result.get('reasoning', [])
                    logger.info(f"   🎯 Primary Failure Type: {primary_failure}")
                    logger.info(f"   🧠 LLM Reasoning Count: {len(reasoning)} items")
                    if reasoning:
                        logger.info(f"   🧠 First Reasoning: {reasoning[0][:100]}...")
                        logger.info(f"   🧠 All Reasoning Items:")
                        for i, reason in enumerate(reasoning, 1):
                            logger.info(f"      {i}. {reason}")
                
                # Log failure report details - CHECK FOR TEST RESULTS
                if isinstance(critique_for_classification, dict) and 'test_results' in critique_for_classification:
                    test_results = critique_for_classification['test_results']
                    logger.info(f"   🧪 Test Results Available: {len(test_results) if test_results else 0} tests")
                    if test_results:
                        failed_tests = [t for t in test_results if t.get('status') == 'failed']
                        passed_tests = [t for t in test_results if t.get('status') == 'passed']
                        logger.info(f"   🧪 Passed Tests: {len(passed_tests)}")
                        logger.info(f"   🧪 Failed Tests: {len(failed_tests)}")
                        
                        # Log details of failed tests
                        for i, test in enumerate(failed_tests[:2], 1):  # Log first 2 failed tests
                            test_name = test.get('name', 'unknown')
                            test_inputs = test.get('inputs', {})
                            expected = test.get('expected_output_spec', 'unknown')
                            actual = test.get('actual_output', 'unknown')
                            logger.info(f"   🧪 Failed Test {i}: {test_name}")
                            logger.info(f"      📥 Inputs: {test_inputs}")
                            logger.info(f"      ✅ Expected: {expected}")
                            logger.info(f"      ❌ Actual: {actual}")
                            
                            # 🆕 CRITICAL: Check if this is the regex * pattern issue
                            if isinstance(test_inputs, dict) and 's' in test_inputs and 'p' in test_inputs:
                                s_val = test_inputs['s']
                                p_val = test_inputs['p']
                                if '*' in p_val:
                                    logger.warning(f"   🎯 REGEX PATTERN FAILURE DETECTED: s='{s_val}', p='{p_val}' (contains '*')")
                                    logger.warning(f"   🎯 This is the specific pattern that needs targeting!")
                else:
                    logger.warning(f"   ⚠️ No test_results attribute in critique_for_classification")
                    logger.warning(f"   ⚠️ This means no specific test failure information will be available for targeting")
                
                logger.info(f"   ✅ Failure context constructed with {len(failure_context)} keys")
                
                # 🎯 SET UP TASK-SPECIFIC FITNESS EVALUATION
                logger.info(f"🎯 CONFIGURING: Task-specific fitness evaluation using actual multi-agent pipeline")
                executor_optimizer.fitness_evaluator.set_task_specific_context(
                    task_description=task_description,
                    planner_agent=planner,
                    executor_agent=executor,
                    critic_agent=critic
                )
                
                # Run evolutionary optimization
                logger.info(f"    🧬 EVOLVING: Optimizing executor prompt through {executor_optimizer.config.max_generations} generations")
                evolution_results = executor_optimizer.optimize_prompt(
                    base_prompt=current_executor_prompt,
                    agent_type="EXECUTOR",
                    failure_context=failure_context
                )
                
                healing_data["evolution_results"] = {
                    "best_fitness": evolution_results.best_fitness,
                    "generation_count": evolution_results.generation_count,
                    "evaluation_count": evolution_results.evaluation_count,
                    "execution_time": evolution_results.execution_time,
                    "termination_reason": evolution_results.termination_reason,
                    "improvement": evolution_results.best_fitness - (evolution_results.convergence_history[0] if evolution_results.convergence_history else 0)
                }
                task_run_log["evolutionary_results"].append(healing_data["evolution_results"])
                
                logger.info(f"    ✨ EVOLUTION COMPLETE: Best fitness {evolution_results.best_fitness:.3f} after {evolution_results.generation_count} generations")
                
                # Update executor with evolved prompt
                optimized_prompt = evolution_results.best_prompt
                executor.system_prompt = optimized_prompt
                healing_data["optimized_prompt"] = optimized_prompt
                
                # 🆕 LOG THE EVOLVED PROMPT
                logger.info(f"    ✨ EVOLVED EXECUTOR PROMPT ({len(optimized_prompt)} chars):")
                logger.info(f"    {'─' * 60}")
                logger.info(f"    {optimized_prompt}{'...' if len(optimized_prompt) > 300 else ''}")
                logger.info(f"    {'─' * 60}")
                
                # Re-execute with optimized prompt
                logger.info(TermColors.color_text("  3. Re-execution with Evolved Prompt:", TermColors.HEADER))
                improved_code = executor.run(plan=current_plan, original_request=task_description)
                healing_data["improved_code"] = improved_code
                
                # Evaluate the improved code
                improved_critique = critic.run(
                    generated_code=improved_code,
                    task_description=task_description,
                    plan=current_plan
                )
                healing_data["improved_critique"] = improved_critique
                
                if isinstance(improved_critique, dict):
                    improved_status = improved_critique.get('overall_status', improved_critique.get('status', 'UNKNOWN'))
                    improved_score = improved_critique.get('quantitative_score', improved_critique.get('score', 0.0))
                    
                    healing_data["improved_status"] = improved_status
                    healing_data["improved_score"] = improved_score
                    
                    logger.info(f"    {TermColors.color_text('Improved Status:', TermColors.GREEN)} {improved_status}")
                    logger.info(f"    {TermColors.color_text('Improved Score:', TermColors.GREEN)} {improved_score}")
                    
                    # 🚨 CRITICAL DIAGNOSTIC: Show the disconnect!
                    if healing_data.get("evolution_results"):
                        evo_fitness = healing_data["evolution_results"]["best_fitness"]
                        logger.warning(f"    🚨 CRITICAL DISCONNECT DETECTED:")
                        logger.warning(f"    🚨 Evolutionary Fitness: {evo_fitness:.3f} (from simple validation tasks)")
                        logger.warning(f"    🚨 Actual Task Score:    {improved_score:.3f} (from critic on real regex task)")
                        logger.warning(f"    🚨 Difference: {evo_fitness - improved_score:+.3f}")
                        logger.warning(f"    🚨 FITNESS EVALUATOR IS BROKEN - HIGH FITNESS ≠ TASK SUCCESS!")
                    
                    # Check for success
                    if improved_status == CRITIC_STATUS_SUCCESS or improved_score >= CRITIC_SUCCESS_THRESHOLD:
                        healing_data["healing_successful"] = True
                        task_run_log["final_status"] = "SUCCESS_EVOLUTIONARY_EXECUTOR"
                        task_run_log["final_code"] = improved_code
                        task_run_log["final_plan"] = current_plan
                        task_run_log["final_score"] = improved_score
                        task_run_log["workflow_phases"].append(healing_data)
                        return task_run_log
                        
                    # Update best solution if improved
                    if improved_score > best_score:
                        best_code = improved_code
                        best_score = improved_score
                        best_source = f"EVOLUTIONARY_EXECUTOR_ITERATION_{healing_iteration}"
                        
            except Exception as e:
                logger.error(f"Evolutionary executor optimization failed: {e}")
                healing_data["evolution_error"] = str(e)
        
        task_run_log["workflow_phases"].append(healing_data)
        
    # === FINAL RESULTS ===
    logger.info("\n" + "=" * 80)
    logger.info("EVOLUTIONARY MULTI-AGENT RESULTS:")
    logger.info("=" * 80)
    
    # Set final results
    task_run_log["final_status"] = "COMPLETED_MAX_HEALING_ITERATIONS" if best_code else "FAILURE_NO_SOLUTION"
    task_run_log["final_code"] = best_code
    task_run_log["final_plan"] = current_plan
    task_run_log["final_score"] = best_score
    
    logger.info(f"Status: {task_run_log['final_status']}")
    logger.info(f"Best Source: {best_source}")
    logger.info(f"Best Score: {task_run_log['final_score']:.2f}")
    logger.info(f"Total Healing Iterations: {task_run_log['total_healing_iterations']}")
    logger.info(f"Planner Healings: {task_run_log['healing_breakdown']['planner_healings']}")
    logger.info(f"Executor Healings: {task_run_log['healing_breakdown']['executor_healings']}")
    logger.info(f"Evolutionary Optimizations: {task_run_log['healing_breakdown']['evolutionary_optimizations']}")
    logger.info(f"Direct Fix Attempts: {task_run_log['healing_breakdown']['direct_fix_attempts']}")
    
    # Log evolutionary statistics
    if task_run_log["evolutionary_results"]:
        logger.info(f"\nEvolutionary Statistics:")
        total_generations = sum(r["generation_count"] for r in task_run_log["evolutionary_results"])
        total_evaluations = sum(r["evaluation_count"] for r in task_run_log["evolutionary_results"])
        total_evolution_time = sum(r["execution_time"] for r in task_run_log["evolutionary_results"])
        average_improvement = sum(r["improvement"] for r in task_run_log["evolutionary_results"]) / len(task_run_log["evolutionary_results"])
        
        logger.info(f"  Total Generations: {total_generations}")
        logger.info(f"  Total Evaluations: {total_evaluations}")
        logger.info(f"  Total Evolution Time: {total_evolution_time:.1f}s")
        logger.info(f"  Average Fitness Improvement: {average_improvement:.3f}")
    
    return task_run_log


def test_evolutionary_healing_with_real_llm():
    """
    Test evolutionary prompt healing with real LLM service.
    """
    import os
    from self_healing_agents.agents import Planner, Executor, Critic
    from self_healing_agents.llm_service import LLMService
    
    print("🧬 EVOLUTIONARY PROMPT OPTIMIZATION TEST")
    print("=" * 60)
    print("🎯 Goal: Test evolutionary optimization for self-healing")
    print("=" * 60)
    
    # Configure environment
    provider = os.environ.get("LLM_PROVIDER", "deepseek")
    model_name = os.environ.get("LLM_MODEL", "deepseek-coder")
    
    # Initialize LLM service
    try:
        llm_service = LLMService(provider=provider, model_name=model_name)
        print(f"✅ LLM Service: {provider}/{model_name}")
    except Exception as e:
        print(f"❌ LLM Service Error: {e}")
        return
    
    # Initialize agents with problematic prompts
    print(f"\n🔧 AGENT CONFIGURATION:")
    print(f"   🤖 Planner: BAD_PLANNER_PROMPT (will trigger optimization)")
    print(f"   🔧 Executor: ULTRA_BUGGY_PROMPT (will trigger optimization)")
    print(f"   🧐 Critic: Standard evaluation")
    
    planner = Planner("BadPlanner", llm_service, PLANNER_SYSTEM_PROMPT)
    executor = Executor("BuggyExecutor", llm_service, DEFAULT_EXECUTOR_SYSTEM_PROMPT)
    critic = Critic("Critic", llm_service)
    
    # Create challenging task
    test_task = {
        "id": "evolutionary_optimization_test",
        "description": """Given an input string s and a pattern p, implement regular expression matching with support for '.' and '*' where:\n\n'.' Matches any single character.\n'*' Matches zero or more of the preceding element.\nThe matching should cover the entire input string (not partial).\n\nExample 1:\nInput: s = \"aa\", p = \"a\"\nOutput: false\nExplanation: \"a\" does not match the entire string \"aa\".\n\nExample 2:\nInput: s = \"aa\", p = \"a*\"\nOutput: true\nExplanation: '*' means zero or more of the preceding element, 'a'. Therefore, by repeating 'a' once, it becomes \"aa\".\n\nExample 3:\nInput: s = \"ab\", p = \".*\"\nOutput: true\nExplanation: \".*\" means \"zero or more (*) of any character (.)\".\n\nConstraints:\n1 <= s.length <= 20\n1 <= p.length <= 20\ns contains only lowercase English letters.\np contains only lowercase English letters, '.', and '*'.\nIt is guaranteed for each appearance of the character '*', there will be a previous valid character to match.""",
        "initial_executor_prompt": DEFAULT_EXECUTOR_SYSTEM_PROMPT
    }
    
    print(f"📝 Running evolutionary test: {test_task['description']}")
    print()
    
    # Run the evolutionary multi-agent task
    result = run_evolutionary_multi_agent_task(
        task_definition=test_task,
        planner=planner,
        executor=executor,
        critic=critic,
        llm_service_instance=llm_service,
        max_healing_iterations=2,  # Back to 2 for proper testing
        use_evolutionary_optimization=True
    )
    
    # Display results
    print("\n" + "=" * 60)
    print("📊 EVOLUTIONARY OPTIMIZATION RESULTS:")
    print("=" * 60)
    print(f"✅ Final Status: {result['final_status']}")
    print(f"🎯 Final Score: {result['final_score']:.2f}")
    print(f"🔧 Total Healing Iterations: {result['total_healing_iterations']}")
    print(f"🧬 Evolutionary Optimizations: {result['healing_breakdown']['evolutionary_optimizations']}")
    
    # Show evolutionary details
    if result['evolutionary_results']:
        print(f"\n🧬 EVOLUTIONARY DETAILS:")
        for i, evo_result in enumerate(result['evolutionary_results'], 1):
            print(f"   Optimization {i}:")
            print(f"      🏆 Best Fitness: {evo_result['best_fitness']:.3f}")
            print(f"      🔄 Generations: {evo_result['generation_count']}")
            print(f"      📊 Evaluations: {evo_result['evaluation_count']}")
            print(f"      ⏱️  Time: {evo_result['execution_time']:.1f}s")
            print(f"      📈 Improvement: {evo_result['improvement']:+.3f}")
            print(f"      🛑 Termination: {evo_result['termination_reason']}")
    
    return result


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Run the test
    test_evolutionary_healing_with_real_llm() 