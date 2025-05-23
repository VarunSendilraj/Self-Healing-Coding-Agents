"""
Enhanced Multi-Agent Self-Healing Harness

This module extends the enhanced harness with intelligent failure classification 
and targeted self-healing for both planning and execution agents.
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
from self_healing_agents.agents import Planner, Executor, Critic, PlannerSelfHealer, ExecutorSelfHealer
from self_healing_agents.llm_service import LLMService, LLMServiceError
from self_healing_agents.classifiers import FailureClassifier, FailureType, PlanValidator
from self_healing_agents.prompts import DEFAULT_EXECUTOR_SYSTEM_PROMPT, ULTRA_BUGGY_PROMPT, BAD_PLANNER_PROMPT
from self_healing_agents.schemas import CRITIC_STATUS_SUCCESS, CRITIC_STATUS_FAILURE_EVALUATION

# Import enhanced harness for base functionality and color support
try:
    from .enhanced_harness import TermColors, CRITIC_SUCCESS_THRESHOLD
except ImportError:
    # Handle direct execution
    from enhanced_harness import TermColors, CRITIC_SUCCESS_THRESHOLD

logger = logging.getLogger(__name__)

def run_enhanced_multi_agent_task(
    task_definition: Dict[str, Any],
    planner: Planner,
    executor: Executor,
    critic: Critic,
    llm_service_instance: LLMService,
    max_healing_iterations: int = 3
) -> Dict[str, Any]:
    """
    Run a single task with enhanced multi-agent self-healing that can target
    both planning and execution failures independently.
    
    Args:
        task_definition: Dictionary containing task information
        planner: Planner agent instance
        executor: Executor agent instance  
        critic: Critic agent instance
        llm_service_instance: LLM service instance
        max_healing_iterations: Maximum number of self-healing attempts
        
    Returns:
        Dictionary containing comprehensive task results
    """
    task_id = task_definition["id"]
    task_description = task_definition["description"]
    initial_prompt = task_definition["initial_executor_prompt"]

    logger.info(f"--- Starting Enhanced Multi-Agent Task: {TermColors.color_text(task_id, TermColors.HEADER)} ---")
    logger.info(f"Description: {TermColors.color_text(task_description, TermColors.CYAN)}")
    
    # Initialize enhanced components
    failure_classifier = FailureClassifier()
    plan_validator = PlanValidator()
    planner_healer = PlannerSelfHealer("PlannerHealer", llm_service_instance)
    executor_healer = ExecutorSelfHealer("ExecutorHealer", llm_service_instance)
    
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
            "direct_fix_attempts": 0
        },
        "classification_history": [],
        "plan_validation_history": []
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
            
            logger.info(f"    {TermColors.color_text('Full Initial Critique:', TermColors.CYAN)}\n{critique_result}")
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
    
    # === PHASE 4: ENHANCED MULTI-AGENT SELF-HEALING ===
    logger.info(f"\n=== PHASE 4: Enhanced Multi-Agent Self-Healing ===")
    
    for healing_iteration in range(1, max_healing_iterations + 1):
        task_run_log["total_healing_iterations"] += 1
        
        logger.info(f"\n----- Healing Iteration {healing_iteration}/{max_healing_iterations} -----")
        
        healing_data = {
            "phase": "MULTI_AGENT_HEALING",
            "iteration_num": healing_iteration,
            "classification_attempted": False,
            "healing_target": None,
            "healing_successful": False
        }
        
        # Step 1: Classify the failure
        logger.info(TermColors.color_text("  1. Failure Classification:", TermColors.HEADER))
        healing_data["classification_attempted"] = True
        
        # Use the best available critique for classification
        critique_for_classification = direct_fix_data.get("direct_fix_critique", initial_phase_data.get("critic_output", {}))
        code_for_classification = best_code or initial_code
        
        classification_result = failure_classifier.classify_failure(
            plan=current_plan,
            code=code_for_classification,
            error_report=critique_for_classification,
            task_description=task_description
        )
        
        healing_data["classification_result"] = classification_result
        task_run_log["classification_history"].append(classification_result)
        
        logger.info(f"    {TermColors.color_text('Failure Type:', TermColors.CYAN)} {classification_result['failure_type'].value}")
        logger.info(f"    {TermColors.color_text('Confidence:', TermColors.CYAN)} {classification_result['confidence']:.2f}")
        logger.info(f"    {TermColors.color_text('Recommended Target:', TermColors.CYAN)} {classification_result['recommended_healing_target']}")
        
        # Step 2: Apply targeted healing
        healing_target = classification_result["recommended_healing_target"]
        healing_data["healing_target"] = healing_target
        
        if healing_target == "PLANNER":
            # === PLANNER HEALING ===
            logger.info(TermColors.color_text("  2. Planner Healing:", TermColors.HEADER))
            task_run_log["healing_breakdown"]["planner_healings"] += 1
            
            try:
                improved_plan = planner_healer.heal_plan(
                    original_plan=current_plan,
                    failure_report=critique_for_classification,
                    task_description=task_description,
                    classification_result=classification_result
                )
                healing_data["improved_plan"] = improved_plan
                logger.info(f"    {TermColors.color_text('Improved Plan:', TermColors.GREEN)}\n{improved_plan}")
                
                # Update current plan
                current_plan = improved_plan
                
                # Re-execute with improved plan
                logger.info(TermColors.color_text("  3. Re-execution with Improved Plan:", TermColors.HEADER))
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
                    
                    # Check for success
                    if improved_status == CRITIC_STATUS_SUCCESS or improved_score >= CRITIC_SUCCESS_THRESHOLD:
                        healing_data["healing_successful"] = True
                        task_run_log["final_status"] = "SUCCESS_PLANNER_HEALING"
                        task_run_log["final_code"] = improved_code
                        task_run_log["final_plan"] = current_plan
                        task_run_log["final_score"] = improved_score
                        task_run_log["workflow_phases"].append(healing_data)
                        return task_run_log
                        
                    # Update best solution if improved
                    if improved_score > best_score:
                        best_code = improved_code
                        best_score = improved_score
                        best_source = f"PLANNER_HEALING_ITERATION_{healing_iteration}"
                        
            except Exception as e:
                logger.error(f"Planner healing failed: {e}")
                healing_data["planner_healing_error"] = str(e)
                
        else:
            # === EXECUTOR HEALING ===
            logger.info(TermColors.color_text("  2. Executor Healing:", TermColors.HEADER))
            task_run_log["healing_breakdown"]["executor_healings"] += 1
            
            try:
                improved_code = executor_healer.heal_code(
                    original_code=code_for_classification,
                    plan=current_plan,
                    failure_report=critique_for_classification,
                    task_description=task_description,
                    classification_result=classification_result
                )
                healing_data["improved_code"] = improved_code
                logger.info(f"    {TermColors.color_text('Improved Code:', TermColors.GREEN)}\n{improved_code}")
                
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
                    
                    # Check for success
                    if improved_status == CRITIC_STATUS_SUCCESS or improved_score >= CRITIC_SUCCESS_THRESHOLD:
                        healing_data["healing_successful"] = True
                        task_run_log["final_status"] = "SUCCESS_EXECUTOR_HEALING"
                        task_run_log["final_code"] = improved_code
                        task_run_log["final_plan"] = current_plan
                        task_run_log["final_score"] = improved_score
                        task_run_log["workflow_phases"].append(healing_data)
                        return task_run_log
                        
                    # Update best solution if improved
                    if improved_score > best_score:
                        best_code = improved_code
                        best_score = improved_score
                        best_source = f"EXECUTOR_HEALING_ITERATION_{healing_iteration}"
                        
            except Exception as e:
                logger.error(f"Executor healing failed: {e}")
                healing_data["executor_healing_error"] = str(e)
        
        task_run_log["workflow_phases"].append(healing_data)
        
    # === FINAL RESULTS ===
    logger.info("\n" + "=" * 80)
    logger.info("ENHANCED MULTI-AGENT RESULTS:")
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
    logger.info(f"Direct Fix Attempts: {task_run_log['healing_breakdown']['direct_fix_attempts']}")
    
    return task_run_log 

def test_planner_healing_with_real_llm():
    """
    Test planner healing functionality with real LLM service to see actual failures and healing
    """
    import os
    from self_healing_agents.agents import Planner, Executor, Critic
    from self_healing_agents.llm_service import LLMService
    from self_healing_agents.prompts import BAD_PLANNER_PROMPT, DEFAULT_EXECUTOR_SYSTEM_PROMPT
    
    print("🚀 REAL LLM PLANNER HEALING TEST")
    print("=" * 60)
    print("🎯 Using REAL LLM service with:")
    print("   - BAD planner prompt (vague, unhelpful plans)")
    print("   - GOOD executor prompt (should generate working code)")
    print("   - Real critic evaluation")
    print("=" * 60)
    
    # Configure LLM service
    provider = os.environ.get("LLM_PROVIDER", "deepseek")
    model_name = os.environ.get("LLM_MODEL", "deepseek-coder")
    
    try:
        llm_service = LLMService(provider=provider, model_name=model_name)
        print(f"✅ LLM service initialized: {provider}/{model_name}")
    except Exception as e:
        print(f"❌ Failed to initialize LLM service: {e}")
        return
    
    # Initialize agents with targeted prompts
    planner = Planner("BadPlanner", llm_service, BAD_PLANNER_PROMPT)  # BAD planning
    executor = Executor("GoodExecutor", llm_service, DEFAULT_EXECUTOR_SYSTEM_PROMPT)  # GOOD execution
    critic = Critic("RealCritic", llm_service)
    
    # Create challenging task that requires good planning
    test_task = {
        "id": "complex_algorithm_planning_test", 
        "description": "Implement a function to find the longest common subsequence (LCS) of two strings. The function should return the length of the longest common subsequence. Use dynamic programming for optimal solution.",
        "initial_executor_prompt": DEFAULT_EXECUTOR_SYSTEM_PROMPT
    }
    
    print(f"📝 Running complex task: {test_task['description']}")
    print()
    
    # Run the enhanced multi-agent task
    result = run_enhanced_multi_agent_task(
        task_definition=test_task,
        planner=planner,
        executor=executor,
        critic=critic,
        llm_service_instance=llm_service,
        max_healing_iterations=3
    )
    
    # Display comprehensive results
    print("\n" + "=" * 60)
    print("📊 REAL LLM PLANNER HEALING RESULTS:")
    print("=" * 60)
    print(f"✅ Final Status: {result['final_status']}")
    print(f"🎯 Final Score: {result['final_score']:.2f}")
    print(f"🔧 Total Healing Iterations: {result['total_healing_iterations']}")
    print(f"📋 Planner Healings: {result['healing_breakdown']['planner_healings']}")
    print(f"⚙️  Executor Healings: {result['healing_breakdown']['executor_healings']}")
    print(f"🔨 Direct Fix Attempts: {result['healing_breakdown']['direct_fix_attempts']}")
    
    # Analyze failure classifications
    if result['classification_history']:
        print(f"\n🔍 FAILURE CLASSIFICATION ANALYSIS:")
        for i, classification in enumerate(result['classification_history'], 1):
            failure_type = classification['failure_type'].value
            confidence = classification['confidence']
            target = classification['recommended_healing_target']
            reasoning = classification.get('reasoning', 'N/A')
            
            print(f"   Iteration {i}:")
            print(f"      🔸 Failure Type: {failure_type}")
            print(f"      🔸 Confidence: {confidence:.2f}")
            print(f"      🔸 Healing Target: {target}")
            print(f"      🔸 Reasoning: {reasoning}")
    
    # Show detailed plan evolution
    if result['workflow_phases']:
        print(f"\n📋 DETAILED PLAN EVOLUTION:")
        for i, phase in enumerate(result['workflow_phases'], 1):
            phase_name = phase.get('phase', 'UNKNOWN')
            print(f"\n   📌 Phase {i}: {phase_name}")
            
            # Show original plan
            if 'planner_output' in phase:
                plan = phase['planner_output']
                if isinstance(plan, dict) and 'plan_steps' in plan:
                    print(f"      📝 Original Plan ({len(plan['plan_steps'])} steps):")
                    for j, step in enumerate(plan['plan_steps'], 1):
                        print(f"         {j}. {step}")
            
            # Show healed plan if available
            if 'improved_plan' in phase:
                healed_plan = phase['improved_plan']
                if isinstance(healed_plan, dict) and 'plan_steps' in healed_plan:
                    print(f"      ✨ HEALED Plan ({len(healed_plan['plan_steps'])} steps):")
                    for j, step in enumerate(healed_plan['plan_steps'], 1):
                        print(f"         {j}. {step}")
            
            # Show scores if available
            if 'critic_score' in phase:
                print(f"      📊 Score: {phase['critic_score']:.2f}")
            if 'improved_score' in phase:
                print(f"      📈 Improved Score: {phase['improved_score']:.2f}")
    
    # Final analysis
    print(f"\n🎯 ANALYSIS:")
    if result['healing_breakdown']['planner_healings'] > 0:
        print(f"   ✅ SUCCESS: Planner healing was triggered and applied!")
        print(f"   🔧 The system detected bad planning and successfully healed the planner {result['healing_breakdown']['planner_healings']} time(s)")
    else:
        print(f"   ⚠️  No planner healing occurred")
        if result['final_score'] >= 0.9:
            print(f"   💡 This might be because the executor was able to work despite the bad plan")
        else:
            print(f"   💡 Check if the failure was classified as execution rather than planning")
    
    if result['total_healing_iterations'] > 0:
        print(f"   🔄 Total healing attempts: {result['total_healing_iterations']}")
        print(f"   📊 Final improvement: Initial plan → Final score {result['final_score']:.2f}")
    
    return result

if __name__ == "__main__":
    # Run the real LLM test instead of mock test
    test_planner_healing_with_real_llm() 