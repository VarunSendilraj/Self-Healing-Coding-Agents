"""
Test script specifically designed to trigger LLM-based failure classification.
This uses a more challenging task that requires good planning to succeed.
"""

import logging
import os
from self_healing_agents.agents import Planner, Executor, Critic
from self_healing_agents.llm_service import LLMService
from self_healing_agents.prompts import BAD_PLANNER_PROMPT, DEFAULT_EXECUTOR_SYSTEM_PROMPT, PLANNER_SYSTEM_PROMPT
from self_healing_agents.evaluation.enhanced_multi_agent_harness import run_enhanced_multi_agent_task
from self_healing_agents.evaluation.enhanced_harness import TermColors

def main():
    """Test LLM-based failure classification with a challenging task."""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("🧪 LLM-BASED FAILURE CLASSIFIER TEST")
    print("=" * 60)
    print("🎯 Goal: Trigger actual failures that require LLM analysis")
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
    
    # Initialize agents with targeted prompts  
    print(f"\n🔧 AGENT CONFIGURATION:")
    print(f"   🤖 Planner: PLANNER_SYSTEM_PROMPT (vague, unhelpful)")
    print(f"   🔧 Executor: DEFAULT_EXECUTOR_SYSTEM_PROMPT (standard)")
    print(f"   🧐 Critic: Standard evaluation")
    print(f"   🤖 Classifier: LLM-based intelligent analysis")
    
    planner = Planner("Planner", llm_service, PLANNER_SYSTEM_PROMPT)
    executor = Executor("Executor", llm_service, DEFAULT_EXECUTOR_SYSTEM_PROMPT)
    critic = Critic("Critic", llm_service)
    
    # Test with a complex task that requires good planning
    complex_task = {
        "id": "complex_data_processing",
        "description": """
A permutation of an array of integers is an arrangement of its members into a sequence or linear order.\n\nFor example, for arr = [1,2,3], the following are all the permutations of arr: [1,2,3], [1,3,2], [2,1,3], [2,3,1], [3,1,2], [3,2,1].\nThe next permutation of an array of integers is the next lexicographically greater permutation of its integer. More formally, if all the permutations of the array are sorted in one container according to their lexicographical order, then the next permutation of that array is the permutation that follows it in the sorted container. If such arrangement is not possible, the array must be rearranged as the lowest possible order (i.e., sorted in ascending order).\n\nFor example, the next permutation of arr = [1,2,3] is [1,3,2].\nSimilarly, the next permutation of arr = [2,3,1] is [3,1,2].\nWhile the next permutation of arr = [3,2,1] is [1,2,3] because [3,2,1] does not have a lexicographical larger rearrangement.\nGiven an array of integers nums, find the next permutation of nums.\n\nThe replacement must be in place and use only constant extra memory.\n\nExample 1:\nInput: nums = [1,2,3]\nOutput: [1,3,2]\n\nExample 2:\nInput: nums = [3,2,1]\nOutput: [1,2,3]\n\nExample 3:\nInput: nums = [1,1,5]\nOutput: [1,5,1]


""",
        "initial_executor_prompt": DEFAULT_EXECUTOR_SYSTEM_PROMPT
    }
    
    print(f"\n📋 COMPLEX TEST TASK:")
    print(f"   ID: {complex_task['id']}")
    print(f"   Task: Multi-method class with data processing capabilities")
    print(f"   Challenge: Requires good planning and coordination between methods")
    
    # Run the enhanced harness
    print(f"\n🏃‍♂️ RUNNING ENHANCED MULTI-AGENT HARNESS...")
    print("=" * 60)
    
    result = run_enhanced_multi_agent_task(
        task_definition=complex_task,
        planner=planner,
        executor=executor,
        critic=critic,
        llm_service_instance=llm_service,
        max_healing_iterations=3
    )
    

    #--------------------------------
    #--------------------------------
    # Comprehensive results analysis
    print(f"\n📊 COMPREHENSIVE RESULTS ANALYSIS:")
    print("=" * 60)
    
    final_status = result['final_status']
    success = 'SUCCESS' in final_status
    
    print(f"   🎯 Final Status: {TermColors.color_text(final_status, TermColors.GREEN if success else TermColors.FAIL)}")
    final_score_str = f"{result['final_score']:.2f}"
    print(f"   📈 Final Score: {TermColors.color_text(final_score_str, TermColors.GREEN)}")
    print(f"   🔄 Total Healing Iterations: {result['total_healing_iterations']}")
    print(f"   🧠 Planner Healings: {result['healing_breakdown']['planner_healings']}")
    print(f"   ⚙️  Executor Healings: {result['healing_breakdown']['executor_healings']}")
    print(f"   🔨 Direct Fix Attempts: {result['healing_breakdown']['direct_fix_attempts']}")
    
    # Detailed LLM Classification Analysis
    if result.get('classification_history'):
        print(f"\n🤖 LLM FAILURE CLASSIFICATION DETAILED ANALYSIS:")
        print("=" * 60)
        
        for i, classification in enumerate(result['classification_history'], 1):
            failure_type = classification['primary_failure_type']
            confidence = classification['confidence']
            target = classification['recommended_healing_target']
            
            print(f"\n📋 Classification Iteration {i}:")
            print(f"   🔸 Failure Type: {TermColors.color_text(failure_type, TermColors.CYAN)}")
            confidence_str = f"{confidence:.2f}"
            print(f"   🔸 Confidence: {TermColors.color_text(confidence_str, TermColors.GREEN)}")
            print(f"   🔸 Recommended Target: {TermColors.color_text(target, TermColors.YELLOW)}")
            print(f"   🔸 Severity: {classification.get('failure_severity', 'N/A')}")
            
            if classification.get("reasoning"):
                print(f"   🧠 LLM Reasoning:")
                for j, reason in enumerate(classification["reasoning"], 1):
                    print(f"      {j}. {reason}")
            
            if classification.get("specific_issues"):
                issues = classification["specific_issues"]
                if issues.get("planning_issues"):
                    print(f"   📋 Planning Issues:")
                    for issue in issues["planning_issues"]:
                        print(f"      - {issue}")
                        
                if issues.get("execution_issues"):
                    print(f"   ⚙️  Execution Issues:")
                    for issue in issues["execution_issues"]:
                        print(f"      - {issue}")
            
            if classification.get("healing_recommendations"):
                print(f"   💡 Healing Recommendations:")
                for rec in classification["healing_recommendations"]:
                    print(f"      - {rec}")
    else:
        print(f"\n⚠️  No failure classification occurred (task may have succeeded initially)")
    
    # Workflow Evolution Analysis
    print(f"\n📈 WORKFLOW EVOLUTION ANALYSIS:")
    print("=" * 60)
    
    for i, phase in enumerate(result['workflow_phases'], 1):
        phase_name = phase.get('phase', 'UNKNOWN')
        print(f"\n🔍 Phase {i}: {TermColors.color_text(phase_name, TermColors.HEADER)}")
        
        if phase_name == "INITIAL_PLANNING_AND_VALIDATION":
            plan_valid = phase.get('plan_validation_passed', False)
            plan_score = phase.get('plan_validation_result', {}).get('quality_score', 0.0)
            
            status_text = 'PASSED' if plan_valid else 'FAILED'
            status_color = TermColors.GREEN if plan_valid else TermColors.FAIL
            print(f"   📋 Plan Validation: {TermColors.color_text(status_text, status_color)} (Score: {plan_score:.2f})")
            
            if phase.get('plan_validation_result', {}).get('issues'):
                print(f"   ⚠️  Plan Issues:")
                for issue in phase['plan_validation_result']['issues']:
                    print(f"      - {issue}")
            
        elif phase_name == "DIRECT_FIX":
            fix_successful = phase.get('direct_fix_successful', False)
            fix_text = 'SUCCESSFUL' if fix_successful else 'FAILED'
            fix_color = TermColors.GREEN if fix_successful else TermColors.FAIL
            print(f"   🔨 Direct Fix: {TermColors.color_text(fix_text, fix_color)}")
            
            if phase.get('direct_fix_score'):
                print(f"   📊 Direct Fix Score: {phase['direct_fix_score']:.2f}")
                
        elif phase_name.startswith("HEALING_ITERATION"):
            healing_successful = phase.get('healing_successful', False)
            healing_target = phase.get('healing_target', 'UNKNOWN')
            
            print(f"   🎯 Target: {TermColors.color_text(healing_target, TermColors.CYAN)}")
            
            success_text = 'YES' if healing_successful else 'NO'
            success_color = TermColors.GREEN if healing_successful else TermColors.FAIL
            print(f"   ✅ Success: {TermColors.color_text(success_text, success_color)}")
            
            if phase.get('improved_score'):
                print(f"   📈 Improved Score: {phase['improved_score']:.2f}")
    
    # Final Assessment
    print(f"\n🎯 FINAL ASSESSMENT:")
    print("=" * 60)
    
    if result['total_healing_iterations'] > 0:
        print(f"✅ SUCCESS: LLM-based classification system was triggered!")
        print(f"🧠 The LLM analyzed failures and made {result['total_healing_iterations']} healing decision(s)")
        
        if result['healing_breakdown']['planner_healings'] > 0:
            print(f"🧠 Planner healing was recommended and applied {result['healing_breakdown']['planner_healings']} time(s)")
        
        if result['healing_breakdown']['executor_healings'] > 0:
            print(f"⚙️  Executor healing was recommended and applied {result['healing_breakdown']['executor_healings']} time(s)")
            
        print(f"📊 Final outcome: {final_status} with score {result['final_score']:.2f}")
        
    else:
        print(f"⚠️  LLM classification was not triggered (task succeeded initially)")
        print(f"💡 Try a more challenging task or adjust the planner prompt to be more problematic")
    
    print(f"\n🎉 LLM-BASED CLASSIFICATION TEST COMPLETE!")
    return result

if __name__ == "__main__":
    main() 