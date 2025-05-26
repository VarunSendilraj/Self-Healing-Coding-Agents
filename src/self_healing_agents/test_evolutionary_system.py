#!/usr/bin/env python3
"""
Test script for the evolutionary prompt optimization system.

This script demonstrates the complete evolutionary self-healing workflow:
1. Initialize agents with poor prompts
2. Run initial task (expected to fail)
3. Apply evolutionary optimization
4. Show improvement results
"""

import os
import sys
import logging
from pathlib import Path

# Add src to path
current_dir = Path(__file__).parent
src_dir = current_dir.parent
sys.path.insert(0, str(src_dir))

from self_healing_agents.agents import Planner, Executor, Critic
from self_healing_agents.llm_service import LLMService
from self_healing_agents.prompts import BAD_PLANNER_PROMPT, ULTRA_BUGGY_PROMPT
from self_healing_agents.evaluation.evolutionary_enhanced_harness import (
    run_evolutionary_multi_agent_task,
    test_evolutionary_healing_with_real_llm
)


def test_simple_evolutionary_optimization():
    """Test basic evolutionary optimization functionality."""
    
    print("🧬 SIMPLE EVOLUTIONARY OPTIMIZATION TEST")
    print("=" * 50)
    
    # Setup environment
    provider = os.environ.get("LLM_PROVIDER", "deepseek")
    model_name = os.environ.get("LLM_MODEL", "deepseek-coder")
    
    try:
        llm_service = LLMService(provider=provider, model_name=model_name)
        print(f"✅ LLM Service initialized: {provider}/{model_name}")
    except Exception as e:
        print(f"❌ Failed to initialize LLM service: {e}")
        print("💡 Make sure LLM_PROVIDER and LLM_MODEL environment variables are set")
        return False
    
    # Test evolutionary components individually
    print("\n🔧 Testing evolutionary components...")
    
    try:
        from self_healing_agents.evolution import (
            EvolutionaryPromptOptimizer,
            EvolutionConfig,
            create_executor_optimizer
        )
        
        # Create a simple test
        config = EvolutionConfig.fast_mode()  # Quick test
        optimizer = EvolutionaryPromptOptimizer(llm_service, config)
        
        # Test with a simple prompt
        test_prompt = "You are a Python programmer. Write code."
        
        print(f"🧬 Running quick optimization test...")
        print(f"   Original prompt: {test_prompt}")
        print(f"   Config: {config.population_size} individuals, {config.max_generations} generations")
        
        # Run optimization
        results = optimizer.optimize_prompt(
            base_prompt=test_prompt,
            agent_type="EXECUTOR"
        )
        
        print(f"\n✅ OPTIMIZATION COMPLETE:")
        print(f"   🏆 Best fitness: {results.best_fitness:.3f}")
        print(f"   🔄 Generations: {results.generation_count}")
        print(f"   📊 Evaluations: {results.evaluation_count}")
        print(f"   ⏱️  Time: {results.execution_time:.1f}s")
        print(f"   🛑 Reason: {results.termination_reason}")
        
        if results.best_fitness > 0:
            print(f"   ✨ Evolved prompt length: {len(results.best_prompt)} chars")
            
        return True
        
    except Exception as e:
        print(f"❌ Component test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_evolutionary_workflow():
    """Test the complete evolutionary self-healing workflow."""
    
    print("\n🔄 FULL EVOLUTIONARY WORKFLOW TEST")
    print("=" * 50)
    
    # This runs the comprehensive test
    try:
        result = test_evolutionary_healing_with_real_llm()
        
        if result:
            print(f"\n🎉 WORKFLOW TEST COMPLETED")
            print(f"   Status: {result['final_status']}")
            print(f"   Score: {result['final_score']:.2f}")
            return True
        else:
            print(f"\n❌ WORKFLOW TEST FAILED")
            return False
            
    except Exception as e:
        print(f"❌ Workflow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🚀 EVOLUTIONARY PROMPT OPTIMIZATION SYSTEM TEST")
    print("=" * 60)
    print("This test demonstrates the evolutionary self-healing system")
    print("=" * 60)
    
    # Check environment
    if not os.environ.get("LLM_PROVIDER"):
        print("⚠️  LLM_PROVIDER not set, using default: deepseek")
        os.environ["LLM_PROVIDER"] = "deepseek"
    
    if not os.environ.get("LLM_MODEL"):
        print("⚠️  LLM_MODEL not set, using default: deepseek-coder") 
        os.environ["LLM_MODEL"] = "deepseek-coder"
    
    success_count = 0
    total_tests = 2
    
    # Test 1: Simple component test
    print(f"\n📝 TEST 1/2: Component Testing")
    if test_simple_evolutionary_optimization():
        success_count += 1
        print(f"✅ Component test PASSED")
    else:
        print(f"❌ Component test FAILED")
    
    # Test 2: Full workflow test
    print(f"\n📝 TEST 2/2: Full Workflow Testing")
    if test_full_evolutionary_workflow():
        success_count += 1
        print(f"✅ Workflow test PASSED")
    else:
        print(f"❌ Workflow test FAILED")
    
    # Final results
    print(f"\n" + "=" * 60)
    print(f"🏁 EVOLUTIONARY SYSTEM TEST RESULTS")
    print(f"=" * 60)
    print(f"✅ Tests passed: {success_count}/{total_tests}")
    print(f"📊 Success rate: {(success_count/total_tests)*100:.1f}%")
    
    if success_count == total_tests:
        print(f"🎉 ALL TESTS PASSED! Evolutionary system is working correctly.")
        return 0
    else:
        print(f"❌ Some tests failed. Check the logs above for details.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 