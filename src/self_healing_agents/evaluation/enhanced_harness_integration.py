#!/usr/bin/env python
"""
Integration script for patching the enhanced_harness.py to use our code analyzer system.
"""

import sys
import os
import importlib.util

# Add the src directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(os.path.dirname(current_dir))
project_root = os.path.dirname(src_dir)

if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import our code analyzer integration
from self_healing_agents.evaluation.code_analyzer_integration import integrate_code_analyzer_with_harness
# Import the dumbified prompts
from self_healing_agents.prompts import ULTRA_BUGGY_PROMPT as BROKEN_CODE_PROMPT

def main():
    """
    Main function to patch the enhanced_harness.py and run it with our code analyzer.
    """
    try:
        # Import the enhanced_harness module
        from self_healing_agents.evaluation import enhanced_harness
        
        # Patch the enhanced_harness module with our code analyzer
        integrate_code_analyzer_with_harness(enhanced_harness)
        
        # Add extra check to ensure we're not doing unnecessary self-healing iterations
        original_run_single_task = enhanced_harness.run_single_task
        
        def patched_run_single_task(*args, **kwargs):
            """
            A patched version of run_single_task that ensures we properly handle the case
            where initial code passes evaluation and doesn't need self-healing.
            """
            result = original_run_single_task(*args, **kwargs)
            
            # Verify that if initial code passed, we marked need_self_healing = False
            if (result.get("final_status") == "SUCCESS" and 
                result.get("final_source") == "INITIAL" and
                result.get("total_iterations", 0) > 0):
                print("WARNING: Initial code passed but self-healing iterations were recorded.")
                print("This is a bug in the harness flow control. Check the return logic in enhanced_harness.py.")
                
                # Ensure iterations are set to 0 for accurate reporting
                result["total_iterations"] = 0
            
            return result
        
        # Apply our patch
        enhanced_harness.run_single_task = patched_run_single_task
        
        # Patch the Executor to use a dumbified prompt to trigger self-healing
        original_executor_init = enhanced_harness.Executor.__init__
        
        def patched_executor_init(self, *args, **kwargs):
            # Call the original init
            original_executor_init(self, *args, **kwargs)
            
            print(f"🔧 ORIGINAL INIT: Executor '{self.name}' initialized with system_prompt: '{self.system_prompt[:70]}...'")
            
            # Override the system prompt to use our dumbified version
            print(f"🔧 PATCHING EXECUTOR: Using dumbified prompt to trigger self-healing behavior")
            self.set_prompt(BROKEN_CODE_PROMPT)
            
            # Log the actual prompt being used
            print("=" * 80)
            print("📝 EXECUTOR PROMPT AFTER PATCHING:")
            print("=" * 80)
            print(BROKEN_CODE_PROMPT)
            print("=" * 80)
        
        # Patch the Executor.set_prompt method to add more logging
        original_set_prompt = enhanced_harness.Executor.set_prompt
        
        def patched_set_prompt(self, new_prompt):
            print(f"🔄 PROMPT CHANGE: Executor '{self.name}' system_prompt changing from:")
            print(f"   OLD: '{self.system_prompt[:70]}...'")
            print(f"   NEW: '{new_prompt[:70]}...'")
            
            # Call the original method
            original_set_prompt(self, new_prompt)
            
            print(f"✅ PROMPT SET: Executor '{self.name}' system_prompt now: '{self.system_prompt[:70]}...'")
        
        # Patch the Executor.run method to log what prompt is being used
        original_executor_run = enhanced_harness.Executor.run
        
        def patched_executor_run(self, plan, original_request):
            print(f"🚀 EXECUTOR RUN: Starting with prompt: '{self.system_prompt[:70]}...'")
            print(f"📋 EXECUTOR RUN: Plan: {plan}")
            print(f"📝 EXECUTOR RUN: Original request: {original_request[:100]}...")
            
            # Call the original method
            result = original_executor_run(self, plan, original_request)
            
            print(f"✅ EXECUTOR RUN: Generated code length: {len(result)} characters")
            print(f"🔍 EXECUTOR RUN: Code preview: {result[:100]}...")
            
            return result
        
        # Apply all the patches
        enhanced_harness.Executor.__init__ = patched_executor_init
        enhanced_harness.Executor.set_prompt = patched_set_prompt  
        enhanced_harness.Executor.run = patched_executor_run
        
        # Also patch the agent creation in enhanced_harness to log when agents are created
        original_main = enhanced_harness.main_enhanced_evaluation_harness
        
        def patched_main():
            print("🏗️  STARTING ENHANCED HARNESS MAIN FUNCTION")
            
            # Call original main, but we'll intercept agent creation
            original_main()
            
        enhanced_harness.main_enhanced_evaluation_harness = patched_main
        
        # Run the enhanced_harness main function
        print("Running the Enhanced Evaluation Harness with CodeAnalyzer integration...")
        print("🎯 Using DUMBIFIED executor prompt to trigger self-healing mechanisms!")
        print("📊 COMPREHENSIVE PROMPT TRACING ENABLED!")
        enhanced_harness.main_enhanced_evaluation_harness()
        
    except ImportError as e:
        print(f"Error importing the enhanced_harness module: {e}")
        print("Make sure you are running this script from the project root directory.")
        sys.exit(1)
    except Exception as e:
        print(f"Error integrating CodeAnalyzer with the Enhanced Evaluation Harness: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 