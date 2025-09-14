#!/usr/bin/env python
"""
Test script to verify that enhanced_harness_integration.py is using subprocess execution.
"""

import os
import sys
import tempfile
import json

# Add the src directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

def create_test_task():
    """Create a test task with file I/O to verify subprocess execution."""
    task = {
        "id": "subprocess_test_task",
        "description": "Write a function that creates a file and returns a number",
        "initial_executor_prompt": "Implement a function called `create_file_and_return` that creates a file named 'harness_subprocess_test.txt' with the current process ID and returns 42."
    }
    return task

def main():
    """Main test function."""
    try:
        # Import required modules
        from self_healing_agents.evaluation import enhanced_harness
        from self_healing_agents.evaluation.code_analyzer_integration import integrate_code_analyzer_with_harness
        from self_healing_agents.agents import Planner, Executor, Critic
        from self_healing_agents.llm_service import LLMService, LLMServiceError
        
        # Create a temporary task file
        test_task = create_test_task()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as task_file:
            json.dump([test_task], task_file)
            task_file_path = task_file.name
            
        print(f"Created test task file at {task_file_path}")
        
        # Patch the enhanced_harness module
        integrate_code_analyzer_with_harness(enhanced_harness)
        
        # Run the task
        print("Running test task through enhanced_harness...")
        
        # Create minimal required instances
        class MockLLMService(LLMService):
            def __init__(self):
                # Override the __init__ to avoid needing API keys
                self.provider = "mock"
                self.model_name = "mock-model"
                self.temperature = 0.7
                self.max_tokens = 1024
                self.api_key = "mock-key"
                self._client = None
                
            def invoke(self, messages, expect_json=False):
                # Just return a fixed response based on the type of message
                if any("implement" in str(msg.get("content", "")).lower() for msg in messages if isinstance(msg, dict)):
                    return """```python
def create_file_and_return():
    import os
    
    # Get process ID
    pid = os.getpid()
    
    # Create file
    with open('harness_subprocess_test.txt', 'w') as f:
        f.write(f"Process ID: {pid}\\n")
        f.write(f"This file was created by the harness test\\n")
    
    return 42
```"""
                elif expect_json:
                    return {"status": "SUCCESS", "message": "Mock response"}
                else:
                    return "This is a mock response"
                    
            def _prepare_messages(self, messages):
                return messages  # Just return as is, not used in our mock
        
        llm_service = MockLLMService()
        planner = Planner("TestPlanner", llm_service)
        executor = Executor("TestExecutor", llm_service)
        critic = Critic("TestCritic", llm_service)
        
        # Run a single task
        result = enhanced_harness.run_single_task(test_task, planner, executor, critic, llm_service)
        print("Task result:", result)
        
        # Check if the file was created
        if os.path.exists('harness_subprocess_test.txt'):
            print("\nSUCCESS: File was created by the subprocess through the harness")
            with open('harness_subprocess_test.txt', 'r') as f:
                print("\nContents of harness_subprocess_test.txt:")
                print(f.read())
        else:
            print("\nFAILURE: No file was created, code might be running in a sandbox")
        
        # Clean up
        if os.path.exists(task_file_path):
            os.unlink(task_file_path)
            
    except Exception as e:
        print(f"Error running test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 