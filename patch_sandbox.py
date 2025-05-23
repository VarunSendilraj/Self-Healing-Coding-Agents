"""
Patch script to add enumerate function to the sandbox environment.
This script monkey patches the _execute_sandboxed_code method in the Critic class
to include enumerate in the allowed builtins.
"""

import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import the Critic class
from self_healing_agents.agents import Critic

# Save the original method
original_execute_sandboxed_code = Critic._execute_sandboxed_code

# Define the patched method
def patched_execute_sandboxed_code(self, code_string):
    """
    Patched version of _execute_sandboxed_code that adds enumerate to allowed builtins.
    """
    # Call the original method to get the result dictionary
    result = original_execute_sandboxed_code(self, code_string)
    
    # Add enumerate to the allowed builtins
    # This won't affect the current result but will be available for future calls
    if hasattr(self, "_allowed_globals"):
        if "__builtins__" in self._allowed_globals:
            self._allowed_globals["__builtins__"]["enumerate"] = enumerate
    
    # Also patch the global allowed_globals dictionary used in the original method
    try:
        # This is a bit hacky, but it directly modifies the code in the method
        frame = sys._getframe(1)
        if "allowed_globals" in frame.f_locals:
            if "__builtins__" in frame.f_locals["allowed_globals"]:
                frame.f_locals["allowed_globals"]["__builtins__"]["enumerate"] = enumerate
                print("Added enumerate to allowed builtins in current frame")
    except Exception as e:
        print(f"Warning: Could not patch frame locals: {e}")
    
    return result

# Apply the monkey patch
Critic._execute_sandboxed_code = patched_execute_sandboxed_code

print("Sandbox environment patched: Added enumerate to allowed builtins.") 