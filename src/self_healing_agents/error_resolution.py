from dataclasses import dataclass, field
import re
import time
from typing import Dict, List, Optional, Tuple, Any


@dataclass
class SimpleFixAttempt:
    """
    Represents a single attempt to fix an error by appending error details to an executor prompt.
    Tracks the original prompt, modified prompt, whether the fix was successful, and a timestamp.
    """
    error_details: Dict[str, str]
    original_prompt: str
    modified_prompt: str
    successful: bool = False
    timestamp: float = field(default_factory=time.time)


class SimpleErrorResolver:
    """
    Handles simple error resolution by appending error information to executor prompts.
    This is the first stage in the error resolution process before triggering more advanced
    self-healing mechanisms.
    """
    
    def __init__(self):
        """Initialize the resolver with an empty list of fix attempts."""
        self.fix_attempts: List[SimpleFixAttempt] = []
    
    def format_error_for_prompt(self, error_details: Dict[str, str], error_context: Optional[str] = None, original_plan_text: Optional[str] = None) -> str:
        """
        Formats error details into a clear, structured prompt addition for the executor.
        
        Args:
            error_details: A dictionary containing error information (type, message, traceback).
            error_context: Optional contextual code around the error location.
            original_plan_text: Optional text of the original plan/prompt that led to the error.
        
        Returns:
            A formatted string to append to the executor prompt.
        """
        sections = ["Your previous attempt to generate code resulted in the following error:"]
        sections.append("\n### ERROR DETAILS ###")
        
        error_type = error_details.get('error_type')
        error_message = error_details.get('error_message')
        traceback_content = error_details.get('traceback') or error_details.get('stderr', '')

        if error_type:
            sections.append(f"Error Type: {error_type}")
        if error_message:
            sections.append(f"Error Message: {error_message}")
        if traceback_content:
            sections.append(f"Traceback:\n{traceback_content}")
        
        if error_context:
            # Try to get line number for more specific context message
            line_number_info = ""
            line_matches = re.findall(r'line (\d+)', traceback_content)
            if line_matches:
                try:
                    line_number_info = f" (around line {line_matches[-1]})"
                except (ValueError, IndexError):
                    pass
            sections.append(f"\nCode Snippet with Error{line_number_info}:")
            sections.append("```python")
            sections.append(error_context)
            sections.append("```")
        
        sections.append("\n### TASK: CORRECT THE CODE ###")
        if original_plan_text:
            sections.append("The original plan/request was:")
            sections.append(f"'''{original_plan_text}'''") # Enclose in triple quotes for clarity

        # Instructions for fixing, with specific guidance for common error types
        if error_type == 'SyntaxError' and 'never closed' in error_message:
            sections.append("\nThis is a syntax error where a parenthesis, bracket, or quote is not properly closed.")
            sections.append("For example, if you have:")
            sections.append("```python")
            sections.append("print('Hello, World!'")
            sections.append("```")
            sections.append("You need to add the missing closing parenthesis like this:")
            sections.append("```python")
            sections.append("print('Hello, World!')")
            sections.append("```")
            
            # Add additional emphasis for this specific error
            sections.append("\n*** IMPORTANT: YOUR TASK IS TO ADD THE MISSING CLOSING PARENTHESIS ***")
            sections.append("The syntax error in the code is that a parenthesis is NOT closed.")
            sections.append("Check each line carefully and ensure EVERY opening parenthesis has a matching closing parenthesis.")
            sections.append("In Python, each opening '(' must have a corresponding closing ')'.")
            sections.append("\nEXAMPLE FIX:")
            sections.append("Original broken code: print('Hello, World!'")
            sections.append("Fixed code: print('Hello, World!')")
            
            # Add the exact steps the LLM should take
            sections.append("\n🚩 SPECIFIC ACTION REQUIRED:")
            sections.append("1. Look at the print statement in the code: print('Hello, World!'")
            sections.append("2. Add a closing parenthesis to the end: print('Hello, World!')")
            sections.append("3. Return the complete fixed function (not just the line):")
            sections.append("   def greet():")
            sections.append("       print('Hello, World!')")
        elif error_type == 'NameError':
            sections.append("\nThis is a name error where a variable is used but not defined.")
            sections.append("Check for typos in variable names or missing variable definitions.")
        
        sections.append("\nPlease analyze the error and fix the code.")
        sections.append("Make sure all parentheses, brackets, and quotes are properly closed.")
        sections.append("Output ONLY the complete, corrected Python code block.")
        sections.append("Do not include markdown code fences (```), explanations, or any text other than the Python code itself.")
        
        return "\n".join(sections)
    
    def extract_error_context(self, code: str, error_details: Dict[str, str]) -> Optional[str]:
        """
        Extracts relevant code context around the error location from the traceback or stderr.
        
        Args:
            code: The original code string.
            error_details: A dictionary containing error information.
            
        Returns:
            A string with relevant code context or None if context can't be determined.
        """
        if not code:
            return None
        
        lines = code.split('\n')
        line_number = None
        
        # Try to extract line number from traceback or stderr
        traceback_text = error_details.get('traceback') or error_details.get('stderr') or ""
        
        # Look for "line X" patterns in the traceback
        line_matches = re.findall(r'line (\d+)', traceback_text)
        if line_matches:
            # Get the line number from the last occurrence, as that's usually the actual error line
            try:
                line_number = int(line_matches[-1])
            except (ValueError, IndexError):
                return None
        
        if line_number is None or line_number <= 0 or line_number > len(lines):
            return None
            
        # Get context: a few lines before and after the error line
        context_lines = []
        start_line = max(1, line_number - 2)
        end_line = min(len(lines), line_number + 2)
        
        for i in range(start_line, end_line + 1):
            prefix = ">>> " if i == line_number else "    "
            line_content = lines[i-1] if i <= len(lines) else ""
            context_lines.append(f"{i}: {prefix}{line_content}")
        
        return "\n".join(context_lines)
    
    def append_error_to_prompt(self, original_prompt: str, 
                              error_details: Dict[str, str], 
                              code: Optional[str] = None) -> Tuple[str, SimpleFixAttempt]:
        """
        Appends formatted error information to an executor prompt.
        Original_prompt here refers to the textual content that led to the error (e.g. plan steps text)
        
        Args:
            original_prompt: The original textual prompt/plan steps that led to the error.
            error_details: A dictionary containing error information.
            code: Optional code string for context extraction.
            
        Returns:
            A tuple containing:
            - The modified textual prompt content with appended error information
            - A SimpleFixAttempt object tracking this fix attempt
        """
        error_context = None
        if code:
            error_context = self.extract_error_context(code, error_details)
        
        # Pass original_prompt as original_plan_text to format_error_for_prompt
        formatted_error_instructions = self.format_error_for_prompt(error_details, error_context, original_plan_text=original_prompt)
        
        # The new "prompt" for the executor is essentially these detailed instructions for fixing.
        # The executor's main system prompt will still guide its general behavior.
        # This formatted_error_instructions becomes the new "plan steps" or task for the executor.
        modified_prompt_content = formatted_error_instructions 
        # We are no longer appending to the original_prompt string in a simple way;
        # formatted_error_instructions IS the new prompt content for the fix.
        
        fix_attempt = SimpleFixAttempt(
            error_details=error_details,
            original_prompt=original_prompt, # This was the plan text that caused the error
            modified_prompt=modified_prompt_content # This is the new detailed fix instruction
        )
        self.fix_attempts.append(fix_attempt)
        
        return modified_prompt_content, fix_attempt
    
    def record_fix_success(self, fix_attempt: SimpleFixAttempt, success: bool) -> None:
        """
        Records whether a fix attempt was successful.
        
        Args:
            fix_attempt: The SimpleFixAttempt to update.
            success: Whether the fix was successful.
        """
        fix_attempt.successful = success
    
    def get_success_rate(self) -> float:
        """
        Calculates the success rate of fix attempts.
        
        Returns:
            A float between 0.0 and 1.0 representing the fraction of successful fixes.
        """
        if not self.fix_attempts:
            return 0.0
        
        successful_attempts = sum(1 for attempt in self.fix_attempts if attempt.successful)
        return successful_attempts / len(self.fix_attempts)
    
    def get_fix_history(self) -> List[Dict[str, Any]]:
        """
        Returns a history of fix attempts in a structured format.
        
        Returns:
            A list of dictionaries containing information about each fix attempt.
        """
        history = []
        for attempt in self.fix_attempts:
            history.append({
                "error_type": attempt.error_details.get("error_type", "Unknown"),
                "error_message": attempt.error_details.get("error_message", ""),
                "timestamp": attempt.timestamp,
                "successful": attempt.successful,
                "prompt_modified": attempt.original_prompt != attempt.modified_prompt
            })
        return history 