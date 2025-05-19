from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Defined statuses based on PRD Task 2.7
CRITIC_STATUS_SUCCESS = "SUCCESS"
CRITIC_STATUS_FAILURE_SYNTAX = "FAILURE_SYNTAX"
CRITIC_STATUS_FAILURE_RUNTIME = "FAILURE_RUNTIME"
CRITIC_STATUS_FAILURE_LOGIC = "FAILURE_LOGIC"
CRITIC_STATUS_FAILURE_EVALUATION = "FAILURE_EVALUATION"


@dataclass
class CriticReport:
    """
    Structured report from the Critic agent, detailing the evaluation of generated code.
    """
    status: str  # e.g., SUCCESS, FAILURE_SYNTAX, FAILURE_RUNTIME, FAILURE_LOGIC
    score: float  # Quantifiable score (0.0 to 1.0) as per PRD Task 2.6
    
    error_details: Optional[Dict[str, str]] = None # E.g., {"type": "SyntaxError", "message": "...", "traceback": "..."}
    test_results: Optional[List[Dict[str, Any]]] = field(default_factory=list) # E.g., [{"input": ..., "expected": ..., "actual": ..., "pass": True/False}]
    summary: Optional[str] = None # Concise summary of issues
    
    # Additional fields found in actual Critic output
    execution_stdout: Optional[str] = None
    execution_stderr: Optional[str] = None
    generated_test_specifications: Optional[List[Dict[str, Any]]] = field(default_factory=list)
    function_to_test: Optional[str] = None
    generated_code_for_report: Optional[str] = None

    def is_failure(self) -> bool:
        """
        Determines if the report indicates a failure.
        """
        return self.status != CRITIC_STATUS_SUCCESS 