from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .error_types import AgentType, ErrorType

# Defined statuses based on PRD Task 2.7
CRITIC_STATUS_SUCCESS = "SUCCESS"
CRITIC_STATUS_FAILURE_SYNTAX = "FAILURE_SYNTAX"
CRITIC_STATUS_FAILURE_RUNTIME = "FAILURE_RUNTIME"
CRITIC_STATUS_FAILURE_LOGIC = "FAILURE_LOGIC"
CRITIC_STATUS_FAILURE_EVALUATION = "FAILURE_EVALUATION"


@dataclass
class TaskDefinition:
    """Definition of a task to be completed by the agents."""
    description: str  # Description of the task
    expected_output: Optional[str] = None  # Expected output or example solution
    test_cases: List[Dict[str, Any]] = field(default_factory=list)  # List of test cases to validate the solution


@dataclass
class PlannerOutput:
    """Output from the Planner agent."""
    plan: List[str]  # List of steps in the plan
    task_definition: TaskDefinition  # The task this plan is for


@dataclass
class ExecutorOutput:
    """Output from the Executor agent."""
    code: str  # The generated code
    planner_output: PlannerOutput  # The plan this code is based on
    errors: Optional[str] = None  # Any errors encountered during generation
    traceback: Optional[str] = None  # Traceback for errors
    stdout: str = ""  # Standard output from execution
    stderr: str = ""  # Standard error from execution


@dataclass
class CriticReport:
    """
    Structured report from the Critic agent, detailing the evaluation of generated code.
    Now includes fields for error classification.
    """
    overall_status: str  # e.g., SUCCESS, FAILURE_SYNTAX, FAILURE_RUNTIME, FAILURE_LOGIC
    score: float  # Quantifiable score (0.0 to 1.0) as per PRD Task 2.6
    
    feedback: str  # Concise feedback message
    error_details: Optional[str] = None  # Detailed error information
    test_summary: Dict[str, Any] = field(default_factory=dict)  # Summary of test results
    raw_code: str = ""  # The code being evaluated
    
    # New fields for Phase 2: Error Classification
    error_type: Optional[ErrorType] = None
    confidence: Optional[float] = None  # Classification confidence (0.0-1.0)
    suggested_agent: Optional[AgentType] = None
    reasoning: Optional[str] = None  # Justification for the classification

    def __post_init__(self):
        """Post-initialization validation."""
        if self.confidence is not None:
            if not (0.0 <= self.confidence <= 1.0):
                raise ValueError("Confidence score must be between 0.0 and 1.0.")
        if self.error_type is not None and not isinstance(self.error_type, ErrorType):
            raise TypeError(f"error_type must be an instance of ErrorType, got {type(self.error_type)}")
        if self.suggested_agent is not None and not isinstance(self.suggested_agent, AgentType):
            raise TypeError(f"suggested_agent must be an instance of AgentType, got {type(self.suggested_agent)}")

    def is_failure(self) -> bool:
        """
        Determines if the report indicates a failure.
        """
        return self.overall_status != CRITIC_STATUS_SUCCESS


@dataclass
class EnhancedCriticReport(CriticReport):
    """
    Enhanced version of the CriticReport with additional fields for error classification.
    This class inherits from CriticReport to maintain backward compatibility.
    """
    # EnhancedCriticReport inherits all fields from CriticReport and adds no new required fields.
    # The error_type, confidence, suggested_agent, and reasoning fields are already defined in CriticReport.
    # This class exists to provide a clear distinction between the basic and enhanced report types.
    
    # Adding fields from the refactored evaluate_code in agents.py
    evaluator_name: str = "UnknownCritic"
    task_description: Optional[str] = None
    generated_code: Optional[str] = None # Renamed from raw_code for clarity if it refers to the input code
    raw_execution_details: Dict[str, Any] = field(default_factory=dict)
    test_generation_details: Dict[str, Any] = field(default_factory=dict)
    test_execution_results: List[Dict[str, Any]] = field(default_factory=list)
    error_classification: Dict[str, Any] = field(default_factory=dict)
    # overall_status, feedback, score are inherited from CriticReport

    def __post_init__(self):
        super().__post_init__() # Call parent's post_init if it has one
        # If raw_code was meant to be generated_code, ensure it's set or handled.
        # The generated_code field is new here, raw_code is in parent.
        # For now, we assume they might be distinct or one might supersede the other.
        # If generated_code is provided and raw_code is empty, maybe populate raw_code?
        if self.generated_code and not self.raw_code:
            self.raw_code = self.generated_code
        elif self.raw_code and not self.generated_code: # If old field raw_code is used, populate new one
            self.generated_code = self.raw_code


pass 