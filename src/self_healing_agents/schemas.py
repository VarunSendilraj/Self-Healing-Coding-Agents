from dataclasses import field # Removed dataclass as we are moving to Pydantic
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator # Added Pydantic imports

from src.self_healing_agents.error_types import AgentType, ErrorType # Import new enums

# Defined statuses based on PRD Task 2.7
CRITIC_STATUS_SUCCESS = "SUCCESS"
CRITIC_STATUS_FAILURE_SYNTAX = "FAILURE_SYNTAX"
CRITIC_STATUS_FAILURE_RUNTIME = "FAILURE_RUNTIME"
CRITIC_STATUS_FAILURE_LOGIC = "FAILURE_LOGIC"
CRITIC_STATUS_FAILURE_EVALUATION = "FAILURE_EVALUATION"


class TaskDefinition(BaseModel):
    """
    Defines a task for the agent system to accomplish.
    """
    description: str  # Description of the task
    task_id: str  # Unique identifier for the task
    
    # Optional additional parameters
    context: Optional[Dict[str, Any]] = None  # Additional context for the task
    requirements: List[str] = Field(default_factory=list)  # Specific requirements for the task
    constraints: List[str] = Field(default_factory=list)  # Constraints for task execution

    @property
    def full_description(self) -> str:
        """
        Returns the full task description including any requirements and constraints.
        """
        desc = self.description
        
        if self.requirements:
            desc += "\n\nRequirements:\n" + "\n".join([f"- {req}" for req in self.requirements])
            
        if self.constraints:
            desc += "\n\nConstraints:\n" + "\n".join([f"- {con}" for con in self.constraints])
            
        return desc


class CriticReport(BaseModel): # Changed from @dataclass to Pydantic BaseModel
    """
    Structured report from the Critic agent, detailing the evaluation of generated code.
    Now includes error classification fields from Phase 2.
    """
    status: str  # e.g., SUCCESS, FAILURE_SYNTAX, FAILURE_RUNTIME, FAILURE_LOGIC
    score: float = Field(..., ge=0.0, le=1.0)  # Quantifiable score (0.0 to 1.0)
    
    error_details: Optional[Dict[str, str]] = None # E.g., {"type": "SyntaxError", "message": "...", "traceback": "..."}
    test_results: List[Dict[str, Any]] = Field(default_factory=list) # E.g., [{"input": ..., "expected": ..., "actual": ..., "pass": True/False}]
    summary: Optional[str] = None # Concise summary of issues
    
    # Additional fields found in actual Critic output from Phase 1
    execution_stdout: Optional[str] = None
    execution_stderr: Optional[str] = None
    generated_test_specifications: List[Dict[str, Any]] = Field(default_factory=list)
    function_to_test: Optional[str] = None
    generated_code_for_report: Optional[str] = None

    # --- Phase 2: Error Classification Fields ---
    error_type: Optional[ErrorType] = None
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    suggested_agent: Optional[AgentType] = None
    reasoning: Optional[str] = None

    def is_failure(self) -> bool:
        """
        Determines if the report indicates a failure.
        """
        return self.status != CRITIC_STATUS_SUCCESS

    @field_validator('error_type', mode='before')
    @classmethod
    def _validate_error_type(cls, value):
        if value is None: return value
        if isinstance(value, ErrorType):
            return value
        try:
            return ErrorType(value)
        except ValueError:
            raise ValueError(f"Invalid error_type: {value}. Must be a valid ErrorType enum member or string.")

    @field_validator('suggested_agent', mode='before')
    @classmethod
    def _validate_suggested_agent(cls, value):
        if value is None: return value
        if isinstance(value, AgentType):
            return value
        try:
            return AgentType(value)
        except ValueError:
            raise ValueError(f"Invalid suggested_agent: {value}. Must be a valid AgentType enum member or string.")
    
    @model_validator(mode='after')
    def _check_classification_fields_consistency(self):
        classification_fields = [self.error_type, self.confidence, self.suggested_agent, self.reasoning]
        any_classification_field_present = any(field is not None for field in classification_fields)
        all_classification_fields_present = all(field is not None for field in classification_fields)

        if not self.is_failure(): # Success Report
            if any_classification_field_present:
                raise ValueError(
                    "Error classification fields (error_type, confidence, suggested_agent, reasoning) must not be present for success reports."
                )
        else: # Failure Report
            if any_classification_field_present and not all_classification_fields_present:
                raise ValueError(
                    "If any of error_type, confidence, suggested_agent, or reasoning are provided for a failure, all must be provided."
                )
            # Confidence range is already validated by Field(ge=0.0, le=1.0) on the confidence field itself.
            # No need for: if all_classification_fields_present and not (0.0 <= self.confidence <= 1.0):

        return self 