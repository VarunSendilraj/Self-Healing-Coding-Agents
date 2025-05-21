from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List

from pydantic import BaseModel, Field

from self_healing_agents.error_types import AgentType, ErrorType

class ClassificationResult(BaseModel):
    """Data class to hold the result of an error classification."""
    error_type: ErrorType
    confidence: float = Field(..., ge=0.0, le=1.0)
    suggested_agent: AgentType
    reasoning: str
    # Optional: Add any raw output from the classifier if needed for debugging
    raw_classifier_output: Optional[Any] = None 

class ErrorClassifier(ABC):
    """
    Abstract Base Class for error classification.
    
    Concrete implementations will provide specific strategies for classifying errors
    (e.g., LLM-based, rule-based, etc.).
    """

    @abstractmethod
    def classify_error(
        self,
        error_details: Dict[str, Any],
        code_context: Optional[str] = None, 
        execution_history: Optional[List[Dict[str, Any]]] = None,
        test_results: Optional[List[Dict[str, Any]]] = None,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> ClassificationResult:
        """
        Analyzes the provided error data and classifies the error.

        Args:
            error_details: A dictionary containing details about the error itself 
                           (e.g., from a try-except block, including type, message, traceback).
            code_context: Optional string containing the relevant code snippet where the error occurred.
            execution_history: Optional list of dictionaries representing the execution steps leading to the error.
            test_results: Optional list of dictionaries detailing outcomes of any test cases that were run.
            additional_context: Optional dictionary for any other contextual information that might be useful.

        Returns:
            A ClassificationResult object containing the classified error type, 
            confidence score, suggested responsible agent, and reasoning.
        """
        pass

    # --- Utility methods (can be implemented in base or overridden/used by subclasses) ---

    def extract_patterns_from_trace(self, traceback_str: Optional[str]) -> Dict[str, Any]:
        """
        Placeholder for a utility method to extract common patterns or keywords from a stack trace.
        Concrete implementations or subclasses might provide more sophisticated parsing.
        
        Args:
            traceback_str: The string representation of the stack trace.
            
        Returns:
            A dictionary of extracted patterns or relevant information.
        """
        if not traceback_str:
            return {"trace_analysis_status": "No traceback provided"}
        
        # Simple example: could look for keywords, count specific exception types, etc.
        patterns = {}
        if "SyntaxError" in traceback_str:
            patterns["contains_syntax_error"] = True
        if "RuntimeError" in traceback_str:
            patterns["contains_runtime_error"] = True
        # This is a very basic placeholder. Actual implementation would be more complex.
        return patterns

    def analyze_test_failures(self, test_results: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Placeholder for a utility method to analyze patterns in test failures.
        
        Args:
            test_results: A list of dictionaries, where each dict represents a test outcome.
                          Expected keys might include 'test_name', 'passed' (bool), 'details'.
                          
        Returns:
            A dictionary summarizing test failure patterns.
        """
        if not test_results:
            return {"test_analysis_status": "No test results provided"}
        
        summary = {
            "total_tests": len(test_results),
            "passed_count": 0,
            "failed_count": 0,
            "failure_types": [] # Could categorize types of failures if more info is available
        }
        for test in test_results:
            if test.get('passed', False):
                summary["passed_count"] += 1
            else:
                summary["failed_count"] += 1
                # Example: if test results contain error types
                if test.get('error_type'):
                    summary["failure_types"].append(test['error_type'])
        
        return summary

    # Subclasses might add more specific utility functions. 