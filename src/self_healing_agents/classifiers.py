from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any

from self_healing_agents.error_types import ErrorType, MEDIUM_CONFIDENCE_THRESHOLD, HIGH_CONFIDENCE_THRESHOLD

# A more detailed error data structure might be defined later,
# for now, using Dict[str, Any] for flexibility.
ErrorDataType = Dict[str, Any]


class ErrorClassifier(ABC):
    """
    Abstract Base Class for error classifiers.

    Defines the interface for classifying errors into predefined categories
    (e.g., PLANNING_ERROR, EXECUTION_ERROR) and providing a confidence score.
    """

    @abstractmethod
    def classify_error(self, error_data: ErrorDataType) -> Tuple[ErrorType, float]:
        """
        Classifies the given error data.

        Args:
            error_data: A dictionary containing details about the error.
                        The exact structure might vary but could include:
                        - 'error_message': str
                        - 'stack_trace': str
                        - 'code_snippet': str
                        - 'test_failures': List[Dict] (details of failing tests)
                        - 'exception_type': str (e.g., 'SyntaxError', 'RuntimeError')

        Returns:
            A tuple containing:
                - ErrorType: The classified type of the error.
                - float: The confidence score (0.0 to 1.0) for the classification.
        """
        pass


class BaseErrorClassifier(ErrorClassifier):
    """
    A basic implementation of ErrorClassifier using simple heuristics.

    This classifier provides a foundational level of error distinction,
    primarily focusing on identifying clear syntax errors versus other
    potential failures. More sophisticated classifiers can extend this
    or implement the ErrorClassifier interface directly.
    """

    def classify_error(self, error_data: ErrorDataType) -> Tuple[ErrorType, float]:
        """
        Classifies errors based on simple heuristics.

        Heuristics:
        1. If 'exception_type' is 'SyntaxError', classify as EXECUTION_ERROR
           with high confidence.
        2. If 'exception_type' matches common runtime error names (e.g., 'NameError', 'TypeError')
           OR 'error_message' contains keywords typical of runtime execution issues,
           classify as EXECUTION_ERROR with medium-high confidence.
        3. If 'test_failures' list is present and not empty, and no clear execution error indicators,
           it might suggest a PLANNING_ERROR (logical flaw), but with lower confidence initially.
           The BaseHeuristic model will default to EXECUTION_ERROR for now and let more sophisticated models to handle PLANNING_ERROR.
        4.  Default to AMBIGUOUS_ERROR with low confidence if no strong signals.

        Args:
            error_data: A dictionary containing details about the error. Expected keys:
                        - 'exception_type': Optional[str]
                        - 'error_message': Optional[str]
                        - 'test_failures': Optional[List[Dict]]

        Returns:
            A tuple containing the ErrorType and confidence score.
        """
        exception_type = error_data.get('exception_type')
        error_message = error_data.get('error_message', '').lower() # Ensure lowercase for matching
        test_failures = error_data.get('test_failures')

        # Heuristic 1: Clear Syntax Error
        if exception_type == 'SyntaxError':
            return ErrorType.EXECUTION_ERROR, HIGH_CONFIDENCE_THRESHOLD

        # Heuristic 2: Common Runtime Execution Errors
        # List of common runtime exception type names (exact match)
        common_runtime_exception_types = [
            'NameError', 'TypeError', 'IndexError', 'AttributeError',
            'ZeroDivisionError', 'FileNotFoundError', 'KeyError',
            'ValueError', 'ImportError', 'ModuleNotFoundError'
        ]
        # List of keywords to check in the error_message (substring match, lowercase)
        runtime_error_message_keywords = [
            'nameerror', 'typeerror', 'indexerror', 'attributeerror',
            'zerodivisionerror', 'filenotfounderror', 'keyerror',
            'valueerror', 'importerror', 'modulenotfounderror',
            'syntaxerror' # Added to catch syntax error messages if exception_type is missing/different
        ]

        if exception_type in common_runtime_exception_types or \
           any(keyword in error_message for keyword in runtime_error_message_keywords):
            return ErrorType.EXECUTION_ERROR, (HIGH_CONFIDENCE_THRESHOLD + MEDIUM_CONFIDENCE_THRESHOLD) / 2 # Medium-high

        # Heuristic 3: Test failures might indicate logical/planning issues
        # For BaseErrorClassifier, we are keeping it simple.
        # If there are test failures, it implies the code ran but produced wrong results.
        # This is an EXECUTION_ERROR in the sense that the implemented code is faulty.
        # A more advanced classifier might differentiate if the logic itself (plan) was flawed.
        if test_failures and len(test_failures) > 0:
            # Could be a logical error (PLANNING_ERROR) if not a clear runtime error.
            # However, a failing test means the code *executed* but wrongly.
            # This base classifier will lean towards EXECUTION_ERROR.
            return ErrorType.EXECUTION_ERROR, MEDIUM_CONFIDENCE_THRESHOLD

        # Default: Ambiguous if no strong signals from the above
        # A more sophisticated model would have more rules or use an LLM.
        # For now, if we reach here, it's hard to tell with simple heuristics.
        # Let's default to AMBIGUOUS_ERROR, but if there's any error message at all,
        # slightly lean towards EXECUTION_ERROR as something went wrong at runtime.
        if error_message or exception_type:
            return ErrorType.AMBIGUOUS_ERROR, MEDIUM_CONFIDENCE_THRESHOLD / 2 # Low confidence ambiguous
            
        return ErrorType.AMBIGUOUS_ERROR, 0.1 # Very low confidence if no info 