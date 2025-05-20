"""
Classifiers for error categorization and analysis.

This package provides different error classification approaches:
- Base classifier: Simple heuristic-based classification
- Rule-based classifier: Pattern matching with confidence scoring
"""
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any

from self_healing_agents.error_types import ErrorType

# Type alias for error data structure
ErrorDataType = Dict[str, Any]

class ErrorClassifier(ABC):
    """
    Abstract Base Class for error classifiers.
    """
    @abstractmethod
    def classify_error(self, error_data: ErrorDataType) -> Tuple[ErrorType, float]:
        """
        Classifies the given error data.
        
        Args:
            error_data: A dictionary containing details about the error.
                        
        Returns:
            A tuple containing:
                - ErrorType: The classified type of the error.
                - float: The confidence score (0.0 to 1.0) for the classification.
        """
        pass

# Import specific classifier implementations
# Import after ErrorClassifier definition to avoid circular imports
from self_healing_agents.classifiers.rule_based import RuleBasedErrorClassifier

__all__ = ['ErrorClassifier', 'RuleBasedErrorClassifier'] 