from dataclasses import dataclass
import difflib
import re
import sys
import os
from typing import Dict, List, Optional, Tuple, Any, Set

# Add the project root to the path if needed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from src.self_healing_agents.error_types import ErrorType
from src.self_healing_agents.schemas import CriticReport


@dataclass
class FixAssessment:
    """
    Represents an assessment of a code fix attempt, including metrics and analysis.
    """
    original_error_resolved: bool  # Whether the original error was resolved
    error_still_present: bool  # Whether the same error is still present
    new_errors_introduced: bool  # Whether new errors were introduced
    error_type_changed: bool  # Whether the error type has changed
    test_improvement: float  # Change in test case pass rate (0.0 to 1.0)
    code_change_magnitude: float  # Relative size of code changes (0.0 to 1.0)
    fix_quality_score: float  # Overall quality score (0.0 to 1.0)
    details: Dict[str, Any]  # Additional assessment details


class ErrorFixEvaluator:
    """
    Evaluates whether a fix attempt successfully resolved an error and assesses the quality of the fix.
    This class compares original and fixed code, analyzes error resolution, and evaluates test results.
    """
    
    def __init__(self):
        """Initialize the evaluator."""
        pass
    
    def evaluate_fix(self, 
                    original_code: str, 
                    fixed_code: str,
                    original_error_details: Dict[str, str],
                    fixed_error_details: Optional[Dict[str, str]],
                    original_test_results: Optional[List[Dict[str, Any]]] = None,
                    fixed_test_results: Optional[List[Dict[str, Any]]] = None) -> FixAssessment:
        """
        Evaluates whether a fix attempt successfully resolved an error.
        
        Args:
            original_code: The original code with errors.
            fixed_code: The fixed code.
            original_error_details: Details of the original error.
            fixed_error_details: Details of any errors in the fixed code (None if no errors).
            original_test_results: Test results for the original code.
            fixed_test_results: Test results for the fixed code.
            
        Returns:
            A FixAssessment object with evaluation results.
        """
        # Check if original error is resolved
        error_resolved = fixed_error_details is None

        # Check if the same error still exists
        error_still_present = False
        error_type_changed = False
        if fixed_error_details:
            original_error_type = original_error_details.get('error_type')
            fixed_error_type = fixed_error_details.get('error_type')
            original_error_msg = original_error_details.get('error_message', '')
            fixed_error_msg = fixed_error_details.get('error_message', '')
            
            error_still_present = (fixed_error_type == original_error_type and 
                                  self._similar_error_messages(original_error_msg, fixed_error_msg))
            error_type_changed = fixed_error_type != original_error_type
        
        # Check if new errors were introduced
        new_errors = False
        if fixed_error_details and not error_still_present:
            new_errors = True
            
        # Calculate test improvement
        test_improvement = self._calculate_test_improvement(original_test_results, fixed_test_results)
        
        # Calculate code change magnitude
        change_magnitude = self._calculate_code_change_magnitude(original_code, fixed_code)
        
        # Generate detailed comparison
        details = self._generate_detailed_comparison(
            original_code, 
            fixed_code,
            original_error_details,
            fixed_error_details,
            original_test_results,
            fixed_test_results
        )
        
        # Calculate overall fix quality score
        fix_quality = self._calculate_fix_quality(
            error_resolved, 
            new_errors, 
            test_improvement, 
            change_magnitude
        )
        
        return FixAssessment(
            original_error_resolved=error_resolved,
            error_still_present=error_still_present,
            new_errors_introduced=new_errors,
            error_type_changed=error_type_changed,
            test_improvement=test_improvement,
            code_change_magnitude=change_magnitude,
            fix_quality_score=fix_quality,
            details=details
        )
    
    def evaluate_fix_from_reports(self, original_report: CriticReport, fixed_report: CriticReport, 
                                 original_code: str, fixed_code: str) -> FixAssessment:
        """
        Evaluates fix attempt based on critic reports.
        
        Args:
            original_report: The critic report for the original code.
            fixed_report: The critic report for the fixed code.
            original_code: The original code with errors.
            fixed_code: The fixed code.
            
        Returns:
            A FixAssessment object with evaluation results.
        """
        original_error_details = original_report.error_details if original_report.error_details else {}
        fixed_error_details = fixed_report.error_details if fixed_report.error_details else None
        
        return self.evaluate_fix(
            original_code=original_code,
            fixed_code=fixed_code,
            original_error_details=original_error_details,
            fixed_error_details=fixed_error_details,
            original_test_results=original_report.test_results,
            fixed_test_results=fixed_report.test_results
        )
    
    def is_error_resolved(self, fixed_error_details: Optional[Dict[str, str]] = None) -> bool:
        """
        Determines if an error has been resolved based on error details.
        
        Args:
            fixed_error_details: Error details from fixed code execution (None if no error).
            
        Returns:
            True if the error has been resolved, False otherwise.
        """
        return fixed_error_details is None
    
    def _similar_error_messages(self, msg1: str, msg2: str, similarity_threshold: float = 0.7) -> bool:
        """
        Checks if two error messages are similar using difflib sequence matcher.
        
        Args:
            msg1: First error message.
            msg2: Second error message.
            similarity_threshold: Threshold for considering messages similar (0.0 to 1.0).
            
        Returns:
            True if the messages are similar, False otherwise.
        """
        if not msg1 or not msg2:
            return False
            
        # Focus on the core part of error messages (remove variable parts like line numbers, etc.)
        def clean_msg(msg):
            # Strip variable parts like line numbers, memory addresses, etc.
            msg = re.sub(r'line \d+', 'line', msg)
            msg = re.sub(r'at 0x[0-9a-f]+', 'at', msg)
            return msg.strip()
            
        clean_msg1 = clean_msg(msg1)
        clean_msg2 = clean_msg(msg2)
        
        # Calculate similarity
        matcher = difflib.SequenceMatcher(None, clean_msg1, clean_msg2)
        similarity = matcher.ratio()
        
        return similarity >= similarity_threshold
    
    def _calculate_test_improvement(self, 
                                   original_results: Optional[List[Dict[str, Any]]], 
                                   fixed_results: Optional[List[Dict[str, Any]]]) -> float:
        """
        Calculates the improvement in test case pass rate.
        
        Args:
            original_results: Test results for original code.
            fixed_results: Test results for fixed code.
            
        Returns:
            A float between -1.0 and 1.0 representing the change in pass rate.
            Positive values indicate improvement, negative values indicate regression.
        """
        if not original_results and not fixed_results:
            return 0.0
            
        def calculate_pass_rate(results):
            if not results:
                return 0.0
            passed = sum(1 for test in results if test.get('status') == 'passed')
            return passed / len(results) if results else 0.0
            
        original_pass_rate = calculate_pass_rate(original_results)
        fixed_pass_rate = calculate_pass_rate(fixed_results)
        
        return fixed_pass_rate - original_pass_rate
    
    def _calculate_code_change_magnitude(self, original_code: str, fixed_code: str) -> float:
        """
        Calculates the relative magnitude of code changes.
        
        Args:
            original_code: The original code.
            fixed_code: The fixed code.
            
        Returns:
            A float between 0.0 and 1.0 representing the magnitude of changes.
            0.0 means no changes, 1.0 means completely different code.
        """
        if not original_code or not fixed_code:
            return 1.0 if original_code != fixed_code else 0.0
            
        # Clean code to remove leading/trailing whitespace and markdown
        def clean_code(code):
            code = code.strip()
            if code.startswith("```python") and code.endswith("```"):
                code = code[len("```python"):-3].strip()
            elif code.startswith("```") and code.endswith("```"):
                code = code[3:-3].strip()
            return code
            
        clean_original = clean_code(original_code)
        clean_fixed = clean_code(fixed_code)
        
        if clean_original == clean_fixed:
            return 0.0
            
        # Calculate the diff
        diff = list(difflib.unified_diff(
            clean_original.splitlines(),
            clean_fixed.splitlines(),
            lineterm=''
        ))
        
        # Count lines changed (added, removed, modified)
        changed_lines = sum(1 for line in diff if line.startswith('+') or line.startswith('-'))
        total_lines = len(clean_original.splitlines()) + len(clean_fixed.splitlines())
        
        if total_lines == 0:
            return 0.0
            
        # Normalize to 0.0-1.0 range
        return min(1.0, changed_lines / total_lines)
    
    def _generate_detailed_comparison(self, 
                                    original_code: str, 
                                    fixed_code: str,
                                    original_error_details: Dict[str, str],
                                    fixed_error_details: Optional[Dict[str, str]],
                                    original_test_results: Optional[List[Dict[str, Any]]],
                                    fixed_test_results: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Generates a detailed comparison between original and fixed code.
        
        Args:
            original_code: The original code.
            fixed_code: The fixed code.
            original_error_details: Details of the original error.
            fixed_error_details: Details of any errors in the fixed code.
            original_test_results: Test results for the original code.
            fixed_test_results: Test results for the fixed code.
            
        Returns:
            A dictionary with detailed comparison information.
        """
        # Clean code for comparison
        def clean_code(code):
            code = code.strip()
            if code.startswith("```python") and code.endswith("```"):
                code = code[len("```python"):-3].strip()
            elif code.startswith("```") and code.endswith("```"):
                code = code[3:-3].strip()
            return code
            
        clean_original = clean_code(original_code)
        clean_fixed = clean_code(fixed_code)
        
        # Generate diff
        diff_lines = list(difflib.unified_diff(
            clean_original.splitlines(),
            clean_fixed.splitlines(),
            lineterm='',
            n=3  # Context lines
        ))
        
        # Analyze changes
        lines_added = sum(1 for line in diff_lines if line.startswith('+') and not line.startswith('+++'))
        lines_removed = sum(1 for line in diff_lines if line.startswith('-') and not line.startswith('---'))
        files_changed = 1  # Assume single file for now
        
        # Compare test results
        original_passed_tests = set()
        fixed_passed_tests = set()
        
        if original_test_results:
            original_passed_tests = {
                test.get('name', f"test_{i}") 
                for i, test in enumerate(original_test_results) 
                if test.get('status') == 'passed'
            }
            
        if fixed_test_results:
            fixed_passed_tests = {
                test.get('name', f"test_{i}") 
                for i, test in enumerate(fixed_test_results) 
                if test.get('status') == 'passed'
            }
        
        newly_passing_tests = fixed_passed_tests - original_passed_tests
        newly_failing_tests = original_passed_tests - fixed_passed_tests
        
        return {
            "diff": "\n".join(diff_lines),
            "stats": {
                "lines_added": lines_added,
                "lines_removed": lines_removed,
                "files_changed": files_changed,
            },
            "test_changes": {
                "newly_passing": list(newly_passing_tests),
                "newly_failing": list(newly_failing_tests),
                "total_original_passing": len(original_passed_tests),
                "total_fixed_passing": len(fixed_passed_tests),
            },
            "original_error": original_error_details,
            "fixed_error": fixed_error_details
        }
    
    def _calculate_fix_quality(self, 
                             error_resolved: bool, 
                             new_errors_introduced: bool, 
                             test_improvement: float, 
                             change_magnitude: float) -> float:
        """
        Calculates an overall fix quality score.
        
        Args:
            error_resolved: Whether the original error was resolved.
            new_errors_introduced: Whether new errors were introduced.
            test_improvement: Change in test case pass rate.
            change_magnitude: Magnitude of code changes.
            
        Returns:
            A float between 0.0 and 1.0 representing the overall fix quality.
            Higher values indicate better fixes.
        """
        # Base score depends on error resolution
        base_score = 0.7 if error_resolved else 0.0
        
        # Penalize for introducing new errors
        if new_errors_introduced:
            base_score *= 0.5
        
        # Adjust based on test improvements
        test_factor = 0.2 * (1 + test_improvement)  # -0.2 to +0.4
        
        # Prefer minimal changes (smaller changes for same outcome are better)
        # Only apply this if the error was resolved or tests improved
        change_penalty = 0.0
        if error_resolved or test_improvement > 0:
            # More penalty for larger changes
            change_penalty = -0.1 * change_magnitude
        
        # Combine factors with bounds checking
        quality_score = base_score + test_factor + change_penalty
        
        # Ensure score is within valid range
        return max(0.0, min(1.0, quality_score)) 