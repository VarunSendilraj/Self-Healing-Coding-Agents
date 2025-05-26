#!/usr/bin/env python3
"""
Test script to validate assumptions about error analysis and prompt generation specificity loss.
"""

import logging
import os
import sys

# Setup logging to see all diagnostic messages
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add the src directory to the path
sys.path.insert(0, 'src')

from self_healing_agents.evolution.evolution_operators import EvolutionOperators
from self_healing_agents.llm_service import LLMService

def test_error_analysis_specificity():
    """Test the error analysis process to see where specificity is lost."""
    
    print("🧪 TESTING ERROR ANALYSIS SPECIFICITY LOSS")
    print("=" * 60)
    
    # Initialize LLM service and operators
    llm_service = LLMService(provider='deepseek', model_name='deepseek-coder')
    operators = EvolutionOperators(llm_service)
    
    print("✅ LLM service and operators initialized")
    
    # 🆕 ENHANCED: Test with specific test failure information
    enhanced_failure_context = {
        "original_task": "Given an input string s and a pattern p, implement regular expression matching with support for '.' and '*' where: '.' Matches any single character. '*' Matches zero or more of the preceding element.",
        "failure_report": {
            "overall_status": "FAILURE",
            "quantitative_score": 0.65,
            "feedback": "Code executed successfully, but 1/6 tests failed for function isMatch."
        },
        "classification": {
            "primary_failure_type": "EXECUTION_FAILURE",
            "reasoning": [
                "The plan correctly outlines a dynamic programming approach with memoization, which is appropriate for this problem.",
                "The code implementation fails on a specific test case involving '*' matching zero elements, suggesting an execution flaw rather than a planning issue.",
                "The plan includes all necessary steps (base cases, handling '.' and '*', recursive/DP approach), but the implementation doesn't correctly handle the zero-match case for '*'."
            ]
        },
        # 🆕 CRITICAL: Add specific test failure information
        "specific_test_failures": [
            {
                "test_name": "test_star_zero_match",
                "inputs": {"s": "a", "p": "ab*a"},
                "expected_output": True,
                "actual_output": False,
                "error_type": "logic_error",
                "regex_pattern_issue": {
                    "string": "a",
                    "pattern": "ab*a",
                    "contains_star": True,
                    "likely_issue": "star_operator_handling"
                }
            },
            {
                "test_name": "test_star_multiple_match",
                "inputs": {"s": "aaa", "p": "a*"},
                "expected_output": True,
                "actual_output": True,
                "error_type": "none"
            }
        ]
    }
    
    print("\n🔍 STEP 1: Error Type Analysis")
    print("-" * 40)
    error_type = operators._analyze_error_type(enhanced_failure_context)
    
    print("\n🔍 STEP 2: Error Details Extraction")
    print("-" * 40)
    error_details = operators._extract_error_details(enhanced_failure_context)
    
    print("\n🔍 STEP 3: Mutation Selection")
    print("-" * 40)
    mutations = operators._select_error_targeted_mutations(error_type, "EXECUTOR", enhanced_failure_context)
    
    print("\n🔍 STEP 4: Improvement Guidelines")
    print("-" * 40)
    improvements = operators._get_error_specific_improvements(error_type, "EXECUTOR", enhanced_failure_context)
    
    print("\n📊 FINAL RESULTS:")
    print("=" * 60)
    print(f"Error Type: {error_type}")
    print(f"Error Details Length: {len(error_details)} characters")
    print(f"Selected Mutations: {mutations}")
    print(f"Improvements Length: {len(improvements)} characters")
    
    print("\n🎯 SPECIFICITY ANALYSIS:")
    print("-" * 40)
    
    # Check for specific regex pattern targeting
    if "ab*a" in error_details:
        print("✅ GOOD: Specific regex pattern 'ab*a' found in error details")
    else:
        print("❌ BAD: Specific regex pattern 'ab*a' NOT found in error details")
    
    if "zero" in error_details.lower():
        print("✅ GOOD: 'zero' concept found in error details")
    else:
        print("❌ BAD: 'zero' concept NOT found in error details")
    
    # Check for regex-specific mutations
    regex_mutations = ["regex_star_operator_focus", "zero_match_handling", "pattern_matching_precision"]
    has_regex_mutations = any(mut in mutations for mut in regex_mutations)
    
    if has_regex_mutations:
        print("✅ GOOD: Regex-specific mutations selected")
        regex_found = [mut for mut in mutations if mut in regex_mutations]
        print(f"   Regex mutations: {regex_found}")
    else:
        print("❌ BAD: No regex/pattern-specific mutations selected")
        print(f"   Selected mutations are generic: {mutations}")
    
    # Check for specific test case targeting in improvements
    if "ab*a" in improvements and "string='a'" in improvements:
        print("✅ GOOD: Specific test case targeting found in improvements")
    else:
        print("❌ BAD: No specific test case targeting in improvements")
    
    print(f"\n📝 Error Details Preview:")
    print(f"   {error_details[:200]}...")
    
    print(f"\n📝 Improvement Guidelines Preview:")
    print(f"   {improvements[:200]}...")
    
    return True

if __name__ == "__main__":
    test_error_analysis_specificity() 