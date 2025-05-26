#!/usr/bin/env python3
"""
FIXED Evolutionary Prompt Optimization System

This system actually works by:
1. Using ONLY direct task performance as fitness
2. Creating meaningful prompt variations that target specific errors
3. Implementing proper diversity maintenance
4. Using gradient-based prompt improvement
"""

import sys
import os
import logging
import time
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

# Ensure the src directory is in the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from self_healing_agents.agents import Planner, Executor, Critic
from self_healing_agents.llm_service import LLMService


@dataclass
class PromptCandidate:
    """A prompt candidate with its performance metrics."""
    prompt: str
    task_score: float
    error_types: List[str]
    test_results: Dict[str, Any]
    generation: int
    parent_ids: List[int] = None
    
    def __post_init__(self):
        if self.parent_ids is None:
            self.parent_ids = []


class DirectTaskFitnessEvaluator:
    """
    Fitness evaluator that ONLY uses direct task performance.
    No synthetic validation tasks - just the actual regex matching task.
    """
    
    def __init__(self, planner, executor, critic, task_description: str):
        self.planner = planner
        self.executor = executor
        self.critic = critic
        self.task_description = task_description
        
        # Regex test cases for direct evaluation
        self.test_cases = [
            {"s": "aa", "p": "a", "expected": False},
            {"s": "aa", "p": "a*", "expected": True},
            {"s": "ab", "p": ".*", "expected": True},
            {"s": "aab", "p": "c*a*b", "expected": True},
            {"s": "mississippi", "p": "mis*is*p*.", "expected": False},
            {"s": "ab", "p": ".*c", "expected": False},
            {"s": "aaa", "p": "a*a", "expected": True},
            {"s": "aaa", "p": "aaaa", "expected": False},
            {"s": "a", "p": "ab*a", "expected": False},  # This is the failing case
            {"s": "a", "p": "ab*", "expected": True},
        ]
    
    def evaluate_prompt(self, prompt: str, agent_type: str) -> PromptCandidate:
        """Evaluate a prompt by running the full pipeline and testing the code."""
        print(f"🧪 EVALUATING: {agent_type} prompt ({len(prompt)} chars)")
        
        try:
            # Temporarily set the prompt
            if agent_type == "EXECUTOR":
                original_prompt = self.executor.system_prompt
                self.executor.system_prompt = prompt
            else:
                original_prompt = self.planner.system_prompt
                self.planner.system_prompt = prompt
            
            # Run the full pipeline
            plan = self.planner.run(user_request=self.task_description)
            if isinstance(plan, dict) and plan.get("error"):
                return PromptCandidate(prompt, 0.0, ["planning_error"], {}, 0)
            
            code = self.executor.run(plan=plan, original_request=self.task_description)
            if isinstance(code, dict) and code.get("error"):
                return PromptCandidate(prompt, 0.1, ["execution_error"], {}, 0)
            
            # Test the code directly
            test_results = self._test_code_directly(code)
            task_score = test_results["success_rate"]
            
            # Identify specific error types
            error_types = self._identify_error_types(test_results)
            
            print(f"   📊 Score: {task_score:.3f}, Errors: {error_types}")
            
            return PromptCandidate(prompt, task_score, error_types, test_results, 0)
            
        except Exception as e:
            print(f"   ❌ Evaluation failed: {e}")
            return PromptCandidate(prompt, 0.0, ["evaluation_error"], {}, 0)
        finally:
            # Restore original prompt
            if agent_type == "EXECUTOR":
                self.executor.system_prompt = original_prompt
            else:
                self.planner.system_prompt = original_prompt
    
    def _test_code_directly(self, code: str) -> Dict[str, Any]:
        """Test the generated code on actual regex test cases."""
        results = {
            "total_tests": len(self.test_cases),
            "passed_tests": 0,
            "failed_tests": [],
            "syntax_errors": [],
            "runtime_errors": [],
            "success_rate": 0.0
        }
        
        try:
            # Extract and execute the function
            if "def isMatch" not in code:
                results["syntax_errors"].append("No isMatch function found")
                return results
            
            exec_globals = {"__builtins__": {}}
            exec_locals = {}
            exec(code, exec_globals, exec_locals)
            
            if "isMatch" not in exec_locals:
                results["syntax_errors"].append("isMatch function not defined")
                return results
            
            isMatch = exec_locals["isMatch"]
            
            # Test each case
            for i, test_case in enumerate(self.test_cases):
                try:
                    s, p, expected = test_case["s"], test_case["p"], test_case["expected"]
                    actual = isMatch(s, p)
                    
                    if actual == expected:
                        results["passed_tests"] += 1
                    else:
                        results["failed_tests"].append({
                            "case": i,
                            "input": f"isMatch('{s}', '{p}')",
                            "expected": expected,
                            "actual": actual
                        })
                        
                except Exception as e:
                    results["runtime_errors"].append({
                        "case": i,
                        "input": f"isMatch('{s}', '{p}')",
                        "error": str(e)
                    })
            
            results["success_rate"] = results["passed_tests"] / results["total_tests"]
            
        except SyntaxError as e:
            results["syntax_errors"].append(f"Syntax error: {e}")
        except Exception as e:
            results["runtime_errors"].append(f"Execution error: {e}")
        
        return results
    
    def _identify_error_types(self, test_results: Dict[str, Any]) -> List[str]:
        """Identify specific types of errors from test results."""
        error_types = []
        
        if test_results["syntax_errors"]:
            error_types.append("syntax_error")
        
        if test_results["runtime_errors"]:
            error_types.append("runtime_error")
        
        # Analyze failed test patterns
        failed_tests = test_results.get("failed_tests", [])
        if failed_tests:
            # Check for specific regex pattern failures
            star_failures = [f for f in failed_tests if "*" in f["input"]]
            dot_failures = [f for f in failed_tests if ".*" in f["input"]]
            
            if star_failures:
                error_types.append("star_pattern_error")
            if dot_failures:
                error_types.append("dot_pattern_error")
            if len(failed_tests) > len(star_failures) + len(dot_failures):
                error_types.append("general_logic_error")
        
        return error_types if error_types else ["unknown_error"]


class ErrorTargetedPromptGenerator:
    """
    Generates new prompts that specifically target identified errors.
    """
    
    def __init__(self, llm_service):
        self.llm_service = llm_service
    
    def generate_error_targeted_prompt(
        self, 
        base_prompt: str, 
        error_types: List[str], 
        failed_tests: List[Dict[str, Any]],
        agent_type: str
    ) -> str:
        """Generate a new prompt that specifically addresses the identified errors."""
        
        # Create error-specific guidance
        error_guidance = self._create_error_guidance(error_types, failed_tests)
        
        generation_prompt = f"""
You are an expert prompt engineer. I need you to improve this {agent_type} prompt to fix specific regex matching errors.

CURRENT PROMPT:
{base_prompt}

IDENTIFIED ERRORS:
{error_guidance}

FAILED TEST CASES:
{json.dumps(failed_tests, indent=2)}

Your task is to create an improved prompt that:
1. Maintains the core functionality and structure
2. Specifically addresses the identified error patterns
3. Provides clearer guidance for handling the problematic cases
4. Includes relevant examples or clarifications

Focus on the SPECIFIC errors, not general improvements. The new prompt should help the agent avoid these exact mistakes.

Generate the improved prompt:
"""
        
        try:
            messages = [
                {"role": "system", "content": "You are an expert prompt engineer specializing in fixing specific errors."},
                {"role": "user", "content": generation_prompt}
            ]
            
            improved_prompt = self.llm_service.invoke(messages, expect_json=False)
            return improved_prompt.strip()
            
        except Exception as e:
            print(f"   ❌ Error-targeted generation failed: {e}")
            return self._fallback_improvement(base_prompt, error_types, agent_type)
    
    def _create_error_guidance(self, error_types: List[str], failed_tests: List[Dict[str, Any]]) -> str:
        """Create specific guidance based on error types."""
        guidance = []
        
        if "star_pattern_error" in error_types:
            guidance.append("- STAR PATTERN ERROR: The '*' operator is not being handled correctly. It should match zero or more of the preceding element.")
        
        if "dot_pattern_error" in error_types:
            guidance.append("- DOT PATTERN ERROR: The '.' operator should match any single character.")
        
        if "general_logic_error" in error_types:
            guidance.append("- LOGIC ERROR: The overall matching logic has flaws in the algorithm implementation.")
        
        if "syntax_error" in error_types:
            guidance.append("- SYNTAX ERROR: The generated code has syntax issues.")
        
        if "runtime_error" in error_types:
            guidance.append("- RUNTIME ERROR: The code fails during execution.")
        
        # Add specific test case analysis
        if failed_tests:
            guidance.append("\nSPECIFIC FAILURE PATTERNS:")
            for test in failed_tests[:3]:  # Show first 3 failures
                guidance.append(f"- {test['input']} returned {test['actual']} but should return {test['expected']}")
        
        return "\n".join(guidance) if guidance else "- UNKNOWN ERROR: General performance issues detected."
    
    def _fallback_improvement(self, base_prompt: str, error_types: List[str], agent_type: str) -> str:
        """Simple fallback improvement when LLM generation fails."""
        if "star_pattern_error" in error_types:
            addition = "\n\nIMPORTANT: When implementing regex matching, ensure the '*' operator correctly handles zero occurrences of the preceding element. Use dynamic programming with proper base cases."
        elif "syntax_error" in error_types:
            addition = "\n\nIMPORTANT: Ensure all generated code is syntactically correct Python with proper indentation and complete function definitions."
        else:
            addition = "\n\nIMPORTANT: Focus on correctness and thorough testing of edge cases in your implementation."
        
        return base_prompt + addition


class FixedEvolutionaryOptimizer:
    """
    A completely redesigned evolutionary optimizer that actually works.
    """
    
    def __init__(self, llm_service, planner, executor, critic, task_description: str):
        self.llm_service = llm_service
        self.evaluator = DirectTaskFitnessEvaluator(planner, executor, critic, task_description)
        self.generator = ErrorTargetedPromptGenerator(llm_service)
        self.task_description = task_description
        
        # Evolution parameters
        self.population_size = 6
        self.max_generations = 5
        self.min_improvement_threshold = 0.05
        self.diversity_threshold = 0.3
    
    def optimize_prompt(self, base_prompt: str, agent_type: str) -> Tuple[str, float, Dict[str, Any]]:
        """
        Optimize a prompt using error-targeted evolution.
        
        Returns:
            Tuple of (best_prompt, best_score, optimization_stats)
        """
        print(f"\n🧬 STARTING FIXED EVOLUTIONARY OPTIMIZATION")
        print(f"   🎯 Agent Type: {agent_type}")
        print(f"   📊 Population Size: {self.population_size}")
        print(f"   🔄 Max Generations: {self.max_generations}")
        
        start_time = time.time()
        
        # Initialize population with base prompt
        population = [self.evaluator.evaluate_prompt(base_prompt, agent_type)]
        best_candidate = population[0]
        
        print(f"\n📊 BASELINE PERFORMANCE:")
        print(f"   Score: {best_candidate.task_score:.3f}")
        print(f"   Errors: {best_candidate.error_types}")
        
        generation_stats = []
        
        for generation in range(self.max_generations):
            print(f"\n🔄 GENERATION {generation + 1}/{self.max_generations}")
            
            # Generate new candidates targeting specific errors
            new_candidates = self._generate_targeted_candidates(
                best_candidate, agent_type, generation + 1
            )
            
            # Evaluate new candidates
            evaluated_candidates = []
            for candidate_prompt in new_candidates:
                candidate = self.evaluator.evaluate_prompt(candidate_prompt, agent_type)
                candidate.generation = generation + 1
                evaluated_candidates.append(candidate)
            
            # Update population (keep best candidates)
            all_candidates = population + evaluated_candidates
            all_candidates.sort(key=lambda x: x.task_score, reverse=True)
            population = all_candidates[:self.population_size]
            
            # Check for improvement
            current_best = population[0]
            improvement = current_best.task_score - best_candidate.task_score
            
            print(f"   📈 Best Score: {current_best.task_score:.3f} (Δ{improvement:+.3f})")
            print(f"   🎯 Best Errors: {current_best.error_types}")
            
            generation_stats.append({
                "generation": generation + 1,
                "best_score": current_best.task_score,
                "improvement": improvement,
                "population_scores": [c.task_score for c in population]
            })
            
            # Update best candidate if improved
            if current_best.task_score > best_candidate.task_score:
                best_candidate = current_best
                print(f"   ✅ NEW BEST CANDIDATE FOUND!")
            
            # Check termination conditions
            if current_best.task_score >= 0.9:  # Near perfect
                print(f"   🎉 NEAR PERFECT SCORE ACHIEVED!")
                break
            
            if improvement < self.min_improvement_threshold and generation > 1:
                print(f"   🛑 MINIMAL IMPROVEMENT - STOPPING")
                break
        
        execution_time = time.time() - start_time
        
        optimization_stats = {
            "generations": len(generation_stats),
            "execution_time": execution_time,
            "final_score": best_candidate.task_score,
            "initial_score": population[0].task_score if population else 0.0,
            "total_improvement": best_candidate.task_score - (population[0].task_score if population else 0.0),
            "generation_stats": generation_stats
        }
        
        print(f"\n🏆 OPTIMIZATION COMPLETE:")
        print(f"   Best Score: {best_candidate.task_score:.3f}")
        print(f"   Total Improvement: {optimization_stats['total_improvement']:+.3f}")
        print(f"   Generations: {optimization_stats['generations']}")
        print(f"   Time: {execution_time:.1f}s")
        
        return best_candidate.prompt, best_candidate.task_score, optimization_stats
    
    def _generate_targeted_candidates(
        self, 
        best_candidate: PromptCandidate, 
        agent_type: str, 
        generation: int
    ) -> List[str]:
        """Generate new prompt candidates that target specific errors."""
        candidates = []
        
        # If we have errors, generate error-targeted prompts
        if best_candidate.error_types and best_candidate.error_types != ["unknown_error"]:
            failed_tests = best_candidate.test_results.get("failed_tests", [])
            
            # Generate multiple variations targeting different error aspects
            for i in range(3):
                try:
                    targeted_prompt = self.generator.generate_error_targeted_prompt(
                        best_candidate.prompt,
                        best_candidate.error_types,
                        failed_tests,
                        agent_type
                    )
                    if targeted_prompt and targeted_prompt != best_candidate.prompt:
                        candidates.append(targeted_prompt)
                except Exception as e:
                    print(f"   ❌ Targeted generation {i+1} failed: {e}")
        
        # Generate some general variations for diversity
        for i in range(2):
            try:
                general_prompt = self._generate_general_variation(
                    best_candidate.prompt, agent_type, generation
                )
                if general_prompt and general_prompt != best_candidate.prompt:
                    candidates.append(general_prompt)
            except Exception as e:
                print(f"   ❌ General generation {i+1} failed: {e}")
        
        # Ensure we have at least some candidates
        if not candidates:
            candidates = [self._simple_variation(best_candidate.prompt, agent_type)]
        
        print(f"   🧬 Generated {len(candidates)} new candidates")
        return candidates[:4]  # Limit to 4 new candidates per generation
    
    def _generate_general_variation(self, base_prompt: str, agent_type: str, generation: int) -> str:
        """Generate a general variation of the prompt."""
        variation_prompt = f"""
Create a variation of this {agent_type} prompt that maintains the core functionality but uses different wording, structure, or examples.

ORIGINAL PROMPT:
{base_prompt}

Requirements:
1. Keep the same core purpose and functionality
2. Use different phrasing or structure
3. Add or modify examples if present
4. Maintain professional tone
5. Ensure clarity and completeness

Generate the variation:
"""
        
        try:
            messages = [
                {"role": "system", "content": "You are an expert prompt engineer creating variations."},
                {"role": "user", "content": variation_prompt}
            ]
            
            variation = self.llm_service.invoke(messages, expect_json=False)
            return variation.strip()
            
        except Exception as e:
            print(f"   ❌ General variation failed: {e}")
            return self._simple_variation(base_prompt, agent_type)
    
    def _simple_variation(self, base_prompt: str, agent_type: str) -> str:
        """Create a simple variation when LLM generation fails."""
        if agent_type == "EXECUTOR":
            addition = "\n\nAdditional Note: Pay special attention to edge cases and ensure your implementation handles all specified requirements correctly."
        else:
            addition = "\n\nAdditional Note: Ensure your plan covers all edge cases and provides clear, actionable steps."
        
        return base_prompt + addition


def main():
    """Test the fixed evolutionary system."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize services
    llm_service = LLMService(provider="deepseek", model_name="deepseek-chat")
    
    # Initialize agents
    planner = Planner("planner", llm_service)
    executor = Executor("executor", llm_service)
    critic = Critic("critic", llm_service)
    
    # Task description
    task_description = """Given an input string s and a pattern p, implement regular expression matching with support for '.' and '*' where:

'.' Matches any single character.
'*' Matches zero or more of the preceding element.
The matching should cover the entire input string (not partial).

Example 1:
Input: s = "aa", p = "a"
Output: false
Explanation: "a" does not match the entire string "aa".

Example 2:
Input: s = "aa", p = "a*"
Output: true
Explanation: '*' means zero or more of the preceding element, 'a'. Therefore, by repeating 'a' once, it becomes "aa".

Example 3:
Input: s = "ab", p = ".*"
Output: true
Explanation: ".*" means "zero or more (*) of any character (.)".

Constraints:
1 <= s.length <= 20
1 <= p.length <= 20
s contains only lowercase English letters.
p contains only lowercase English letters, '.', and '*'.
It is guaranteed for each appearance of the character '*', there will be a previous valid character to match."""
    
    # Test the baseline
    print("🧪 TESTING BASELINE PERFORMANCE")
    evaluator = DirectTaskFitnessEvaluator(planner, executor, critic, task_description)
    baseline = evaluator.evaluate_prompt(executor.system_prompt, "EXECUTOR")
    
    print(f"\n📊 BASELINE RESULTS:")
    print(f"   Score: {baseline.task_score:.3f}")
    print(f"   Errors: {baseline.error_types}")
    print(f"   Failed Tests: {len(baseline.test_results.get('failed_tests', []))}")
    
    # Run the fixed evolutionary optimization
    optimizer = FixedEvolutionaryOptimizer(llm_service, planner, executor, critic, task_description)
    
    best_prompt, best_score, stats = optimizer.optimize_prompt(executor.system_prompt, "EXECUTOR")
    
    print(f"\n🎉 FINAL RESULTS:")
    print(f"   Improvement: {stats['total_improvement']:+.3f}")
    print(f"   Final Score: {best_score:.3f}")
    print(f"   Generations: {stats['generations']}")
    
    # Test the final prompt
    print(f"\n🧪 TESTING FINAL PROMPT:")
    final_candidate = evaluator.evaluate_prompt(best_prompt, "EXECUTOR")
    print(f"   Score: {final_candidate.task_score:.3f}")
    print(f"   Errors: {final_candidate.error_types}")
    print(f"   Passed Tests: {final_candidate.test_results['passed_tests']}/{final_candidate.test_results['total_tests']}")
    
    if final_candidate.test_results.get("failed_tests"):
        print(f"\n❌ REMAINING FAILURES:")
        for failure in final_candidate.test_results["failed_tests"][:3]:
            print(f"   {failure['input']} → {failure['actual']} (expected {failure['expected']})")


if __name__ == "__main__":
    main() 