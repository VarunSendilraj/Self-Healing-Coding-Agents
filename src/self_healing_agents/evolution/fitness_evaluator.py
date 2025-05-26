"""
Fitness evaluation for evolutionary prompt optimization.
"""

import logging
import time
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from .evolution_config import EvolutionConfig

logger = logging.getLogger(__name__)


@dataclass
class FitnessMetrics:
    """Container for fitness evaluation metrics."""
    success_rate: float = 0.0
    efficiency: float = 0.0
    coherence: float = 0.0
    generalization: float = 0.0
    total_fitness: float = 0.0
    execution_time: float = 0.0
    token_count: int = 0
    task_scores: List[float] = None
    
    def __post_init__(self):
        if self.task_scores is None:
            self.task_scores = []


class PromptFitnessEvaluator:
    """
    Evaluates prompt fitness using multi-objective criteria with task-specific evaluation.
    """
    
    def __init__(
        self, 
        config: EvolutionConfig,
        llm_service,
        validation_tasks: List[Dict[str, Any]] = None
    ):
        self.config = config
        self.llm_service = llm_service
        self.validation_tasks = validation_tasks or self._create_default_validation_tasks()
        self.evaluation_count = 0
        self.evaluation_cache: Dict[str, FitnessMetrics] = {}
        
        # Task-specific evaluation components (set via context)
        self.task_specific_context = None
        self.planner_agent = None
        self.executor_agent = None 
        self.critic_agent = None
        
        # Regex-specific test cases for direct performance testing
        self.regex_test_cases = [
            {"s": "aa", "p": "a", "expected": False},
            {"s": "aa", "p": "a*", "expected": True},
            {"s": "ab", "p": ".*", "expected": True},
            {"s": "aab", "p": "c*a*b", "expected": True},
            {"s": "mississippi", "p": "mis*is*p*.", "expected": False},
            {"s": "ab", "p": ".*c", "expected": False},
            {"s": "aaa", "p": "a*a", "expected": True},
            {"s": "aaa", "p": "aaaa", "expected": False},
        ]
    
    def set_task_specific_context(
        self,
        task_description: str,
        planner_agent=None,
        executor_agent=None,
        critic_agent=None
    ):
        """Set context for task-specific fitness evaluation using real multi-agent pipeline."""
        self.task_specific_context = {
            "task_description": task_description,
            "original_task_description": task_description
        }
        self.planner_agent = planner_agent
        self.executor_agent = executor_agent 
        self.critic_agent = critic_agent
        
        # Create task-specific validation that uses the ACTUAL task
        if "regex" in task_description.lower() or "regular expression" in task_description.lower():
            # For regex tasks, create targeted validation tasks
            self.validation_tasks = self._create_regex_validation_tasks(task_description)
        else:
            # For other tasks, create generic task-specific validation
            self.validation_tasks = [{
                "description": task_description,
                "expected_plan_elements": self._extract_key_concepts(task_description),
                "expected_code_elements": ["def", "return"],
                "plan": {"steps": ["Implement the solution"]}
            }]
        
        logger.info(f"🎯 TASK-SPECIFIC FITNESS: Set to evaluate on actual task")
        logger.info(f"   📋 Task: {task_description[:100]}...")
        logger.info(f"   🧪 Created {len(self.validation_tasks)} targeted validation tasks")
    
    def _create_regex_validation_tasks(self, task_description: str) -> List[Dict[str, Any]]:
        """Create regex-specific validation tasks that test actual regex matching performance."""
        return [
            {
                "description": task_description,
                "task_type": "regex_matching",
                "test_cases": self.regex_test_cases,
                "expected_plan_elements": ["regex", "matching", "pattern", "string", "dynamic", "programming"],
                "expected_code_elements": ["def", "isMatch", "return", "dp", "memo"],
                "plan": {"steps": ["Implement regex matching with DP", "Handle . and * patterns", "Return boolean result"]}
            }
        ]
    
    def _extract_key_concepts(self, task_description: str) -> List[str]:
        """Extract key concepts from task description for validation."""
        # Simple keyword extraction
        keywords = []
        text = task_description.lower()
        
        # Common programming concepts
        if "function" in text: keywords.append("function")
        if "algorithm" in text: keywords.append("algorithm")
        if "data structure" in text: keywords.append("data_structure")
        if "sort" in text: keywords.append("sorting")
        if "search" in text: keywords.append("searching")
        if "tree" in text: keywords.append("tree")
        if "graph" in text: keywords.append("graph")
        if "dynamic programming" in text or "dp" in text: keywords.append("dynamic_programming")
        if "recursion" in text: keywords.append("recursion")
        if "iteration" in text: keywords.append("iteration")
        
        return keywords if keywords else ["implementation", "solution"]

    def evaluate_prompt(
        self, 
        prompt: str, 
        agent_type: str,
        failure_context: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Evaluate a single prompt's fitness.
        
        Args:
            prompt: The prompt to evaluate
            agent_type: Type of agent (PLANNER or EXECUTOR)
            failure_context: Context about the original failure
            
        Returns:
            Combined fitness score (0.0 - 1.0)
        """
        # Check cache first
        cache_key = self._get_cache_key(prompt, agent_type, failure_context)
        if cache_key in self.evaluation_cache:
            logger.debug(f"🔄 FITNESS CACHE: Using cached result for prompt")
            return self.evaluation_cache[cache_key].total_fitness
        
        logger.info(f"🎯 FITNESS EVAL: Evaluating {agent_type} prompt fitness")
        start_time = time.time()
        
        try:
            # Run evaluation on validation tasks
            metrics = self._evaluate_on_validation_tasks(prompt, agent_type, failure_context)
            
            # Calculate combined fitness score
            fitness_score = self._calculate_combined_fitness(metrics)
            metrics.total_fitness = fitness_score
            metrics.execution_time = time.time() - start_time
            
            # Cache result
            self.evaluation_cache[cache_key] = metrics
            self.evaluation_count += 1
            
            logger.info(f"✅ FITNESS EVAL: Score {fitness_score:.3f} "
                       f"(success: {metrics.success_rate:.3f}, "
                       f"efficiency: {metrics.efficiency:.3f}, "
                       f"coherence: {metrics.coherence:.3f})")
            
            return fitness_score
            
        except Exception as e:
            logger.error(f"❌ FITNESS EVAL ERROR: {e}")
            return 0.0  # Return minimum fitness on error

    def batch_evaluate(
        self, 
        prompts: List[str], 
        agent_type: str,
        failure_context: Optional[Dict[str, Any]] = None
    ) -> List[float]:
        """
        Evaluate multiple prompts efficiently.
        
        Args:
            prompts: List of prompts to evaluate
            agent_type: Type of agent
            failure_context: Context about the original failure
            
        Returns:
            List of fitness scores
        """
        logger.info(f"📊 BATCH FITNESS: Evaluating {len(prompts)} {agent_type} prompts")
        
        scores = []
        for i, prompt in enumerate(prompts):
            logger.debug(f"Evaluating prompt {i+1}/{len(prompts)}")
            score = self.evaluate_prompt(prompt, agent_type, failure_context)
            scores.append(score)
            
            # Check evaluation budget
            if self.evaluation_count >= self.config.max_evaluations:
                logger.warning(f"⚠️ BUDGET EXCEEDED: Reached max evaluations ({self.config.max_evaluations})")
                # Fill remaining with average score
                avg_score = sum(scores) / len(scores) if scores else 0.0
                scores.extend([avg_score] * (len(prompts) - len(scores)))
                break
        
        logger.info(f"✅ BATCH FITNESS: Completed. Score range: {min(scores):.3f} - {max(scores):.3f}")
        return scores
    
    def get_detailed_metrics(self, prompt: str, agent_type: str) -> Optional[FitnessMetrics]:
        """Get detailed fitness metrics for a prompt."""
        cache_key = self._get_cache_key(prompt, agent_type)
        return self.evaluation_cache.get(cache_key)
    
    def _evaluate_on_validation_tasks(
        self, 
        prompt: str, 
        agent_type: str,
        failure_context: Optional[Dict[str, Any]]
    ) -> FitnessMetrics:
        """Evaluate prompt on validation task set."""
        metrics = FitnessMetrics()
        
        # Use subset of validation tasks to limit evaluations
        task_subset = self.validation_tasks[:self.config.validation_tasks_count]
        
        task_scores = []
        total_tokens = 0
        execution_times = []
        
        for task in task_subset:
            try:
                task_start = time.time()
                score, tokens = self._evaluate_single_task(prompt, agent_type, task, failure_context)
                task_execution_time = time.time() - task_start
                
                task_scores.append(score)
                total_tokens += tokens
                execution_times.append(task_execution_time)
                
            except Exception as e:
                logger.warning(f"Task evaluation failed: {e}")
                task_scores.append(0.0)
                execution_times.append(0.0)
        
        # Calculate success rate
        metrics.success_rate = sum(task_scores) / len(task_scores) if task_scores else 0.0
        
        # Calculate efficiency (inverse of time and tokens)
        avg_time = sum(execution_times) / len(execution_times) if execution_times else 1.0
        avg_tokens = total_tokens / len(task_subset) if task_subset else 1
        
        # Normalize efficiency (higher is better)
        time_efficiency = min(1.0, 1.0 / (avg_time + 0.1))  # Avoid division by zero
        token_efficiency = min(1.0, 100.0 / (avg_tokens + 1))  # Normalize to reasonable range
        metrics.efficiency = (time_efficiency + token_efficiency) / 2.0
        
        # Calculate coherence
        metrics.coherence = self._evaluate_prompt_coherence(prompt, agent_type)
        
        # Calculate generalization (consistency across tasks)
        if len(task_scores) > 1:
            score_variance = sum((score - metrics.success_rate) ** 2 for score in task_scores) / len(task_scores)
            metrics.generalization = max(0.0, 1.0 - score_variance)  # Lower variance = better generalization
        else:
            metrics.generalization = metrics.success_rate
        
        metrics.task_scores = task_scores
        metrics.token_count = total_tokens
        
        return metrics
    
    def _evaluate_single_task(
        self, 
        prompt: str, 
        agent_type: str, 
        task: Dict[str, Any],
        failure_context: Optional[Dict[str, Any]]
    ) -> Tuple[float, int]:
        """
        Evaluate prompt on a single validation task.
        
        Returns:
            Tuple of (score, token_count)
        """
        try:
            # Check if this is a regex task with direct testing capability
            if task.get("task_type") == "regex_matching" and self.task_specific_context:
                return self._evaluate_regex_task_directly(prompt, agent_type, task, failure_context)
            elif agent_type == "PLANNER":
                # Use pipeline evaluation if available, otherwise fallback
                if self.task_specific_context:
                    return self._evaluate_planner_task_with_pipeline(prompt, task, failure_context)
                else:
                    return self._evaluate_planner_task(prompt, task, failure_context)
            elif agent_type == "EXECUTOR":
                # Use pipeline evaluation if available, otherwise fallback
                if self.task_specific_context:
                    return self._evaluate_executor_task_with_pipeline(prompt, task, failure_context)
                else:
                    return self._evaluate_executor_task(prompt, task, failure_context)
            else:
                logger.error(f"Unknown agent type: {agent_type}")
                return 0.0, 0
                
        except Exception as e:
            logger.error(f"Single task evaluation failed: {e}")
            return 0.0, 0
    
    def _evaluate_regex_task_directly(
        self, 
        prompt: str, 
        agent_type: str, 
        task: Dict[str, Any],
        failure_context: Optional[Dict[str, Any]]
    ) -> Tuple[float, int]:
        """
        Directly evaluate regex task by testing the generated code on actual test cases.
        This provides the most accurate fitness evaluation for regex matching tasks.
        """
        logger.info(f"🧪 DIRECT REGEX EVAL: Testing {agent_type} prompt on actual regex test cases")
        
        try:
            # Temporarily set the agent's prompt
            if agent_type == "PLANNER":
                original_prompt = self.planner_agent.system_prompt
                self.planner_agent.system_prompt = prompt
                
                # Get plan with evolved prompt
                plan = self.planner_agent.run(user_request=self.task_specific_context["task_description"])
                if isinstance(plan, dict) and plan.get("error"):
                    return 0.0, 100
                
                # Execute with current executor
                code = self.executor_agent.run(plan=plan, original_request=self.task_specific_context["task_description"])
                
                # Restore original prompt
                self.planner_agent.system_prompt = original_prompt
                
            else:  # EXECUTOR
                original_prompt = self.executor_agent.system_prompt
                self.executor_agent.system_prompt = prompt
                
                # Get plan with current planner
                plan = self.planner_agent.run(user_request=self.task_specific_context["task_description"])
                if isinstance(plan, dict) and plan.get("error"):
                    return 0.0, 100
                
                # Execute with evolved prompt
                code = self.executor_agent.run(plan=plan, original_request=self.task_specific_context["task_description"])
                
                # Restore original prompt
                self.executor_agent.system_prompt = original_prompt
            
            if isinstance(code, dict) and code.get("error"):
                return 0.1, 150
            
            # Test the generated code directly on regex test cases
            test_results = self._test_regex_code_directly(code, task.get("test_cases", []))
            
            # Calculate fitness based on test results
            if test_results["total_tests"] == 0:
                return 0.0, 200
            
            success_rate = test_results["passed_tests"] / test_results["total_tests"]
            
            # Bonus for handling edge cases correctly
            edge_case_bonus = 0.0
            if test_results["passed_tests"] >= test_results["total_tests"] * 0.8:
                edge_case_bonus = 0.1
            
            # Penalty for syntax errors or runtime errors
            error_penalty = 0.0
            if test_results["syntax_errors"] > 0:
                error_penalty = 0.3
            elif test_results["runtime_errors"] > 0:
                error_penalty = 0.2
            
            final_score = max(0.0, success_rate + edge_case_bonus - error_penalty)
            
            logger.info(f"   ✅ DIRECT TEST RESULTS: {test_results['passed_tests']}/{test_results['total_tests']} passed, score: {final_score:.3f}")
            
            token_count = len(str(code).split())
            return final_score, token_count
            
        except Exception as e:
            logger.error(f"   ❌ Direct regex evaluation failed: {e}")
            return 0.0, 100
    
    def _test_regex_code_directly(self, code: str, test_cases: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Test the generated regex code directly on test cases.
        
        Returns:
            Dictionary with test results
        """
        results = {
            "total_tests": len(test_cases),
            "passed_tests": 0,
            "failed_tests": 0,
            "syntax_errors": 0,
            "runtime_errors": 0
        }
        
        try:
            # Extract the function from the code
            # Look for isMatch function definition
            if "def isMatch" not in code:
                results["syntax_errors"] = 1
                return results
            
            # Create a safe execution environment
            exec_globals = {"__builtins__": {}}
            exec_locals = {}
            
            # Execute the code to define the function
            exec(code, exec_globals, exec_locals)
            
            # Get the isMatch function
            if "isMatch" not in exec_locals:
                results["syntax_errors"] = 1
                return results
            
            isMatch = exec_locals["isMatch"]
            
            # Test each case
            for test_case in test_cases:
                try:
                    s = test_case["s"]
                    p = test_case["p"]
                    expected = test_case["expected"]
                    
                    actual = isMatch(s, p)
                    
                    if actual == expected:
                        results["passed_tests"] += 1
                    else:
                        results["failed_tests"] += 1
                        logger.debug(f"   ❌ Test failed: isMatch('{s}', '{p}') = {actual}, expected {expected}")
                        
                except Exception as e:
                    results["runtime_errors"] += 1
                    logger.debug(f"   ❌ Runtime error on test case {test_case}: {e}")
            
        except SyntaxError as e:
            results["syntax_errors"] = 1
            logger.debug(f"   ❌ Syntax error in generated code: {e}")
        except Exception as e:
            results["runtime_errors"] = 1
            logger.debug(f"   ❌ Execution error: {e}")
        
        return results
    
    def _evaluate_planner_task(
        self, 
        prompt: str, 
        task: Dict[str, Any],
        failure_context: Optional[Dict[str, Any]]
    ) -> Tuple[float, int]:
        """Evaluate planner prompt on a planning task."""
        task_description = task.get("description", "")
        expected_elements = task.get("expected_plan_elements", [])
        
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Create a plan for: {task_description}"}
        ]
        
        try:
            response = self.llm_service.invoke(messages, expect_json=True)
            token_count = len(str(response).split())  # Rough token estimate
            
            # Evaluate plan quality
            score = self._score_plan_quality(response, expected_elements, task)
            return score, token_count
            
        except Exception as e:
            logger.warning(f"Planner task evaluation failed: {e}")
            return 0.0, 0
    
    def _evaluate_executor_task(
        self, 
        prompt: str, 
        task: Dict[str, Any],
        failure_context: Optional[Dict[str, Any]]
    ) -> Tuple[float, int]:
        """Evaluate executor prompt on a coding task."""
        task_description = task.get("description", "")
        plan = task.get("plan", {"steps": ["Implement the solution"]})
        expected_code_elements = task.get("expected_code_elements", [])
        
        # Create executor-style prompt
        user_content = (
            f"Original User Request:\n{task_description}\n\n"
            f"Execution Plan:\n{plan.get('steps', 'No specific steps provided in plan.')}\n\n"
            f"Please generate Python code to accomplish this. Ensure you only output the raw Python code, without any markdown formatting or explanations."
        )
        
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_content}
        ]
        
        try:
            response = self.llm_service.invoke(messages, expect_json=False)
            token_count = len(str(response).split())  # Rough token estimate
            
            # Evaluate code quality
            score = self._score_code_quality(response, expected_code_elements, task)
            return score, token_count
            
        except Exception as e:
            logger.warning(f"Executor task evaluation failed: {e}")
            return 0.0, 0
    
    def _score_plan_quality(
        self, 
        plan: Dict[str, Any], 
        expected_elements: List[str], 
        task: Dict[str, Any]
    ) -> float:
        """Score the quality of a generated plan."""
        score = 0.0
        max_score = 1.0
        
        # 🔍 DIAGNOSTIC LOG: Show what we're evaluating
        logger.info(f"🔍 PLAN QUALITY DIAGNOSTIC:")
        logger.info(f"   📋 Task: {task.get('description', 'No description')}")
        logger.info(f"   📋 Plan received: {plan}")
        logger.info(f"   📋 Expected elements: {expected_elements}")
        
        # Check if response is valid JSON with plan structure
        if not isinstance(plan, dict):
            logger.warning(f"   ❌ Plan is not a dict: {type(plan)}")
            return 0.0
        
        plan_steps = plan.get("plan_steps", plan.get("steps", []))
        if not plan_steps:
            logger.warning(f"   ❌ No plan steps found")
            return 0.1  # Minimal score for structural response
        
        logger.info(f"   📋 Plan steps: {plan_steps}")
        
        # Check for expected elements
        plan_text = str(plan).lower()
        element_score = 0.0
        for element in expected_elements:
            if element.lower() in plan_text:
                element_score += 1.0
                logger.info(f"   ✅ Found expected element: '{element}'")
            else:
                logger.info(f"   ❌ Missing expected element: '{element}'")
        
        if expected_elements:
            element_score /= len(expected_elements)
        else:
            element_score = 0.5  # Default score when no expectations
        
        logger.info(f"   📊 Element score: {element_score:.3f} ({element_score * 0.4:.3f} weighted)")
        
        # Check plan completeness (number of steps)
        steps_score = min(1.0, len(plan_steps) / 5.0)  # Normalize to 5 steps
        logger.info(f"   📊 Steps score: {steps_score:.3f} ({len(plan_steps)} steps, weighted: {steps_score * 0.3:.3f})")
        
        # Check step quality (non-empty, descriptive)
        quality_score = 0.0
        for i, step in enumerate(plan_steps):
            step_str = str(step)
            step_len = len(step_str.strip())
            if step_len > 10:  # Reasonable length
                quality_score += 1.0
                logger.info(f"   ✅ Step {i+1} quality OK: '{step_str}' ({step_len} chars)")
            else:
                logger.info(f"   ❌ Step {i+1} too short: '{step_str}' ({step_len} chars)")
        
        if plan_steps:
            quality_score /= len(plan_steps)
        
        logger.info(f"   📊 Quality score: {quality_score:.3f} (weighted: {quality_score * 0.3:.3f})")
        
        # Combine scores
        score = (element_score * 0.4 + steps_score * 0.3 + quality_score * 0.3)
        
        # 🚨 CRITICAL DIAGNOSTIC: Show evaluation method
        if self.task_specific_context:
            logger.info(f"   ✅ USING REAL TASK EVALUATION!")
            logger.info(f"   ✅ Task: {self.task_specific_context['task_description'][:100]}...")
            logger.info(f"   ✅ This evaluates prompts on the ACTUAL regex matching task!")
        else:
            logger.warning(f"   🚨 CRITICAL ISSUE DETECTED:")
            logger.warning(f"   🚨 FITNESS EVALUATOR IS USING SIMPLE VALIDATION TASKS")
            logger.warning(f"   🚨 NOT EVALUATING ACTUAL TASK PERFORMANCE!")
            logger.warning(f"   🚨 Final score: {score:.3f} - This measures prompt STRUCTURE, not TASK SUCCESS!")
            logger.warning(f"   🚨 Bad prompts can score high if they produce STRUCTURED bad plans!")
        
        return min(1.0, score)
    
    def _score_code_quality(
        self, 
        code: str, 
        expected_elements: List[str], 
        task: Dict[str, Any]
    ) -> float:
        """Score the quality of generated code."""
        if not code or not isinstance(code, str):
            return 0.0
        
        score = 0.0
        
        # Check for basic code structure
        code_lower = code.lower()
        
        # Check for expected elements
        element_score = 0.0
        for element in expected_elements:
            if element.lower() in code_lower:
                element_score += 1.0
        
        if expected_elements:
            element_score /= len(expected_elements)
        else:
            element_score = 0.5  # Default score
        
        # Check for Python syntax elements
        syntax_indicators = ["def ", "import ", "return ", "if ", "for ", "while "]
        syntax_score = 0.0
        for indicator in syntax_indicators:
            if indicator in code_lower:
                syntax_score += 1.0
        
        syntax_score = min(1.0, syntax_score / 3.0)  # Normalize
        
        # Check code length (not too short, not excessively long)
        length_score = 0.0
        code_lines = [line.strip() for line in code.split('\n') if line.strip()]
        if 3 <= len(code_lines) <= 50:  # Reasonable range
            length_score = 1.0
        elif len(code_lines) > 0:
            length_score = 0.5
        
        # Combine scores
        score = (element_score * 0.5 + syntax_score * 0.3 + length_score * 0.2)
        return min(1.0, score)
    
    def _evaluate_prompt_coherence(self, prompt: str, agent_type: str) -> float:
        """Evaluate semantic coherence of the prompt."""
        try:
            # Use LLM to evaluate prompt coherence
            evaluation_prompt = f"""
Evaluate the coherence and quality of this {agent_type} agent prompt:

PROMPT TO EVALUATE:
{prompt}

Rate the prompt on these criteria (0.0 to 1.0):
1. Role clarity: How clear is the agent's role and identity?
2. Instruction clarity: How clear and actionable are the instructions?
3. Logical structure: How well-organized and logical is the prompt?
4. Completeness: Does it contain all necessary information?

Respond with JSON:
{{
    "role_clarity": 0.0-1.0,
    "instruction_clarity": 0.0-1.0,
    "logical_structure": 0.0-1.0,
    "completeness": 0.0-1.0,
    "overall_coherence": 0.0-1.0
}}"""

            messages = [
                {"role": "system", "content": "You are an expert prompt evaluator. Provide objective, honest assessments."},
                {"role": "user", "content": evaluation_prompt}
            ]
            
            response = self.llm_service.invoke(messages, expect_json=True)
            
            # Calculate weighted average
            weights = {
                "role_clarity": 0.25,
                "instruction_clarity": 0.35,
                "logical_structure": 0.25,
                "completeness": 0.15
            }
            
            weighted_score = 0.0
            for criterion, weight in weights.items():
                score = response.get(criterion, 0.5)  # Default to middle score
                weighted_score += score * weight
            
            return min(1.0, max(0.0, weighted_score))
            
        except Exception as e:
            logger.debug(f"Coherence evaluation failed: {e}")
            # Fallback: simple heuristic evaluation
            return self._heuristic_coherence_score(prompt, agent_type)
    
    def _heuristic_coherence_score(self, prompt: str, agent_type: str) -> float:
        """Simple heuristic-based coherence evaluation."""
        score = 0.0
        
        # Check for role definition
        if "you are" in prompt.lower():
            score += 0.25
        
        # Check for instructions
        instruction_words = ["should", "must", "need to", "generate", "create", "provide"]
        if any(word in prompt.lower() for word in instruction_words):
            score += 0.25
        
        # Check for reasonable length
        if 50 <= len(prompt.split()) <= 500:
            score += 0.25
        
        # Check for agent-specific terms
        if agent_type == "PLANNER" and any(word in prompt.lower() for word in ["plan", "strategy", "steps"]):
            score += 0.25
        elif agent_type == "EXECUTOR" and any(word in prompt.lower() for word in ["code", "implement", "generate"]):
            score += 0.25
        
        return score
    
    def _calculate_combined_fitness(self, metrics: FitnessMetrics) -> float:
        """Calculate weighted combination of fitness components."""
        
        # 🔍 DIAGNOSTIC LOG: Show fitness calculation breakdown
        logger.warning(f"🔍 FITNESS CALCULATION DIAGNOSTIC:")
        logger.warning(f"   📊 Success Rate: {metrics.success_rate:.3f} (weight: {self.config.success_rate_weight})")
        logger.warning(f"   📊 Efficiency: {metrics.efficiency:.3f} (weight: {self.config.efficiency_weight})")
        logger.warning(f"   📊 Coherence: {metrics.coherence:.3f} (weight: {self.config.coherence_weight})")
        logger.warning(f"   📊 Generalization: {metrics.generalization:.3f} (weight: {self.config.generalization_weight})")
        logger.warning(f"   📊 Task Scores: {metrics.task_scores}")
        
        fitness = (
            metrics.success_rate * self.config.success_rate_weight +
            metrics.efficiency * self.config.efficiency_weight +
            metrics.coherence * self.config.coherence_weight +
            metrics.generalization * self.config.generalization_weight
        )
        
        # 🚨 CRITICAL DIAGNOSTIC: Show weight sum issue
        weight_sum = (self.config.success_rate_weight + self.config.efficiency_weight + 
                     self.config.coherence_weight + self.config.generalization_weight)
        logger.warning(f"   🚨 WEIGHT SUM ISSUE: {weight_sum:.3f} (should be ~1.0)")
        logger.warning(f"   🚨 Raw fitness before clamping: {fitness:.3f}")
        
        if self.task_specific_context:
            logger.info(f"   ✅ SUCCESS RATE = actual task performance on regex matching!")
            logger.info(f"   ✅ USING REAL MULTI-AGENT PIPELINE EVALUATION!")
        else:
            logger.warning(f"   🚨 SUCCESS RATE = simple validation on factorial/max/palindrome tasks")
            logger.warning(f"   🚨 NOT ACTUAL REGEX MATCHING TASK PERFORMANCE!")
        
        final_fitness = min(1.0, max(0.0, fitness))
        logger.warning(f"   🚨 Final fitness: {final_fitness:.3f}")
        
        return final_fitness
    

    def validate_top_prompts(self, prompts_and_scores: List[tuple], agent_type: str) -> List[tuple]:
        """
        Re-evaluate top prompts to ensure fitness scores are accurate.
        
        Args:
            prompts_and_scores: List of (prompt, score) tuples
            agent_type: Type of agent
            
        Returns:
            List of (prompt, validated_score) tuples, sorted by validated score
        """
        logger.info(f"🔍 VALIDATION: Re-evaluating top {len(prompts_and_scores)} prompts")
        
        validated_results = []
        for prompt, original_score in prompts_and_scores:
            # Clear cache for this prompt to force re-evaluation
            cache_key = self._get_cache_key(prompt, agent_type)
            if cache_key in self.evaluation_cache:
                del self.evaluation_cache[cache_key]
            
            # Re-evaluate
            validated_score = self.evaluate_prompt(prompt, agent_type, None)
            validated_results.append((prompt, validated_score))
            
            score_diff = validated_score - original_score
            if abs(score_diff) > 0.1:
                logger.warning(f"   🚨 SCORE CHANGE: {original_score:.3f} → {validated_score:.3f} (Δ{score_diff:+.3f})")
            else:
                logger.info(f"   ✅ SCORE STABLE: {validated_score:.3f}")
        
        # Sort by validated score
        validated_results.sort(key=lambda x: x[1], reverse=True)
        return validated_results

    def _get_cache_key(self, prompt: str, agent_type: str, failure_context: Optional[Dict[str, Any]] = None) -> str:
        """Generate cache key for prompt evaluation."""
        # Use full hash to avoid collisions + include task context
        import hashlib
        # Include task context in cache key to avoid cross-task contamination
        task_context = getattr(self, 'task_specific_context', {})
        task_desc = task_context.get('task_description', 'generic') if task_context else 'generic'
        
        # Include failure context to ensure different failure scenarios are cached separately
        failure_key = ""
        if failure_context:
            # Create a stable key from failure context
            failure_items = []
            for key in sorted(failure_context.keys()):
                value = str(failure_context[key])[:100]  # Limit length
                failure_items.append(f"{key}:{value}")
            failure_key = "|".join(failure_items)
        
        # Use SHA256 for better collision resistance
        combined_content = f"{agent_type}|{prompt}|{task_desc}|{failure_key}"
        prompt_hash = hashlib.sha256(combined_content.encode()).hexdigest()[:32]
        return f"{agent_type}_{prompt_hash}"
    
    def _create_default_validation_tasks(self) -> List[Dict[str, Any]]:
        """Create default validation tasks for fitness evaluation."""
        return [
            {
                "description": "Create a function to calculate the factorial of a number",
                "expected_plan_elements": ["function", "factorial", "recursion", "iteration"],
                "expected_code_elements": ["def", "factorial", "return"],
                "plan": {"steps": ["Define function", "Implement calculation", "Return result"]}
            },
            {
                "description": "Implement a function to find the maximum element in a list",
                "expected_plan_elements": ["function", "maximum", "list", "iterate"],
                "expected_code_elements": ["def", "max", "list", "return"],
                "plan": {"steps": ["Define function", "Iterate through list", "Track maximum", "Return result"]}
            },
            {
                "description": "Write a function to check if a string is a palindrome",
                "expected_plan_elements": ["function", "palindrome", "string", "comparison"],
                "expected_code_elements": ["def", "palindrome", "string", "return"],
                "plan": {"steps": ["Define function", "Compare characters", "Return boolean result"]}
            }
        ]
    
    def _evaluate_planner_task_with_pipeline(
        self, 
        prompt: str, 
        task: Dict[str, Any],
        failure_context: Optional[Dict[str, Any]]
    ) -> Tuple[float, int]:
        """Evaluate planner prompt using full multi-agent pipeline on actual task."""
        if not (self.task_specific_context and self.planner_agent and self.executor_agent and self.critic_agent):
            # Fallback to original evaluation
            return self._evaluate_planner_task(prompt, task, failure_context)
        
        task_description = self.task_specific_context["task_description"]
        
        try:
            # Temporarily set the planner's prompt
            original_prompt = self.planner_agent.system_prompt
            self.planner_agent.system_prompt = prompt
            
            logger.info(f"🧪 PIPELINE TEST: Evaluating planner prompt on real task")
            
            # Step 1: Planning with evolved prompt
            plan = self.planner_agent.run(user_request=task_description)
            if isinstance(plan, dict) and plan.get("error"):
                logger.warning(f"   ❌ Planning failed: {plan.get('error')}")
                return 0.0, 100
            
            # Step 2: Execution
            code = self.executor_agent.run(plan=plan, original_request=task_description)
            if isinstance(code, dict) and code.get("error"):
                logger.warning(f"   ❌ Execution failed: {code.get('error')}")
                return 0.1, 150
                
            # Step 3: Evaluation with critic
            critique = self.critic_agent.run(
                generated_code=code,
                task_description=task_description,
                plan=plan
            )
            
            if isinstance(critique, dict):
                score = critique.get('quantitative_score', critique.get('score', 0.0))
                status = critique.get('overall_status', critique.get('status', 'UNKNOWN'))
                
                logger.info(f"   ✅ PIPELINE RESULT: Score {score:.3f}, Status: {status}")
                
                # Convert critic score to fitness score
                fitness_score = max(0.0, min(1.0, score))
                token_count = len(str(plan).split()) + len(str(code).split())
                
                return fitness_score, token_count
            else:
                logger.warning(f"   ❌ Critic evaluation failed")
                return 0.0, 200
                
        except Exception as e:
            logger.error(f"   ❌ Pipeline evaluation failed: {e}")
            return 0.0, 100
        finally:
            # Restore original prompt
            if 'original_prompt' in locals():
                self.planner_agent.system_prompt = original_prompt
    
    def _evaluate_executor_task_with_pipeline(
        self, 
        prompt: str, 
        task: Dict[str, Any],
        failure_context: Optional[Dict[str, Any]]
    ) -> Tuple[float, int]:
        """Evaluate executor prompt using full multi-agent pipeline on actual task."""
        if not (self.task_specific_context and self.planner_agent and self.executor_agent and self.critic_agent):
            # Fallback to original evaluation
            return self._evaluate_executor_task(prompt, task, failure_context)
            
        task_description = self.task_specific_context["task_description"]
        
        try:
            # Temporarily set the executor's prompt
            original_prompt = self.executor_agent.system_prompt
            self.executor_agent.system_prompt = prompt
            
            logger.info(f"🧪 PIPELINE TEST: Evaluating executor prompt on real task")
            
            # Step 1: Get a plan (use existing planner)
            plan = self.planner_agent.run(user_request=task_description)
            if isinstance(plan, dict) and plan.get("error"):
                logger.warning(f"   ❌ Planning failed: {plan.get('error')}")
                return 0.0, 100
            
            # Step 2: Execution with evolved prompt
            code = self.executor_agent.run(plan=plan, original_request=task_description)
            if isinstance(code, dict) and code.get("error"):
                logger.warning(f"   ❌ Execution failed: {code.get('error')}")
                return 0.1, 150
                
            # Step 3: Evaluation with critic
            critique = self.critic_agent.run(
                generated_code=code,
                task_description=task_description,
                plan=plan
            )
            
            if isinstance(critique, dict):
                score = critique.get('quantitative_score', critique.get('score', 0.0))
                status = critique.get('overall_status', critique.get('status', 'UNKNOWN'))
                
                logger.info(f"   ✅ PIPELINE RESULT: Score {score:.3f}, Status: {status}")
                
                # Convert critic score to fitness score
                fitness_score = max(0.0, min(1.0, score))
                token_count = len(str(plan).split()) + len(str(code).split())
                
                return fitness_score, token_count
            else:
                logger.warning(f"   ❌ Critic evaluation failed")
                return 0.0, 200
                
        except Exception as e:
            logger.error(f"   ❌ Pipeline evaluation failed: {e}")
            return 0.0, 100
        finally:
            # Restore original prompt
            if 'original_prompt' in locals():
                self.executor_agent.system_prompt = original_prompt 