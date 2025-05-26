"""
Evolution operators for prompt optimization using LLM-guided crossover and mutation.
"""

import logging
import random
import re
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PromptComponent:
    """Represents a semantic component of a prompt."""
    role_definition: str = ""
    instructions: str = ""
    examples: str = ""
    constraints: str = ""
    output_format: str = ""


class EvolutionOperators:
    """
    LLM-guided evolution operators for semantic crossover and mutation with error-targeted improvements.
    """
    
    def __init__(self, llm_service):
        self.llm_service = llm_service
        self._crossover_prompt = self._build_crossover_prompt()
        self._mutation_prompt = self._build_mutation_prompt()
        self._component_extraction_prompt = self._build_component_extraction_prompt()
        self._coherence_validation_prompt = self._build_coherence_validation_prompt()
        self._error_targeted_prompt = self._build_error_targeted_prompt()
    
    def crossover(self, parent1: str, parent2: str, agent_type: str, failure_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Perform LLM-guided semantic crossover between two parent prompts with error targeting.
        
        Args:
            parent1: First parent prompt
            parent2: Second parent prompt
            agent_type: Type of agent (PLANNER or EXECUTOR)
            failure_context: Context about the original failure to target
            
        Returns:
            Offspring prompt combining best elements from both parents while addressing errors
        """
        logger.info(f"🧬 CROSSOVER: Performing error-targeted semantic crossover for {agent_type}")
        
        try:
            # Extract components from both parents
            components1 = self._extract_prompt_components(parent1, agent_type)
            components2 = self._extract_prompt_components(parent2, agent_type)
            
            # Perform error-targeted recombination
            if failure_context:
                offspring = self._error_targeted_recombination(
                    components1, components2, agent_type, parent1, parent2, failure_context
                )
            else:
                offspring = self._semantic_recombination(
                    components1, components2, agent_type, parent1, parent2
                )
            
            # Validate coherence
            if self._validate_coherence(offspring, agent_type):
                logger.info("✅ CROSSOVER: Generated coherent offspring")
                return offspring
            else:
                logger.warning("⚠️ CROSSOVER: Offspring failed coherence check, using fallback")
                return self._fallback_crossover(parent1, parent2)
                
        except Exception as e:
            logger.error(f"❌ CROSSOVER ERROR: {e}")
            return self._fallback_crossover(parent1, parent2)
    
    def mutate(self, prompt: str, agent_type: str, mutation_rate: float = 0.3, failure_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Perform error-targeted semantic mutation on a prompt.
        
        Args:
            prompt: Original prompt to mutate
            agent_type: Type of agent (PLANNER or EXECUTOR)
            mutation_rate: Probability of applying each mutation type
            failure_context: Context about the original failure to target
            
        Returns:
            Mutated prompt with preserved semantic meaning and targeted error fixes
        """
        logger.info(f"🔀 MUTATION: Performing error-targeted mutation for {agent_type}")
        
        try:
            if failure_context:
                # Apply error-targeted mutations
                mutated_prompt = self._apply_error_targeted_mutations(
                    prompt, agent_type, failure_context, mutation_rate
                )
            else:
                # Apply general mutations
                mutated_prompt = self._apply_general_mutations(
                    prompt, agent_type, mutation_rate
                )
            
            # Validate coherence
            if self._validate_coherence(mutated_prompt, agent_type):
                logger.info(f"✅ MUTATION: Applied targeted mutations successfully")
                return mutated_prompt
            else:
                logger.warning("⚠️ MUTATION: Mutated prompt failed coherence check")
                return self._minor_mutation(prompt, agent_type)
                
        except Exception as e:
            logger.error(f"❌ MUTATION ERROR: {e}")
            return self._minor_mutation(prompt, agent_type)
    
    def _error_targeted_recombination(
        self, 
        comp1: PromptComponent, 
        comp2: PromptComponent, 
        agent_type: str,
        parent1: str,
        parent2: str,
        failure_context: Dict[str, Any]
    ) -> str:
        """Perform error-targeted recombination that addresses specific failure patterns."""
        
        # Extract error information
        error_type = self._analyze_error_type(failure_context)
        error_details = self._extract_error_details(failure_context)
        
        recombination_prompt = f"""
Create an improved {agent_type} prompt by intelligently combining the best elements from two parent prompts while specifically addressing the identified failure pattern.

FAILURE ANALYSIS:
Error Type: {error_type}
Error Details: {error_details}
Original Task: {failure_context.get('original_task', 'Not specified')}

PARENT 1 COMPONENTS:
- Role Definition: {comp1.role_definition}
- Instructions: {comp1.instructions}
- Examples: {comp1.examples}
- Constraints: {comp1.constraints}
- Output Format: {comp1.output_format}

PARENT 2 COMPONENTS:
- Role Definition: {comp2.role_definition}
- Instructions: {comp2.instructions}
- Examples: {comp2.examples}
- Constraints: {comp2.constraints}
- Output Format: {comp2.output_format}

ORIGINAL PARENT PROMPTS:
Parent 1: {parent1}
Parent 2: {parent2}

ERROR-TARGETED RECOMBINATION GUIDELINES:
1. Identify which parent components better address the specific error type
2. Combine elements that strengthen the agent's capability in the failure area
3. Enhance instructions to prevent the specific failure pattern
4. Add relevant constraints or examples that guide away from the error
5. Maintain overall prompt coherence while targeting the weakness

SPECIFIC IMPROVEMENTS NEEDED:
{self._get_error_specific_improvements(error_type, agent_type, failure_context)}

Create a coherent, improved prompt that inherits strengths from both parents while specifically addressing the failure pattern:"""

        try:
            messages = [
                {"role": "system", "content": self._error_targeted_prompt},
                {"role": "user", "content": recombination_prompt}
            ]
            
            result = self.llm_service.invoke(messages, expect_json=False)
            logger.info(f"✅ ERROR-TARGETED CROSSOVER: Generated prompt targeting {error_type}")
            return result
            
        except Exception as e:
            logger.error(f"Error-targeted recombination failed: {e}")
            return self._semantic_recombination(comp1, comp2, agent_type, parent1, parent2)
    
    def _apply_error_targeted_mutations(
        self, 
        prompt: str, 
        agent_type: str, 
        failure_context: Dict[str, Any], 
        mutation_rate: float
    ) -> str:
        """Apply mutations specifically targeting the identified error patterns."""
        
        logger.info(f"🔀 ERROR-TARGETED MUTATION: Starting targeted mutation process")
        logger.info(f"   📊 Input Prompt Length: {len(prompt)} characters")
        logger.info(f"   🎯 Agent Type: {agent_type}")
        logger.info(f"   📈 Mutation Rate: {mutation_rate}")
        
        error_type = self._analyze_error_type(failure_context)
        error_details = self._extract_error_details(failure_context)
        
        logger.info(f"   🔍 Analyzed Error Type: {error_type}")
        logger.info(f"   📝 Error Details Length: {len(error_details)} characters")
        
        # Select mutations based on error type
        targeted_mutations = self._select_error_targeted_mutations(error_type, agent_type, failure_context)
        
        logger.info(f"   🎯 Targeted Mutations: {targeted_mutations}")
        
        mutation_prompt = f"""
Improve this {agent_type} prompt by applying targeted modifications to address the specific failure pattern.

CURRENT PROMPT:
{prompt}

FAILURE ANALYSIS:
Error Type: {error_type}
Error Details: {error_details}
Original Task: {failure_context.get('original_task', 'Not specified')}

TARGETED IMPROVEMENT STRATEGY:
{self._get_error_specific_improvements(error_type, agent_type, failure_context)}

MUTATION GUIDELINES:
1. Holistically improve the prompt to address the root cause of the failure
2. Strengthen the agent's capabilities in the specific failure area
3. Add clarity and specificity where the error suggests confusion
4. Enhance instructions to guide the agent away from the failure pattern
5. Maintain the core identity and functionality of the agent
6. Do NOT simply append error descriptions - integrate improvements naturally

Apply the following targeted mutations:
{', '.join(targeted_mutations)}

Generate the improved prompt that addresses the failure while maintaining coherence:"""

        logger.info(f"   📋 Generated Mutation Prompt Length: {len(mutation_prompt)} characters")
        logger.info(f"   🔍 DIAGNOSTIC: Mutation prompt contains error_type='{error_type}' and mutations={targeted_mutations}")
        
        # 🆕 LOG CRITICAL CONTEXT THAT MIGHT BE MISSING
        original_task = failure_context.get('original_task', 'Not specified')
        logger.info(f"   📋 Original Task in Context: {original_task[:100]}...")
        
        # Check if we have specific test failure information
        if 'Failed Tests:' in error_details:
            logger.info(f"   ✅ GOOD: Failed test cases are included in error details")
        else:
            logger.warning(f"   ⚠️ MISSING: No specific failed test cases in error details")
            logger.warning(f"   ⚠️ This means prompts won't target specific failure patterns")

        try:
            messages = [
                {"role": "system", "content": self._error_targeted_prompt},
                {"role": "user", "content": mutation_prompt}
            ]
            
            logger.info(f"   🤖 Invoking LLM for error-targeted mutation...")
            result = self.llm_service.invoke(messages, expect_json=False)
            
            logger.info(f"   ✅ Generated Improved Prompt Length: {len(result)} characters")
            logger.info(f"   📈 Length Change: {len(result) - len(prompt):+d} characters")
            logger.info(f"   📝 Improved Prompt Preview: {result[:150]}...")
            
            logger.info(f"✅ ERROR-TARGETED MUTATION: Applied {len(targeted_mutations)} targeted mutations for {error_type}")
            return result
            
        except Exception as e:
            logger.error(f"Error-targeted mutation failed: {e}")
            logger.warning(f"   🔄 Falling back to general mutations")
            return self._apply_general_mutations(prompt, agent_type, mutation_rate)
    
    def _apply_general_mutations(self, prompt: str, agent_type: str, mutation_rate: float) -> str:
        """Apply general mutations when no specific error context is available."""
        # Determine which mutation operations to apply
        mutations = []
        if random.random() < mutation_rate:
            mutations.append("instruction_refinement")
        if random.random() < mutation_rate * 0.8:  # Slightly lower probability
            mutations.append("example_enhancement")
        if random.random() < mutation_rate * 0.8:
            mutations.append("structure_optimization")
        if random.random() < mutation_rate * 0.6:  # Lower probability for role changes
            mutations.append("role_enhancement")
        
        if not mutations:
            mutations = ["instruction_refinement"]  # At least one mutation
        
        # Apply mutations sequentially
        mutated_prompt = prompt
        for mutation_type in mutations:
            mutated_prompt = self._apply_mutation(
                mutated_prompt, mutation_type, agent_type
            )
        
        return mutated_prompt
    
    def _analyze_error_type(self, failure_context: Dict[str, Any]) -> str:
        """Analyze the failure context to determine the primary error type."""
        if not failure_context:
            logger.warning("🚨 ERROR ANALYSIS: No failure context provided - using general_failure")
            return "general_failure"
        
        logger.info("🔍 ERROR ANALYSIS: Starting detailed failure context analysis")
        logger.info(f"   📋 Failure Context Keys: {list(failure_context.keys())}")
        
        # Check for specific error patterns
        error_report = failure_context.get('failure_report', {})
        classification = failure_context.get('classification', {})
        
        logger.info(f"   📊 Error Report Type: {type(error_report)}")
        logger.info(f"   📊 Classification Type: {type(classification)}")
        
        # Look for specific failure types
        if classification:
            primary_failure = classification.get('primary_failure_type', 'unknown')
            logger.info(f"   🎯 LLM Classification Primary Failure: {primary_failure}")
            
            # Log detailed LLM reasoning if available
            reasoning = classification.get('reasoning', [])
            if reasoning:
                logger.info(f"   🧠 LLM Detailed Reasoning:")
                for i, reason in enumerate(reasoning[:3], 1):
                    logger.info(f"      {i}. {reason}")
            
            if primary_failure != 'unknown':
                logger.info(f"   ✅ Using LLM classification: {primary_failure}")
                return primary_failure
        
        # Analyze error report content
        if isinstance(error_report, dict):
            status = error_report.get('overall_status', error_report.get('status', ''))
            feedback = str(error_report.get('feedback', ''))
            
            logger.info(f"   📝 Error Report Status: {status}")
            logger.info(f"   📝 Error Report Feedback: {feedback[:200]}...")
            
            # Pattern matching for common error types
            if 'syntax' in feedback.lower() or 'syntaxerror' in feedback.lower():
                logger.info(f"   🔍 Pattern Match: syntax_error")
                return "syntax_error"
            elif 'logic' in feedback.lower() or 'incorrect' in feedback.lower():
                logger.info(f"   🔍 Pattern Match: logic_error")
                return "logic_error"
            elif 'incomplete' in feedback.lower() or 'missing' in feedback.lower():
                logger.info(f"   🔍 Pattern Match: incomplete_implementation")
                return "incomplete_implementation"
            elif 'edge case' in feedback.lower() or 'boundary' in feedback.lower():
                logger.info(f"   🔍 Pattern Match: edge_case_failure")
                return "edge_case_failure"
            elif 'performance' in feedback.lower() or 'timeout' in feedback.lower():
                logger.info(f"   🔍 Pattern Match: performance_issue")
                return "performance_issue"
            elif 'understanding' in feedback.lower() or 'misunderstood' in feedback.lower():
                logger.info(f"   🔍 Pattern Match: task_misunderstanding")
                return "task_misunderstanding"
        
        logger.warning(f"   ⚠️ No specific error pattern found - using general_failure")
        return "general_failure"
    
    def _extract_error_details(self, failure_context: Dict[str, Any]) -> str:
        """Extract specific error details from the failure context."""
        if not failure_context:
            logger.warning("🚨 ERROR DETAILS: No failure context provided")
            return "No specific error details available"
        
        logger.info("🔍 ERROR DETAILS: Extracting specific error information")
        details = []
        
        # 🆕 PRIORITY: Extract specific test failures first
        specific_test_failures = failure_context.get('specific_test_failures', [])
        if specific_test_failures:
            logger.info(f"   🎯 FOUND SPECIFIC TEST FAILURES: {len(specific_test_failures)} tests")
            test_details = []
            for i, test_failure in enumerate(specific_test_failures, 1):
                test_name = test_failure.get('test_name', f'Test_{i}')
                inputs = test_failure.get('inputs', {})
                expected = test_failure.get('expected_output', 'unknown')
                actual = test_failure.get('actual_output', 'unknown')
                
                test_detail = f"Test '{test_name}': inputs={inputs}, expected={expected}, actual={actual}"
                test_details.append(test_detail)
                
                # 🎯 SPECIAL HANDLING FOR REGEX PATTERNS
                if 'regex_pattern_issue' in test_failure:
                    regex_issue = test_failure['regex_pattern_issue']
                    pattern = regex_issue['pattern']
                    string = regex_issue['string']
                    issue_type = regex_issue.get('likely_issue', 'unknown')
                    
                    regex_detail = f"REGEX PATTERN FAILURE: string='{string}' pattern='{pattern}' issue={issue_type}"
                    test_details.append(regex_detail)
                    logger.info(f"   🎯 REGEX SPECIFIC: {regex_detail}")
            
            if test_details:
                details.append(f"Failed Tests: {'; '.join(test_details)}")
                logger.info(f"   ✅ SPECIFIC TEST FAILURES: Added {len(test_details)} detailed test failures")
        else:
            logger.warning(f"   ⚠️ NO SPECIFIC TEST FAILURES: Will use generic error details")
        
        # Extract from error report
        error_report = failure_context.get('failure_report', {})
        if isinstance(error_report, dict):
            feedback = error_report.get('feedback', '')
            if feedback:
                details.append(f"Feedback: {feedback}")
                logger.info(f"   📝 Extracted Feedback: {feedback[:100]}...")
            
            score = error_report.get('quantitative_score', error_report.get('score'))
            if score is not None:
                details.append(f"Score: {score}")
                logger.info(f"   📊 Extracted Score: {score}")
        
        # Extract from classification
        classification = failure_context.get('classification', {})
        if isinstance(classification, dict):
            reasoning = classification.get('reasoning', [])
            if reasoning:
                reasoning_text = '; '.join(reasoning[:3])  # Limit to first 3 reasons
                details.append(f"Reasoning: {reasoning_text}")
                logger.info(f"   🧠 Extracted Reasoning: {reasoning_text}")
        
        # Extract task information
        task = failure_context.get('original_task', '')
        if task:
            task_snippet = task[:100] + "..." if len(task) > 100 else task
            details.append(f"Task: {task_snippet}")
            logger.info(f"   📋 Extracted Task: {task_snippet}")
        
        result = "; ".join(details) if details else "No specific error details available"
        logger.info(f"   ✅ Final Error Details ({len(result)} chars): {result[:150]}...")
        return result
    
    def _select_error_targeted_mutations(self, error_type: str, agent_type: str, failure_context: Optional[Dict[str, Any]] = None) -> List[str]:
        """Select specific mutations based on the error type, agent type, and specific test failures."""
        logger.info(f"🎯 MUTATION SELECTION: Selecting mutations for error_type='{error_type}', agent_type='{agent_type}'")
        
        # 🆕 ENHANCED: Use specific test failures for targeted mutation selection
        targeted_mutations = []
        
        if failure_context and 'specific_test_failures' in failure_context:
            specific_test_failures = failure_context['specific_test_failures']
            logger.info(f"   🎯 ANALYZING SPECIFIC TEST FAILURES: {len(specific_test_failures)} tests")
            
            for test_failure in specific_test_failures:
                # 🎯 REGEX-SPECIFIC MUTATIONS
                if 'regex_pattern_issue' in test_failure:
                    regex_issue = test_failure['regex_pattern_issue']
                    pattern = regex_issue['pattern']
                    
                    if '*' in pattern:
                        targeted_mutations.extend([
                            "regex_star_operator_focus",
                            "zero_match_handling",
                            "pattern_matching_precision"
                        ])
                        logger.info(f"   🎯 REGEX STAR ISSUE: Added regex-specific mutations for pattern '{pattern}'")
                
                # Check for specific input patterns that suggest certain mutation types
                inputs = test_failure.get('inputs', {})
                if isinstance(inputs, dict):
                    # Edge case detection
                    if any(val == "" or val == 0 or val is None for val in inputs.values()):
                        targeted_mutations.append("edge_case_awareness")
                        logger.info(f"   🎯 EDGE CASE DETECTED: Added edge case mutations for inputs {inputs}")
                    
                    # Complex input detection
                    if len(inputs) > 2 or any(isinstance(val, (list, dict)) for val in inputs.values()):
                        targeted_mutations.append("complex_input_handling")
                        logger.info(f"   🎯 COMPLEX INPUT: Added complex input mutations for {inputs}")
        
        # Base mutation map
        base_mutation_map = {
            "syntax_error": ["syntax_focus", "code_structure_enhancement", "validation_strengthening"],
            "logic_error": ["logic_clarification", "algorithm_guidance", "step_by_step_enhancement"],
            "incomplete_implementation": ["completeness_focus", "requirement_emphasis", "thoroughness_enhancement"],
            "edge_case_failure": ["edge_case_awareness", "boundary_handling", "robustness_enhancement"],
            "performance_issue": ["efficiency_focus", "optimization_guidance", "complexity_awareness"],
            "task_misunderstanding": ["clarity_enhancement", "requirement_emphasis", "example_enrichment"],
            "general_failure": ["instruction_refinement", "example_enhancement", "structure_optimization"]
        }
        
        # Add base mutations for the error type
        base_mutations = base_mutation_map.get(error_type, base_mutation_map["general_failure"])
        targeted_mutations.extend(base_mutations)
        
        # Remove duplicates while preserving order
        final_mutations = []
        seen = set()
        for mutation in targeted_mutations:
            if mutation not in seen:
                final_mutations.append(mutation)
                seen.add(mutation)
        
        logger.info(f"   📋 Final Selected Mutations: {final_mutations}")
        if failure_context and 'specific_test_failures' in failure_context:
            logger.info(f"   🎯 INCLUDES TEST-SPECIFIC MUTATIONS: Yes")
        else:
            logger.info(f"   ⚠️ GENERIC MUTATIONS ONLY: No specific test failures available")
        
        return final_mutations
    
    def _get_error_specific_improvements(self, error_type: str, agent_type: str, failure_context: Optional[Dict[str, Any]] = None) -> str:
        """Get specific improvement guidelines based on error type, agent type, and specific test failures."""
        logger.info(f"🎯 IMPROVEMENT GUIDELINES: Generating for error_type='{error_type}', agent_type='{agent_type}'")
        
        # 🆕 ENHANCED: Use specific test failures for targeted improvements
        specific_guidance = ""
        if failure_context and 'specific_test_failures' in failure_context:
            specific_test_failures = failure_context['specific_test_failures']
            logger.info(f"   🎯 USING SPECIFIC TEST FAILURES: {len(specific_test_failures)} tests for targeted guidance")
            
            test_guidance = []
            for test_failure in specific_test_failures:
                inputs = test_failure.get('inputs', {})
                expected = test_failure.get('expected_output', 'unknown')
                actual = test_failure.get('actual_output', 'unknown')
                
                # 🎯 REGEX-SPECIFIC GUIDANCE
                if 'regex_pattern_issue' in test_failure:
                    regex_issue = test_failure['regex_pattern_issue']
                    pattern = regex_issue['pattern']
                    string = regex_issue['string']
                    
                    if '*' in pattern:
                        test_guidance.append(f"""
- CRITICAL REGEX ISSUE: Pattern '{pattern}' with string '{string}' - Expected: {expected}, Got: {actual}
- The '*' operator must match ZERO OR MORE of the preceding element
- For pattern '{pattern}': ensure correct handling when '*' matches zero occurrences
- Test case specifically: string='{string}' should {'match' if expected else 'not match'} pattern='{pattern}'""")
                
                # Generic test failure guidance
                else:
                    test_guidance.append(f"""
- SPECIFIC TEST FAILURE: Input {inputs} - Expected: {expected}, Got: {actual}
- Ensure the implementation correctly handles this exact input case
- Review the logic for processing input: {inputs}""")
            
            if test_guidance:
                specific_guidance = f"""
SPECIFIC TEST FAILURE TARGETING:
{''.join(test_guidance)}

TARGETED IMPROVEMENTS NEEDED:"""
                logger.info(f"   ✅ GENERATED SPECIFIC GUIDANCE: {len(specific_guidance)} chars")
        else:
            logger.warning(f"   ⚠️ NO SPECIFIC TEST FAILURES: Using generic guidance")
        
        # Base improvements by error type
        base_improvements = {
            "syntax_error": f"""
- Emphasize proper {agent_type.lower()} syntax and structure
- Add explicit guidance on code formatting and syntax rules
- Include validation steps to catch syntax issues
- Strengthen attention to detail in code generation""",
            
            "logic_error": f"""
- Enhance logical reasoning and algorithm design capabilities
- Add step-by-step problem decomposition guidance
- Strengthen understanding of problem requirements
- Include verification steps for logical correctness""",
            
            "incomplete_implementation": f"""
- Emphasize thoroughness and completeness in {agent_type.lower()} output
- Add explicit requirements checking
- Strengthen attention to all aspects of the task
- Include completeness validation steps""",
            
            "edge_case_failure": f"""
- Enhance awareness of edge cases and boundary conditions
- Add explicit edge case consideration steps
- Strengthen robustness and defensive programming
- Include comprehensive testing mindset""",
            
            "performance_issue": f"""
- Focus on efficiency and optimization in {agent_type.lower()} approach
- Add time and space complexity awareness
- Strengthen algorithmic thinking
- Include performance consideration steps""",
            
            "task_misunderstanding": f"""
- Enhance task comprehension and requirement analysis
- Add explicit requirement clarification steps
- Strengthen communication and understanding
- Include verification of task understanding""",
            
            "general_failure": f"""
- Improve overall {agent_type.lower()} capability and reliability
- Enhance attention to detail and quality
- Strengthen problem-solving approach
- Include comprehensive validation steps"""
        }
        
        base_improvement = base_improvements.get(error_type, base_improvements["general_failure"])
        
        # Combine specific and base guidance
        final_guidance = specific_guidance + base_improvement
        
        logger.info(f"   📝 Final Improvement Guidelines ({len(final_guidance)} chars)")
        if specific_guidance:
            logger.info(f"   🎯 INCLUDES SPECIFIC TEST TARGETING: Yes")
        else:
            logger.info(f"   ⚠️ GENERIC GUIDANCE ONLY: No specific test failures available")
        
        return final_guidance
    
    def _extract_prompt_components(self, prompt: str, agent_type: str) -> PromptComponent:
        """Extract semantic components from a prompt using LLM analysis."""
        extraction_prompt = f"""
Analyze the following {agent_type} prompt and extract its key components:

PROMPT TO ANALYZE:
{prompt}

Extract and identify these components:
1. Role Definition: How the agent's identity/role is defined
2. Instructions: Core task instructions and directives  
3. Examples: Any demonstrations or examples provided
4. Constraints: Limitations, rules, or requirements
5. Output Format: Specifications for response format

Respond in JSON format:
{{
    "role_definition": "extracted role definition text",
    "instructions": "extracted instructions text", 
    "examples": "extracted examples text",
    "constraints": "extracted constraints text",
    "output_format": "extracted output format specifications"
}}
"""
        
        try:
            messages = [
                {"role": "system", "content": self._component_extraction_prompt},
                {"role": "user", "content": extraction_prompt}
            ]
            
            response = self.llm_service.invoke(messages, expect_json=True)
            
            return PromptComponent(
                role_definition=response.get("role_definition", ""),
                instructions=response.get("instructions", ""),
                examples=response.get("examples", ""),
                constraints=response.get("constraints", ""),
                output_format=response.get("output_format", "")
            )
            
        except Exception as e:
            logger.error(f"Component extraction failed: {e}")
            # Fallback: simple text splitting
            return self._simple_component_extraction(prompt)
    
    def _semantic_recombination(
        self, 
        comp1: PromptComponent, 
        comp2: PromptComponent, 
        agent_type: str,
        parent1: str,
        parent2: str
    ) -> str:
        """Perform LLM-guided semantic recombination of components."""
        
        recombination_prompt = f"""
Create an improved {agent_type} prompt by intelligently combining the best elements from two parent prompts.

PARENT 1 COMPONENTS:
- Role Definition: {comp1.role_definition}
- Instructions: {comp1.instructions}
- Examples: {comp1.examples}
- Constraints: {comp1.constraints}
- Output Format: {comp1.output_format}

PARENT 2 COMPONENTS:
- Role Definition: {comp2.role_definition}
- Instructions: {comp2.instructions}
- Examples: {comp2.examples}
- Constraints: {comp2.constraints}
- Output Format: {comp2.output_format}

ORIGINAL PARENT PROMPTS:
Parent 1: {parent1}
Parent 2: {parent2}

RECOMBINATION GUIDELINES:
1. Select the clearest and most effective role definition
2. Combine the best instructional elements from both parents
3. Include the most helpful examples and demonstrations
4. Merge complementary constraints without redundancy
5. Use the most appropriate output format specification

Create a coherent, improved prompt that inherits the strengths of both parents while maintaining {agent_type} agent functionality.

Generate the complete improved prompt:"""

        try:
            messages = [
                {"role": "system", "content": self._crossover_prompt},
                {"role": "user", "content": recombination_prompt}
            ]
            
            return self.llm_service.invoke(messages, expect_json=False)
            
        except Exception as e:
            logger.error(f"Semantic recombination failed: {e}")
            return self._fallback_crossover(parent1, parent2)
    
    def _apply_mutation(self, prompt: str, mutation_type: str, agent_type: str) -> str:
        """Apply a specific type of mutation to the prompt."""
        
        mutation_prompts = {
            "instruction_refinement": f"""
Refine the instructions in this {agent_type} prompt for greater clarity and effectiveness:

ORIGINAL PROMPT:
{prompt}

REFINEMENT GOALS:
- Make instructions more specific and actionable
- Clarify any ambiguous directives
- Add helpful clarifications where needed
- Maintain the core functionality

Generate the refined prompt:""",

            "example_enhancement": f"""
Enhance the examples and demonstrations in this {agent_type} prompt:

ORIGINAL PROMPT:
{prompt}

ENHANCEMENT GOALS:
- Add relevant, helpful examples if missing
- Improve existing examples for clarity
- Ensure examples align with instructions
- Make examples more comprehensive

Generate the enhanced prompt:""",

            "structure_optimization": f"""
Optimize the structure and organization of this {agent_type} prompt:

ORIGINAL PROMPT:
{prompt}

OPTIMIZATION GOALS:
- Improve logical flow and organization
- Enhance readability and clarity
- Optimize formatting and emphasis
- Maintain all essential content

Generate the optimized prompt:""",

            "role_enhancement": f"""
Strengthen the agent identity and role definition in this {agent_type} prompt:

ORIGINAL PROMPT:
{prompt}

ENHANCEMENT GOALS:
- Clarify the agent's role and capabilities
- Strengthen professional identity
- Add relevant expertise indicators
- Maintain core functionality

Generate the enhanced prompt:""",

            # Error-targeted mutation types
            "syntax_focus": f"""
Enhance this {agent_type} prompt to improve syntax awareness and code structure:

ORIGINAL PROMPT:
{prompt}

SYNTAX FOCUS GOALS:
- Emphasize proper syntax and formatting
- Add explicit syntax validation steps
- Strengthen attention to code structure
- Include syntax error prevention guidance

Generate the syntax-focused prompt:""",

            "code_structure_enhancement": f"""
Improve this {agent_type} prompt to enhance code structure and organization:

ORIGINAL PROMPT:
{prompt}

STRUCTURE ENHANCEMENT GOALS:
- Emphasize clean, well-organized code
- Add guidance on code architecture
- Strengthen structural thinking
- Include best practices for code organization

Generate the structure-enhanced prompt:""",

            "validation_strengthening": f"""
Strengthen this {agent_type} prompt to include better validation and verification:

ORIGINAL PROMPT:
{prompt}

VALIDATION GOALS:
- Add explicit validation steps
- Emphasize verification of outputs
- Strengthen quality assurance mindset
- Include error checking guidance

Generate the validation-strengthened prompt:""",

            "logic_clarification": f"""
Enhance this {agent_type} prompt to improve logical reasoning and clarity:

ORIGINAL PROMPT:
{prompt}

LOGIC CLARIFICATION GOALS:
- Strengthen logical reasoning steps
- Add explicit problem decomposition
- Enhance algorithmic thinking
- Include logical verification steps

Generate the logic-clarified prompt:""",

            "algorithm_guidance": f"""
Improve this {agent_type} prompt to provide better algorithmic guidance:

ORIGINAL PROMPT:
{prompt}

ALGORITHM GUIDANCE GOALS:
- Add explicit algorithmic thinking steps
- Strengthen problem-solving approach
- Include algorithm selection guidance
- Enhance computational thinking

Generate the algorithm-guided prompt:""",

            "step_by_step_enhancement": f"""
Enhance this {agent_type} prompt to emphasize step-by-step problem solving:

ORIGINAL PROMPT:
{prompt}

STEP-BY-STEP GOALS:
- Add explicit step-by-step methodology
- Strengthen systematic approach
- Include problem breakdown guidance
- Enhance methodical thinking

Generate the step-by-step enhanced prompt:""",

            "completeness_focus": f"""
Improve this {agent_type} prompt to emphasize completeness and thoroughness:

ORIGINAL PROMPT:
{prompt}

COMPLETENESS GOALS:
- Emphasize thorough implementation
- Add completeness checking steps
- Strengthen attention to all requirements
- Include comprehensive coverage guidance

Generate the completeness-focused prompt:""",

            "requirement_emphasis": f"""
Enhance this {agent_type} prompt to better emphasize requirement analysis:

ORIGINAL PROMPT:
{prompt}

REQUIREMENT EMPHASIS GOALS:
- Strengthen requirement understanding
- Add explicit requirement checking
- Emphasize specification adherence
- Include requirement validation steps

Generate the requirement-emphasized prompt:""",

            "thoroughness_enhancement": f"""
Improve this {agent_type} prompt to enhance thoroughness and attention to detail:

ORIGINAL PROMPT:
{prompt}

THOROUGHNESS GOALS:
- Emphasize comprehensive coverage
- Add detail-oriented guidance
- Strengthen meticulous approach
- Include thoroughness verification

Generate the thoroughness-enhanced prompt:""",

            "edge_case_awareness": f"""
Enhance this {agent_type} prompt to improve edge case awareness:

ORIGINAL PROMPT:
{prompt}

EDGE CASE AWARENESS GOALS:
- Add explicit edge case consideration
- Strengthen boundary condition thinking
- Include corner case analysis
- Enhance robustness mindset

Generate the edge-case-aware prompt:""",

            "boundary_handling": f"""
Improve this {agent_type} prompt to enhance boundary condition handling:

ORIGINAL PROMPT:
{prompt}

BOUNDARY HANDLING GOALS:
- Emphasize boundary condition analysis
- Add explicit boundary testing
- Strengthen defensive programming
- Include boundary validation steps

Generate the boundary-handling prompt:""",

            "robustness_enhancement": f"""
Enhance this {agent_type} prompt to improve robustness and error handling:

ORIGINAL PROMPT:
{prompt}

ROBUSTNESS GOALS:
- Emphasize robust implementation
- Add error handling guidance
- Strengthen defensive programming
- Include resilience considerations

Generate the robustness-enhanced prompt:""",

            "efficiency_focus": f"""
Improve this {agent_type} prompt to emphasize efficiency and optimization:

ORIGINAL PROMPT:
{prompt}

EFFICIENCY GOALS:
- Emphasize performance optimization
- Add efficiency consideration steps
- Strengthen algorithmic efficiency
- Include complexity awareness

Generate the efficiency-focused prompt:""",

            "optimization_guidance": f"""
Enhance this {agent_type} prompt to provide better optimization guidance:

ORIGINAL PROMPT:
{prompt}

OPTIMIZATION GOALS:
- Add explicit optimization steps
- Strengthen performance thinking
- Include optimization strategies
- Enhance efficiency mindset

Generate the optimization-guided prompt:""",

            "complexity_awareness": f"""
Improve this {agent_type} prompt to enhance complexity awareness:

ORIGINAL PROMPT:
{prompt}

COMPLEXITY AWARENESS GOALS:
- Add time/space complexity consideration
- Strengthen algorithmic analysis
- Include complexity trade-offs
- Enhance performance consciousness

Generate the complexity-aware prompt:""",

            "clarity_enhancement": f"""
Enhance this {agent_type} prompt to improve clarity and understanding:

ORIGINAL PROMPT:
{prompt}

CLARITY GOALS:
- Improve overall clarity and readability
- Add explicit clarification steps
- Strengthen communication effectiveness
- Include understanding verification

Generate the clarity-enhanced prompt:""",

            "example_enrichment": f"""
Improve this {agent_type} prompt by enriching examples and demonstrations:

ORIGINAL PROMPT:
{prompt}

EXAMPLE ENRICHMENT GOALS:
- Add more comprehensive examples
- Improve example quality and relevance
- Include diverse demonstration cases
- Enhance learning through examples

Generate the example-enriched prompt:""",

            # 🆕 REGEX-SPECIFIC MUTATIONS
            "regex_star_operator_focus": f"""
Enhance this {agent_type} prompt to specifically improve regex star operator handling:

ORIGINAL PROMPT:
{prompt}

REGEX STAR OPERATOR GOALS:
- Emphasize correct handling of '*' operator (zero or more matches)
- Add explicit guidance on zero-match cases for '*'
- Strengthen understanding of '*' operator semantics
- Include examples of '*' matching zero occurrences

Generate the regex-star-focused prompt:""",

            "zero_match_handling": f"""
Improve this {agent_type} prompt to enhance zero-match case handling:

ORIGINAL PROMPT:
{prompt}

ZERO MATCH HANDLING GOALS:
- Emphasize proper handling of zero-match scenarios
- Add explicit consideration of empty/zero cases
- Strengthen edge case awareness for zero matches
- Include validation for zero-occurrence patterns

Generate the zero-match-enhanced prompt:""",

            "pattern_matching_precision": f"""
Enhance this {agent_type} prompt to improve pattern matching precision:

ORIGINAL PROMPT:
{prompt}

PATTERN MATCHING PRECISION GOALS:
- Emphasize exact pattern matching requirements
- Add precision in pattern interpretation
- Strengthen attention to pattern details
- Include comprehensive pattern analysis

Generate the pattern-precision-enhanced prompt:""",

            "complex_input_handling": f"""
Improve this {agent_type} prompt to enhance complex input handling:

ORIGINAL PROMPT:
{prompt}

COMPLEX INPUT HANDLING GOALS:
- Emphasize robust handling of complex inputs
- Add guidance for multi-parameter scenarios
- Strengthen input validation and processing
- Include comprehensive input analysis

Generate the complex-input-enhanced prompt:"""
        }

        mutation_prompt = mutation_prompts.get(mutation_type, mutation_prompts["instruction_refinement"])

        try:
            messages = [
                {"role": "system", "content": self._mutation_prompt},
                {"role": "user", "content": mutation_prompt}
            ]
            
            return self.llm_service.invoke(messages, expect_json=False)
            
        except Exception as e:
            logger.error(f"Mutation {mutation_type} failed: {e}")
            return prompt  # Return original if mutation fails
    
    def _validate_coherence(self, prompt: str, agent_type: str) -> bool:
        """Validate that a prompt maintains semantic coherence and agent functionality."""
        
        validation_prompt = f"""
Evaluate whether this {agent_type} prompt is coherent and functional:

PROMPT TO VALIDATE:
{prompt}

VALIDATION CRITERIA:
1. Role consistency: Does it maintain a clear {agent_type} agent identity?
2. Instruction clarity: Are the instructions clear and actionable?
3. Logical flow: Is the prompt well-structured and logical?
4. Completeness: Does it contain all necessary components?
5. Functionality: Would this prompt enable effective {agent_type} operation?

Respond with JSON:
{{
    "is_coherent": true/false,
    "role_consistent": true/false,
    "instructions_clear": true/false,
    "well_structured": true/false,
    "complete": true/false,
    "functional": true/false,
    "overall_score": 0.0-1.0,
    "issues": ["list", "of", "any", "issues"]
}}"""

        try:
            messages = [
                {"role": "system", "content": self._coherence_validation_prompt},
                {"role": "user", "content": validation_prompt}
            ]
            
            response = self.llm_service.invoke(messages, expect_json=True)
            
            # Consider coherent if overall score > 0.7 and no critical issues
            overall_score = response.get("overall_score", 0.0)
            is_coherent = response.get("is_coherent", False)
            
            return overall_score > 0.7 and is_coherent
            
        except Exception as e:
            logger.error(f"Coherence validation failed: {e}")
            return True  # Default to accepting if validation fails
    
    def _fallback_crossover(self, parent1: str, parent2: str) -> str:
        """Simple fallback crossover that combines prompts textually."""
        # Simple approach: take first half of parent1 + second half of parent2
        lines1 = parent1.split('\n')
        lines2 = parent2.split('\n')
        
        mid1 = len(lines1) // 2
        mid2 = len(lines2) // 2
        
        combined = lines1[:mid1] + lines2[mid2:]
        return '\n'.join(combined)
    
    def _minor_mutation(self, prompt: str, agent_type: str) -> str:
        """Apply minor textual mutations as fallback."""
        # Simple mutations: add emphasis, rephrase slightly
        mutations = [
            lambda p: p.replace("You are", "You are an expert"),
            lambda p: p.replace("should", "must"),
            lambda p: p.replace("can", "should"),
            lambda p: p + "\n\nFocus on clarity and effectiveness in your responses."
        ]
        
        mutation = random.choice(mutations)
        return mutation(prompt)
    
    def _simple_component_extraction(self, prompt: str) -> PromptComponent:
        """Simple regex-based component extraction as fallback."""
        # Basic pattern matching for common prompt structures
        role_match = re.search(r'You are (.*?)\.', prompt, re.IGNORECASE)
        role_definition = role_match.group(1) if role_match else ""
        
        # Extract instructions (often after role definition)
        instructions = prompt[:len(prompt)//2]  # First half as instructions
        
        # Look for examples
        examples_match = re.search(r'(example|for instance|such as)(.*?)(\n\n|\Z)', prompt, re.IGNORECASE | re.DOTALL)
        examples = examples_match.group(2) if examples_match else ""
        
        return PromptComponent(
            role_definition=role_definition,
            instructions=instructions,
            examples=examples,
            constraints="",
            output_format=""
        )
    
    def _build_crossover_prompt(self) -> str:
        """Build the system prompt for crossover operations."""
        return """You are an expert prompt engineer specializing in combining and optimizing prompts for AI agents.

Your task is to perform intelligent crossover between two parent prompts to create an improved offspring prompt.

Key principles:
1. Preserve the core functionality and role of the target agent
2. Combine the best elements from both parents
3. Ensure semantic coherence and logical flow
4. Maintain or improve clarity and effectiveness
5. Avoid redundancy while preserving essential information

Generate complete, functional prompts that inherit strengths from both parents."""
    
    def _build_mutation_prompt(self) -> str:
        """Build the system prompt for mutation operations."""
        return """You are an expert prompt engineer specializing in prompt refinement and optimization.

Your task is to apply controlled mutations to improve prompt effectiveness while preserving core functionality.

Key principles:
1. Maintain the agent's core role and capabilities
2. Improve clarity, specificity, and actionability
3. Preserve semantic meaning while enhancing expression
4. Make incremental improvements, not radical changes
5. Ensure all mutations maintain prompt coherence

Generate improved prompts that enhance the original while preserving its essential nature."""
    
    def _build_component_extraction_prompt(self) -> str:
        """Build the system prompt for component extraction."""
        return """You are an expert prompt analyst specializing in decomposing AI agent prompts into semantic components.

Your task is to identify and extract the key functional components of prompts.

Extract these components accurately:
- Role Definition: Agent identity and capabilities
- Instructions: Core directives and tasks
- Examples: Demonstrations and illustrations
- Constraints: Rules, limitations, requirements
- Output Format: Response structure specifications

Be precise and comprehensive in your analysis."""
    
    def _build_coherence_validation_prompt(self) -> str:
        """Build the system prompt for coherence validation."""
        return """You are an expert prompt quality assessor specializing in evaluating AI agent prompt effectiveness.

Your task is to evaluate prompt coherence, functionality, and quality across multiple dimensions.

Assessment criteria:
1. Role consistency: Clear, appropriate agent identity
2. Instruction clarity: Unambiguous, actionable directives
3. Logical structure: Well-organized, coherent flow
4. Completeness: All necessary components present
5. Functionality: Enables effective agent operation

Provide honest, accurate assessments with specific feedback on any issues identified."""

    def _build_error_targeted_prompt(self) -> str:
        """Build the system prompt for error-targeted recombination."""
        return """You are an expert prompt engineer specializing in creating error-targeted prompts for AI agents.

Your task is to intelligently combine the best elements from two parent prompts while specifically addressing the identified failure pattern.

Key principles:
1. Preserve the core functionality and role of the target agent
2. Combine elements that strengthen the agent's capability in the failure area
3. Enhance instructions to prevent the specific failure pattern
4. Add relevant constraints or examples that guide away from the error
5. Maintain overall prompt coherence while targeting the weakness

Generate coherent, improved prompts that inherit strengths from both parents while specifically addressing the failure pattern.""" 