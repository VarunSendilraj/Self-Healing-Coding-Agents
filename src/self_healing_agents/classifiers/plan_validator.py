"""
Plan Validator for Multi-Agent Self-Healing System

This module validates plans before execution to identify potential issues
that could lead to failures during execution.
"""

from typing import Dict, Any, List, Optional, Tuple
import re
import logging

logger = logging.getLogger(__name__)

class PlanValidator:
    """
    Validates plans for quality, completeness, and feasibility before execution.
    """
    
    def __init__(self):
        self.required_plan_fields = ["steps", "requirements", "approach"]
        self.vague_indicators = [
            "implement", "create", "handle", "process", "manage", "deal with",
            "work with", "take care of", "figure out", "sort out"
        ]
        self.technical_keywords = [
            "function", "class", "variable", "loop", "condition", "import", 
            "algorithm", "data structure", "input", "output", "return"
        ]
        
    def validate_plan(
        self, 
        plan: Dict[str, Any], 
        task_description: str,
        complexity_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Validate a plan for quality and feasibility.
        
        Args:
            plan: The plan to validate
            task_description: Original task description
            complexity_threshold: Minimum complexity score for acceptance
            
        Returns:
            Dict containing validation results
        """
        logger.info("📋 PLAN VALIDATION: Starting plan quality assessment...")
        
        validation_result = {
            "is_valid": True,
            "quality_score": 0.0,
            "issues": [],
            "warnings": [],
            "suggestions": [],
            "complexity_score": 0.0,
            "completeness_score": 0.0,
            "specificity_score": 0.0,
            "feasibility_score": 0.0
        }
        
        # Validate plan structure
        structure_score = self._validate_structure(plan, validation_result)
        
        # Assess plan completeness  
        completeness_score = self._assess_completeness(plan, task_description, validation_result)
        validation_result["completeness_score"] = completeness_score
        
        # Assess plan specificity (not too vague)
        specificity_score = self._assess_specificity(plan, validation_result)
        validation_result["specificity_score"] = specificity_score
        
        # Assess plan feasibility
        feasibility_score = self._assess_feasibility(plan, task_description, validation_result)
        validation_result["feasibility_score"] = feasibility_score
        
        # Calculate overall quality score
        validation_result["quality_score"] = (
            structure_score * 0.25 + 
            completeness_score * 0.3 + 
            specificity_score * 0.25 + 
            feasibility_score * 0.2
        )
        
        # Determine if plan passes validation
        if validation_result["quality_score"] < complexity_threshold:
            validation_result["is_valid"] = False
            validation_result["issues"].append(
                f"Plan quality score {validation_result['quality_score']:.2f} below threshold {complexity_threshold}"
            )
            
        logger.info(f"📋 PLAN VALIDATION: Quality Score: {validation_result['quality_score']:.2f}")
        logger.info(f"📋 PLAN VALIDATION: Valid: {validation_result['is_valid']}")
        
        return validation_result
        
    def _validate_structure(self, plan: Dict[str, Any], result: Dict[str, Any]) -> float:
        """Validate the basic structure of the plan."""
        score = 1.0
        
        if not isinstance(plan, dict):
            result["issues"].append("Plan is not in dictionary format")
            return 0.0
            
        # Check for required fields
        missing_fields = []
        for field in self.required_plan_fields:
            if field not in plan and f"plan_{field}" not in plan:
                missing_fields.append(field)
                
        if missing_fields:
            result["warnings"].append(f"Missing recommended fields: {missing_fields}")
            score -= 0.3
            
        # Check for steps
        steps = plan.get("steps", plan.get("plan_steps", []))
        if not steps:
            result["issues"].append("No execution steps found in plan")
            score -= 0.5
        elif len(steps) < 2:
            result["warnings"].append("Plan has very few steps - may be too high-level")
            score -= 0.2
            
        return max(score, 0.0)
        
    def _assess_completeness(self, plan: Dict[str, Any], task_description: str, result: Dict[str, Any]) -> float:
        """Assess how complete the plan is relative to the task."""
        score = 1.0
        
        steps = plan.get("steps", plan.get("plan_steps", []))
        if not steps:
            return 0.0
            
        # Check if plan addresses key task components
        task_lower = task_description.lower()
        plan_text = str(plan).lower()
        
        # Look for key elements that should be addressed
        key_elements = []
        if "function" in task_lower or "def " in task_lower:
            key_elements.append("function definition")
        if "input" in task_lower or "parameter" in task_lower:
            key_elements.append("input handling")
        if "output" in task_lower or "return" in task_lower:
            key_elements.append("output specification")
        if "test" in task_lower or "example" in task_lower:
            key_elements.append("test cases")
        if "error" in task_lower or "exception" in task_lower:
            key_elements.append("error handling")
            
        missing_elements = []
        for element in key_elements:
            element_keywords = element.split()
            if not any(keyword in plan_text for keyword in element_keywords):
                missing_elements.append(element)
                
        if missing_elements:
            result["warnings"].append(f"Plan may not address: {missing_elements}")
            score -= 0.2 * len(missing_elements)
            
        # Check for step coverage
        if len(steps) > 0:
            # Simple heuristic: longer, more detailed plans tend to be more complete
            avg_step_length = sum(len(str(step)) for step in steps) / len(steps)
            if avg_step_length < 30:  # Very short steps
                result["warnings"].append("Plan steps are quite brief - may lack detail")
                score -= 0.2
                
        return max(score, 0.0)
        
    def _assess_specificity(self, plan: Dict[str, Any], result: Dict[str, Any]) -> float:
        """Assess how specific (not vague) the plan is."""
        score = 1.0
        
        steps = plan.get("steps", plan.get("plan_steps", []))
        if not steps:
            return 0.0
            
        vague_steps = []
        technical_content = 0
        
        for i, step in enumerate(steps):
            step_text = str(step).lower()
            
            # Check for vague language
            vague_count = sum(1 for indicator in self.vague_indicators if indicator in step_text)
            if vague_count > 0 and len(step_text.split()) < 8:
                vague_steps.append(f"Step {i+1}: {step}")
                
            # Check for technical specificity
            tech_count = sum(1 for keyword in self.technical_keywords if keyword in step_text)
            technical_content += tech_count
            
        if vague_steps:
            result["warnings"].append(f"Vague steps found: {len(vague_steps)} steps need more detail")
            score -= 0.3 * min(len(vague_steps) / len(steps), 0.8)
            
        # Bonus for technical specificity
        if technical_content > len(steps):  # More than one technical term per step on average
            score += 0.1
            
        return max(score, 0.0)
        
    def _assess_feasibility(self, plan: Dict[str, Any], task_description: str, result: Dict[str, Any]) -> float:
        """Assess if the plan is feasible to implement."""
        score = 1.0
        
        steps = plan.get("steps", plan.get("plan_steps", []))
        if not steps:
            return 0.0
            
        # Check for circular dependencies (basic)
        step_outputs = []
        step_inputs = []
        
        for step in steps:
            step_text = str(step).lower()
            
            # Look for potential circular references
            if "use result from step" in step_text:
                result["warnings"].append("Potential step dependencies detected - verify order")
                score -= 0.1
                
            # Check for impossible operations
            impossible_patterns = [
                "modify input", "change the given", "alter the provided",
                "update the original", "fix the input"
            ]
            for pattern in impossible_patterns:
                if pattern in step_text:
                    result["issues"].append(f"Infeasible operation detected: {pattern}")
                    score -= 0.3
                    
        # Check for missing imports/dependencies  
        plan_text = str(plan).lower()
        task_lower = task_description.lower()
        
        # Look for operations that need imports
        import_needs = {
            "math": ["sqrt", "sin", "cos", "log", "exp", "pi"],
            "re": ["regex", "pattern", "match", "search"],
            "collections": ["counter", "defaultdict", "deque"],
            "itertools": ["permutations", "combinations", "product"]
        }
        
        missing_imports = []
        for module, keywords in import_needs.items():
            if any(keyword in task_lower or keyword in plan_text for keyword in keywords):
                if f"import {module}" not in plan_text:
                    missing_imports.append(module)
                    
        if missing_imports:
            result["suggestions"].append(f"Consider specifying imports: {missing_imports}")
            score -= 0.1
            
        return max(score, 0.0)
        
    def suggest_improvements(self, plan: Dict[str, Any], validation_result: Dict[str, Any]) -> List[str]:
        """Generate specific suggestions for improving the plan."""
        suggestions = []
        
        if validation_result["completeness_score"] < 0.7:
            suggestions.append("Add more detailed steps covering all aspects of the task")
            
        if validation_result["specificity_score"] < 0.7:
            suggestions.append("Replace vague terms with specific technical actions")
            suggestions.append("Include specific function names, variable names, or algorithms")
            
        if validation_result["feasibility_score"] < 0.7:
            suggestions.append("Verify that all operations are implementable")
            suggestions.append("Add necessary import statements or dependencies")
            
        # Add suggestions based on plan content
        steps = plan.get("steps", plan.get("plan_steps", []))
        if steps and len(steps) < 3:
            suggestions.append("Break down complex operations into smaller, manageable steps")
            
        return suggestions 