"""
Robust LLM Response Parser for handling various output formats.
Handles JSON wrapped in markdown, mixed content, and provides fallbacks.
"""

import json
import re
import logging
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)


class LLMResponseParser:
    """
    Robust parser for LLM responses that can handle various output formats.
    """
    
    @staticmethod
    def extract_json(response: str, expect_dict: bool = True) -> Optional[Union[Dict, Any]]:
        """
        Extract JSON from LLM response with multiple fallback strategies.
        
        Args:
            response: Raw LLM response text
            expect_dict: Whether to expect a dictionary response
            
        Returns:
            Parsed JSON object or None if extraction fails
        """
        if not response or not isinstance(response, str):
            logger.warning(f"Invalid response type: {type(response)}")
            return None
        
        response = response.strip()
        
        # Strategy 1: Direct JSON parsing
        try:
            parsed = json.loads(response)
            logger.debug("✅ Direct JSON parsing successful")
            return parsed
        except json.JSONDecodeError:
            logger.debug("❌ Direct JSON parsing failed, trying markdown extraction")
        
        # Strategy 2: Extract JSON from markdown code blocks
        json_content = LLMResponseParser._extract_from_markdown(response)
        if json_content:
            try:
                parsed = json.loads(json_content)
                logger.debug("✅ Markdown JSON extraction successful")
                return parsed
            except json.JSONDecodeError:
                logger.debug("❌ Markdown JSON parsing failed")
        
        # Strategy 3: Extract JSON using regex patterns
        json_content = LLMResponseParser._extract_with_regex(response)
        if json_content:
            try:
                parsed = json.loads(json_content)
                logger.debug("✅ Regex JSON extraction successful")
                return parsed
            except json.JSONDecodeError:
                logger.debug("❌ Regex JSON parsing failed")
        
        # Strategy 4: Clean and try parsing
        cleaned_response = LLMResponseParser._clean_response(response)
        if cleaned_response:
            try:
                parsed = json.loads(cleaned_response)
                logger.debug("✅ Cleaned JSON parsing successful")
                return parsed
            except json.JSONDecodeError:
                logger.debug("❌ Cleaned JSON parsing failed")
        
        logger.warning(f"Failed to extract JSON from response: {response[:200]}...")
        return None
    
    @staticmethod
    def _extract_from_markdown(response: str) -> Optional[str]:
        """Extract JSON from markdown code blocks."""
        # Pattern for ```json ... ``` blocks
        json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
        match = re.search(json_pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # Pattern for ``` ... ``` blocks (without language specifier)
        general_pattern = r'```\s*(\{.*?\})\s*```'
        match = re.search(general_pattern, response, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        return None
    
    @staticmethod
    def _extract_with_regex(response: str) -> Optional[str]:
        """Extract JSON using regex patterns."""
        # Look for JSON object patterns
        patterns = [
            r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # Simple nested object
            r'\{.*\}',  # Any content between braces
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            for match in matches:
                # Try to validate it's proper JSON
                try:
                    json.loads(match)
                    return match
                except json.JSONDecodeError:
                    continue
        
        return None
    
    @staticmethod
    def _clean_response(response: str) -> Optional[str]:
        """Clean the response and try to extract JSON."""
        # Remove common prefixes/suffixes
        cleaned = response.strip()
        
        # Remove common explanatory text
        patterns_to_remove = [
            r'^.*?(?=\{)',  # Remove everything before first {
            r'\}.*$',       # Remove everything after last }
        ]
        
        for pattern in patterns_to_remove:
            cleaned = re.sub(pattern, '', cleaned, flags=re.DOTALL)
        
        # Add back the closing brace if needed
        if cleaned.startswith('{') and not cleaned.endswith('}'):
            # Count braces to add missing closing braces
            open_braces = cleaned.count('{')
            close_braces = cleaned.count('}')
            if open_braces > close_braces:
                cleaned += '}' * (open_braces - close_braces)
        
        return cleaned if cleaned.startswith('{') and cleaned.endswith('}') else None
    
    @staticmethod
    def extract_plan_from_text(response: str) -> Dict[str, Any]:
        """
        Extract plan information from text response when JSON parsing fails.
        Creates a structured plan from unstructured text.
        
        Args:
            response: Raw text response
            
        Returns:
            Dictionary with extracted plan information
        """
        if not response:
            return {"steps": ["Implement solution"], "approach": "Basic implementation"}
        
        # Try to extract steps from numbered or bulleted lists
        steps = []
        
        # Pattern for numbered steps (1. 2. etc.)
        numbered_pattern = r'^\s*\d+\.\s*(.+)$'
        numbered_matches = re.findall(numbered_pattern, response, re.MULTILINE)
        if numbered_matches:
            steps.extend([step.strip() for step in numbered_matches])
        
        # Pattern for bulleted steps (- * etc.)
        bullet_pattern = r'^\s*[-*•]\s*(.+)$'
        bullet_matches = re.findall(bullet_pattern, response, re.MULTILINE)
        if bullet_matches:
            steps.extend([step.strip() for step in bullet_matches])
        
        # If no structured steps found, split by sentences/lines
        if not steps:
            lines = [line.strip() for line in response.split('\n') if line.strip()]
            # Filter out very short lines and metadata
            steps = [line for line in lines if len(line) > 10 and not line.startswith('```')]
        
        # Ensure we have at least some steps
        if not steps:
            steps = ["Analyze the problem", "Implement solution", "Test and validate"]
        
        # Extract other components if possible
        approach = "Systematic implementation based on requirements"
        requirements = ["Python standard library"]
        
        # Look for approach/strategy mentions
        approach_keywords = ["approach", "strategy", "method", "algorithm"]
        for line in response.split('\n'):
            if any(keyword in line.lower() for keyword in approach_keywords):
                approach = line.strip()
                break
        
        # Look for requirements/dependencies
        req_keywords = ["require", "import", "depend", "need"]
        for line in response.split('\n'):
            if any(keyword in line.lower() for keyword in req_keywords):
                requirements.append(line.strip())
        
        return {
            "steps": steps[:10],  # Limit to reasonable number
            "approach": approach,
            "requirements": list(set(requirements)),  # Remove duplicates
            "extraction_method": "text_parsing",
            "original_response_length": len(response)
        }
    
    @staticmethod
    def is_valid_json_response(response: Any) -> bool:
        """Check if response is a valid JSON object."""
        return isinstance(response, (dict, list))
    
    @staticmethod
    def create_fallback_plan(original_plan: Optional[Dict] = None, 
                           context: Optional[str] = None) -> Dict[str, Any]:
        """Create a basic fallback plan when all parsing fails."""
        base_steps = [
            "Understand the problem requirements",
            "Design the solution approach", 
            "Implement core functionality",
            "Handle edge cases",
            "Test and validate solution"
        ]
        
        if original_plan and isinstance(original_plan, dict):
            # Preserve any extractable information
            original_steps = original_plan.get("steps", original_plan.get("plan_steps", []))
            if original_steps:
                base_steps = original_steps
        
        return {
            "steps": base_steps,
            "approach": "Incremental development with testing",
            "requirements": ["Python standard library"],
            "fallback_reason": "JSON parsing failed",
            "context": context or "No context provided"
        } 