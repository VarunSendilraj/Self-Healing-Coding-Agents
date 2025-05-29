#!/usr/bin/env python3
"""
Unit tests for robust LLM response parsing.
Tests various scenarios including JSON in markdown, mixed content, and fallbacks.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import json
import unittest
from src.self_healing_agents.utils.llm_response_parser import LLMResponseParser


class TestLLMResponseParser(unittest.TestCase):
    """Test cases for robust LLM response parsing."""
    
    def test_direct_json_parsing(self):
        """Test direct JSON parsing without any wrapping."""
        response = '{"steps": ["step1", "step2"], "approach": "test"}'
        result = LLMResponseParser.extract_json(response)
        
        self.assertIsNotNone(result)
        self.assertEqual(result["steps"], ["step1", "step2"])
        self.assertEqual(result["approach"], "test")
    
    def test_markdown_json_extraction(self):
        """Test extraction from markdown code blocks."""
        response = '''Here's the improved plan:

```json
{
  "steps": ["analyze", "implement", "test"],
  "requirements": ["python"],
  "approach": "incremental"
}
```

This should work better.'''
        
        result = LLMResponseParser.extract_json(response)
        
        self.assertIsNotNone(result)
        self.assertEqual(result["steps"], ["analyze", "implement", "test"])
        self.assertEqual(result["requirements"], ["python"])
        self.assertEqual(result["approach"], "incremental")
    
    def test_markdown_without_language_tag(self):
        """Test extraction from markdown blocks without 'json' tag."""
        response = '''```
{
  "steps": ["step1", "step2"],
  "approach": "test approach"
}
```'''
        
        result = LLMResponseParser.extract_json(response)
        
        self.assertIsNotNone(result)
        self.assertEqual(result["steps"], ["step1", "step2"])
    
    def test_mixed_content_extraction(self):
        """Test extraction when JSON is mixed with explanatory text."""
        response = '''Looking at the problem, I need to create an improved plan:

{
  "steps": [
    "Define the function",
    "Implement the logic", 
    "Handle edge cases"
  ],
  "requirements": ["no external deps"],
  "approach": "dynamic programming"
}

This plan should address the issues mentioned.'''
        
        result = LLMResponseParser.extract_json(response)
        
        self.assertIsNotNone(result)
        self.assertEqual(len(result["steps"]), 3)
        self.assertEqual(result["approach"], "dynamic programming")
    
    def test_text_plan_extraction(self):
        """Test extracting plan structure from plain text."""
        response = '''Here's the improved plan:

1. Understand the regex matching requirements
2. Implement base cases for empty strings
3. Handle '.' wildcard character matching
4. Implement '*' zero-or-more matching logic
5. Use memoization for efficiency
6. Test with provided examples

The approach will be dynamic programming with recursion.
Requirements include Python functools for memoization.'''
        
        result = LLMResponseParser.extract_plan_from_text(response)
        
        self.assertIsNotNone(result)
        self.assertIn("steps", result)
        self.assertEqual(len(result["steps"]), 6)
        self.assertIn("regex matching", result["steps"][0])
        self.assertIn("memoization", result["steps"][4])
    
    def test_numbered_list_extraction(self):
        """Test extracting numbered lists from text."""
        response = '''1. First step here
2. Second step with details
3. Third and final step'''
        
        result = LLMResponseParser.extract_plan_from_text(response)
        
        self.assertIsNotNone(result)
        self.assertEqual(len(result["steps"]), 3)
        self.assertIn("First step", result["steps"][0])
        self.assertIn("Second step", result["steps"][1])
        self.assertIn("Third and final", result["steps"][2])
    
    def test_bullet_list_extraction(self):
        """Test extracting bullet lists from text."""
        response = '''- Analyze the problem requirements
- Design solution approach
- Implement core functionality
- Add error handling
- Test thoroughly'''
        
        result = LLMResponseParser.extract_plan_from_text(response)
        
        self.assertIsNotNone(result)
        self.assertEqual(len(result["steps"]), 5)
        self.assertIn("Analyze", result["steps"][0])
        self.assertIn("Test thoroughly", result["steps"][4])
    
    def test_fallback_plan_creation(self):
        """Test creating fallback plans when parsing fails."""
        result = LLMResponseParser.create_fallback_plan(
            context="JSON parsing failed for planner healing"
        )
        
        self.assertIsNotNone(result)
        self.assertIn("steps", result)
        self.assertIn("approach", result)
        self.assertIn("requirements", result)
        self.assertEqual(result["fallback_reason"], "JSON parsing failed")
    
    def test_invalid_json_recovery(self):
        """Test recovery from malformed JSON."""
        response = '''Here's an improved plan:
{
  "steps": ["step1", "step2"
  "approach": "test"
}'''  # Missing comma, malformed JSON
        
        # Should return None for invalid JSON, then fallback to text extraction
        json_result = LLMResponseParser.extract_json(response)
        self.assertIsNone(json_result)
        
        # But text extraction should still work
        text_result = LLMResponseParser.extract_plan_from_text(response)
        self.assertIsNotNone(text_result)
    
    def test_empty_response_handling(self):
        """Test handling of empty or None responses."""
        self.assertIsNone(LLMResponseParser.extract_json(""))
        self.assertIsNone(LLMResponseParser.extract_json(None))
        
        # Text extraction should provide defaults
        result = LLMResponseParser.extract_plan_from_text("")
        self.assertIsNotNone(result)
        self.assertIn("steps", result)
    
    def test_complex_nested_json(self):
        """Test parsing complex nested JSON structures."""
        response = '''```json
{
  "steps": [
    {
      "step": 1,
      "description": "Initialize DP table",
      "details": "Create 2D array for memoization"
    },
    {
      "step": 2, 
      "description": "Handle base cases",
      "details": "Empty string and pattern cases"
    }
  ],
  "approach": "Dynamic programming with memoization",
  "requirements": ["Python", "No external libs"],
  "improvements_made": [
    "Added specific algorithmic approach",
    "Included memoization for efficiency"
  ]
}
```'''
        
        result = LLMResponseParser.extract_json(response)
        
        self.assertIsNotNone(result)
        self.assertEqual(len(result["steps"]), 2)
        self.assertIsInstance(result["steps"][0], dict)
        self.assertEqual(result["steps"][0]["step"], 1)
        self.assertIn("memoization", result["approach"])


def run_parser_tests():
    """Run all parser tests."""
    print("🧪 Testing Robust LLM Response Parser")
    print("=" * 50)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestLLMResponseParser)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 50)
    if result.wasSuccessful():
        print("✅ All parser tests passed!")
    else:
        print(f"❌ {len(result.failures)} test(s) failed")
        print(f"❌ {len(result.errors)} error(s) occurred")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_parser_tests()
    sys.exit(0 if success else 1) 