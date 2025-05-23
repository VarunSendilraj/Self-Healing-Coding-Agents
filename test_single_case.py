#!/usr/bin/env python3
"""
Simple test script for the direct error-fixing enhancement.
This script tests a single case with the new enhancement where it first tries a direct
error fix before engaging the Self-Healing Module.
"""

import os
import sys
from dotenv import load_dotenv

# Now we can use direct imports since we installed the package
from self_healing_agents.orchestration import Orchestrator
from self_healing_agents.llm_service import LLMService

# Load environment variables from .env file
load_dotenv()

def main():
    """Main entry point for the test script."""
    # Get API key from environment variables
    api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    provider = "deepseek" if os.getenv("DEEPSEEK_API_KEY") else "openai"
    
    if not api_key:
        print("Error: No API key found. Please set DEEPSEEK_API_KEY or OPENAI_API_KEY in your .env file.")
        return
    
    # Determine the model name based on the provider
    if provider == "deepseek":
        model_name = "deepseek-chat" # Use appropriate model name for Deepseek
    else:
        model_name = "gpt-4" # Or any other available OpenAI model
    
    # Create LLM service with required model_name parameter
    print(f"Using LLM provider: {provider}, model: {model_name}")
    llm_service = LLMService(api_key=api_key, provider=provider, model_name=model_name)
    
    # Test case with a common error
    test_description = "Write a function called 'find_max_subarray' that takes a list of integers and returns the maximum sum of any contiguous subarray."
    
    print(f"\n{'='*80}")
    print(f"Testing with description: {test_description}")
    print(f"{'='*80}")
    
    # Create orchestrator with the test description
    orchestrator = Orchestrator(user_request=test_description)
    
    # Inject the LLM service into the orchestrator's agent factory
    orchestrator.agents_factory.set_llm_service(llm_service)
    
    # Run the enhanced flow that includes direct fix attempt
    print("\nRunning enhanced flow with direct error-fixing attempt...\n")
    result = orchestrator.run_enhanced_flow()
    
    # Display results
    print(f"\n{'='*80}")
    print("RESULTS:")
    print(f"{'='*80}")
    print(f"Status: {result.get('status', 'UNKNOWN')}")
    print(f"Best Source: {result.get('best_source', 'N/A')}")
    print(f"Best Score: {result.get('best_score', 0.0)}")
    
    print("\nFinal Code:")
    print(f"{'-'*40}")
    print(result.get('final_code', 'No code available'))
    print(f"{'-'*40}")
    
if __name__ == "__main__":
    main()
