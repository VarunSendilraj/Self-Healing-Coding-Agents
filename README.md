# Self-Healing Agents System

A multi-agent system for automatically fixing and improving Python code through self-healing capabilities.

## Overview

This system provides:
1. Multi-agent code analysis and healing
2. Automatic error detection and correction
3. Code execution in a controlled environment  
4. Test case generation and validation
5. Comprehensive evaluation harness

## Project Structure

```
├── src/                           # Main source code
│   └── self_healing_agents/       # Core package
│       ├── agents.py              # Main agent implementations
│       ├── llm_service.py         # LLM integration
│       ├── orchestrator.py        # Agent orchestration
│       ├── classifiers/           # Failure classification
│       └── evaluation/            # Evaluation tools
├── tests/                         # Unit tests
├── Examples/                      # Example tasks and demos
│   ├── twoSum/                   # TwoSum problem examples
│   ├── *.md                      # Algorithm examples
│   └── *.json                    # Task definitions
├── demonstrate_self_healing.py    # Main demo script
├── run_healing_tests.py          # Test runner
└── requirements.txt              # Dependencies
```

## Main Entry Points

- `run_healing_tests.py` - Main test runner for the healing system
- `demonstrate_self_healing.py` - Demonstration of self-healing functionality

## Core Components

- `src/code_analyzer.py`: Provides an API for running code and analyzing results
- `src/code_runner.py`: Executes code in a controlled environment with resource limits
- `src/self_healing_agents/`: Main package with agent implementations

## Usage

### Running the System

```bash
# Install dependencies
pip install -r requirements.txt

# Run quick demo
python run_healing_tests.py quick

# Run comprehensive tests  
python run_healing_tests.py comprehensive

# Run demonstration
python demonstrate_self_healing.py
```

### Basic Code Analysis

```python
from src.code_analyzer import CodeAnalyzer

# Initialize the analyzer
analyzer = CodeAnalyzer()

# Execute a simple code snippet
code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

print(fibonacci(10))
"""

result = analyzer.run_code(code)
analysis = analyzer.analyze_result(result)

print(f"Execution successful: {analysis['success']}")
print(f"Execution time: {analysis['execution_time_ms']}ms")
print(f"Output: {result['stdout']}")
```

### Running Test Cases

```python
# Define a function to test
code = """
def twoSum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []
"""

# Define test cases
test_cases = [
    {
        "function_name": "twoSum",
        "inputs": {"nums": [2, 7, 11, 15], "target": 9},
        "expected": [0, 1]
    },
    {
        "function_name": "twoSum",
        "inputs": {"nums": [3, 2, 4], "target": 6},
        "expected": [1, 2]
    }
]

# Run the tests
results = analyzer.test_with_cases(code, test_cases)

print(f"Tests passed: {results['tests_passed']}/{results['tests_total']}")
print(f"Pass percentage: {results['pass_percentage']}%")
```

## Security Considerations

This system executes arbitrary Python code, which can be dangerous. It implements several safeguards:

1. Resource limits (memory and execution time)
2. Execution in a subprocess
3. Capture of exceptions and error output

However, it is not completely secure against all types of attacks. Use with caution, especially with untrusted code.

## Requirements

- Python 3.6+
- Standard library modules only

## Future Improvements

- Add support for executing code in Docker containers for improved isolation
- Implement more advanced code analysis features
- Add support for other programming languages 