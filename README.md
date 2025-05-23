# Code Execution and Analysis System

A system for safely executing and analyzing Python code snippets.

## Overview

This system provides a way to:
1. Execute Python code in a controlled environment
2. Capture execution results, including stdout, stderr, and performance metrics
3. Run test cases against functions and analyze the results
4. Provide recommendations for code optimization and error correction

## Components

- `code_runner.py`: Executes code in a controlled environment with resource limits
- `code_analyzer.py`: Provides an API for running code and analyzing results

## Usage

### Basic Execution

```python
from code_analyzer import CodeAnalyzer

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