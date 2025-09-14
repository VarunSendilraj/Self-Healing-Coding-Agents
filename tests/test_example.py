#!/usr/bin/env python
"""
Example script demonstrating the usage of the Code Analysis System.
"""

import json
from code_analyzer import CodeAnalyzer

def demonstrate_basic_execution():
    """Demonstrate basic code execution."""
    print("\n=== Basic Code Execution ===\n")
    
    analyzer = CodeAnalyzer()
    
    # A simple recursive function (inefficient by design)
    code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

result = fibonacci(20)
print(f"Fibonacci(20) = {result}")
"""
    
    print("Executing code:")
    print(code)
    
    result = analyzer.run_code(code)
    analysis = analyzer.analyze_result(result)
    
    print("\nExecution Results:")
    print(f"Success: {result['success']}")
    print(f"Output: {result['stdout'].strip()}")
    print(f"Execution time: {analysis['execution_time_ms']:.2f}ms")
    print(f"Memory usage: {analysis['memory_usage_mb']:.2f}MB")
    
    if analysis['recommendations']:
        print("\nRecommendations:")
        for rec in analysis['recommendations']:
            print(f"- {rec}")

def demonstrate_error_handling():
    """Demonstrate error handling capabilities."""
    print("\n=== Error Handling ===\n")
    
    analyzer = CodeAnalyzer()
    
    # Code with syntax error
    code_with_error = """
def buggy_function():
    x = 10
    if x > 5
        return "Greater than 5"  # Missing colon after if condition
    return "Not greater than 5"

print(buggy_function())
"""
    
    print("Executing code with syntax error:")
    print(code_with_error)
    
    result = analyzer.run_code(code_with_error)
    analysis = analyzer.analyze_result(result)
    
    print("\nExecution Results:")
    print(f"Success: {result['success']}")
    print(f"Error type: {result['error_type']}")
    print(f"Error message: {result['error_message']}")
    
    if analysis['recommendations']:
        print("\nRecommendations:")
        for rec in analysis['recommendations']:
            print(f"- {rec}")

def demonstrate_test_cases():
    """Demonstrate running test cases against a function."""
    print("\n=== Test Case Execution ===\n")
    
    analyzer = CodeAnalyzer()
    
    # TwoSum implementation
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
    
    print("Testing function:")
    print(code)
    
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
        },
        {
            "function_name": "twoSum",
            "inputs": {"nums": [3, 3], "target": 6},
            "expected": [0, 1]
        }
    ]
    
    print("\nTest cases:")
    for i, test in enumerate(test_cases):
        print(f"Test {i+1}: twoSum({test['inputs']['nums']}, {test['inputs']['target']}) → Expected: {test['expected']}")
    
    results = analyzer.test_with_cases(code, test_cases)
    
    print("\nTest Results:")
    print(f"Tests passed: {results['tests_passed']}/{results['tests_total']} ({results['pass_percentage']}%)")
    
    print("\nDetailed Results:")
    for result in results['detailed_results']:
        status = "✅ PASS" if result['matches_expected'] else "❌ FAIL"
        print(f"Test {result['test_case']}: {status} (Actual: {result['actual']})")

def demonstrate_performance_analysis():
    """Demonstrate performance analysis of different implementations."""
    print("\n=== Performance Analysis ===\n")
    
    analyzer = CodeAnalyzer()
    
    # Inefficient implementation (O(n²))
    inefficient_code = """
def find_duplicate(nums):
    n = len(nums)
    for i in range(n):
        for j in range(i+1, n):
            if nums[i] == nums[j]:
                return nums[i]
    return -1

# Test with a larger array
import random
nums = [random.randint(1, 100) for _ in range(1000)]
nums.append(nums[500])  # Ensure a duplicate exists
result = find_duplicate(nums)
print(f"Found duplicate: {result}")
"""
    
    # Efficient implementation (O(n))
    efficient_code = """
def find_duplicate(nums):
    seen = set()
    for num in nums:
        if num in seen:
            return num
        seen.add(num)
    return -1

# Test with a larger array
import random
nums = [random.randint(1, 100) for _ in range(1000)]
nums.append(nums[500])  # Ensure a duplicate exists
result = find_duplicate(nums)
print(f"Found duplicate: {result}")
"""
    
    print("Comparing performance of two implementations:")
    
    print("\nRunning inefficient implementation (O(n²))...")
    inefficient_result = analyzer.run_code(inefficient_code)
    inefficient_analysis = analyzer.analyze_result(inefficient_result)
    
    print("\nRunning efficient implementation (O(n))...")
    efficient_result = analyzer.run_code(efficient_code)
    efficient_analysis = analyzer.analyze_result(efficient_result)
    
    print("\nPerformance Comparison:")
    print(f"Inefficient implementation: {inefficient_analysis['execution_time_ms']:.2f}ms")
    print(f"Efficient implementation: {efficient_analysis['execution_time_ms']:.2f}ms")
    
    if inefficient_analysis['execution_time_ms'] > efficient_analysis['execution_time_ms']:
        speedup = inefficient_analysis['execution_time_ms'] / efficient_analysis['execution_time_ms']
        print(f"The efficient implementation is {speedup:.1f}x faster!")

if __name__ == "__main__":
    print("=== Code Analysis System Demo ===")
    
    demonstrate_basic_execution()
    demonstrate_error_handling()
    demonstrate_test_cases()
    demonstrate_performance_analysis()
    
    print("\n=== Demo Complete ===") 