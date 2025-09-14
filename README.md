# Self-Healing Agent

A multi-agent framework that explores **prompt optimization for coding tasks** by building a **self-healing loop**: a process where agents automatically detect failures, classify root causes, and update their own prompts to improve overall system performance.

This project treats coding as a **deterministic testing ground** for studying how multi-agent systems can autonomously adapt and evolve their prompts, with the long-term goal of building **robust, self-correcting agent ecosystems**.

---

## Purpose of v1: Experiment Goals

The first version of the Self-Healing Agents System (v1) serves as a **prototype** to test the viability of automated prompt optimization in a multi-agent environment. Coding tasks provide a uniquely suited domain for this research because they:

* Allow for **deterministic validation** (unit tests, execution traces, correctness checks)
* Generate **clear feedback signals** for success or failure
* Enable **precise diagnosis** of error sources

The overarching goal of v1 is to investigate:

1. **Prompt Self-Healing:** Can agent prompts be automatically repaired based on failure evidence?
2. **Multi-Agent Optimization:** Does distributing roles across Planner, Executor, and Critic agents improve performance compared to monolithic prompts?
3. **Closed-Loop Adaptation:** Can a system iteratively refine itself with minimal human intervention, using coding tasks as a controlled sandbox?
4. **Scalability & Generalization:** How well does this framework extend from simple algorithms to more complex computational problems?


<img width="735" height="625" alt="image" src="https://github.com/user-attachments/assets/b2d0498f-3c6c-408e-9408-692c7b3313c3" />

---

## Project Structure

```
├── src/                           # Main source code
│   └── self_healing_agents/       # Core package
│       ├── agents.py              # Agent implementations
│       ├── llm_service.py         # LLM integration
│       ├── orchestrator.py        # Agent orchestration
│       ├── classifiers/           # Failure classification
│       └── evaluation/            # Evaluation tools
├── tests/                         # Unit tests
├── Examples/                      # Example tasks and demos
│   ├── twoSum/                    # TwoSum problem examples
│   ├── *.md                       # Algorithm examples
│   └── *.json                     # Task definitions
├── demonstrate_self_healing.py    # Main demo script
├── run_healing_tests.py           # Test runner
└── requirements.txt               # Dependencies
```

---

## Main Entry Points

* `run_healing_tests.py` – Main test runner for the healing system
* `demonstrate_self_healing.py` – Demonstration of self-healing functionality

---

## Core Components

* **`src/code_analyzer.py`** – Runs code and analyzes results
* **`src/code_runner.py`** – Executes code in a controlled, resource-limited environment
* **`src/self_healing_agents/`** – Implements Planner, Executor, Critic, and self-healing logic

---

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

analyzer = CodeAnalyzer()

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

test_cases = [
    {"function_name": "twoSum", "inputs": {"nums": [2, 7, 11, 15], "target": 9}, "expected": [0, 1]},
    {"function_name": "twoSum", "inputs": {"nums": [3, 2, 4], "target": 6}, "expected": [1, 2]}
]

results = analyzer.test_with_cases(code, test_cases)

print(f"Tests passed: {results['tests_passed']}/{results['tests_total']}")
print(f"Pass percentage: {results['pass_percentage']}%")
```

---

## Experimental Results (v1)

We evaluated the v1 system on a broad range of algorithmic tasks, from beginner-friendly exercises to advanced problems. Each run was scored against generated test cases, and the **self-healing loop** was triggered when failures occurred.

### Results Overview

* **Tasks attempted:** 25+
* **Initial success rate:** \~60%
* **Improvement via self-healing:** modest increases, often recovering simple/medium tasks but struggling with advanced system-level challenges
* **Strengths:** Deterministic validation, clear improvements on modular problems
* **Weaknesses:** Limited gains on complex, multi-step algorithmic tasks

### Sample Task Outcomes

- ✅ Easy to Medium: arithmetic, string reversal, Fibonacci (memoized), palindrome detection, binary search, merge sort
- ❌ Hard to Very Hard: graph BFS, stack implementation, cache systems, parsers, compiler lexer

### Key Insights

* **Multi-Agent Advantage:** Planner/Executor/Critic separation improved interpretability and targeted repair.
* **Prompt Healing Works:** The loop salvaged several failing runs with minimal compute cost.
* **Scaling Challenge:** Hard problems require more than localized fixes — they demand higher-level reasoning or architecture changes.

---

## Requirements

* Python 3.6+
* Standard library modules only

---

## Future Improvements

* Docker-based runtime isolation for security
* More advanced error classification for complex task recovery
* Multi-language support beyond Python
* Genetic prompt evolution for structured prompt optimization
* Larger-scale benchmarking to evaluate generalization and robustness
