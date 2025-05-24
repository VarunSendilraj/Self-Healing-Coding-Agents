# Enhanced Multi-Agent Self-Healing Test Suite

This comprehensive test suite verifies that the enhanced multi-agent harness can correctly identify and heal both planning and execution failures using LLM-based intelligent classification.

## 🎯 Overview

The test suite includes:

### **Bad Prompts for Testing**
- **`BAD_PLANNER_PROMPT`**: Creates vague, incomplete plans with logical gaps
- **`CATASTROPHIC_EXECUTOR_PROMPT`**: Generates broken code with undefined variables
- **`SYNTAX_ERROR_EXECUTOR_PROMPT`**: Creates code with syntax errors
- **`WRONG_ALGORITHM_EXECUTOR_PROMPT`**: Uses completely wrong algorithmic approaches
- **`INCOMPLETE_EXECUTOR_PROMPT`**: Only partially solves problems
- **`VARIABLE_MESS_EXECUTOR_PROMPT`**: Has severe variable management issues

### **Test Categories**
1. **Planner Healing Tests**: Complex problems with bad planner + good executor
2. **Executor Healing Tests**: Simple problems with good planner + bad executor  
3. **Mixed Complexity Tests**: Medium problems with both agents having issues

## 🚀 Quick Start

### Prerequisites
Make sure you have your LLM service configured:
```bash
export LLM_PROVIDER=deepseek  # or your provider
export LLM_MODEL=deepseek-coder  # or your model
```

### Run Tests

1. **Quick Demo** (3 targeted tests - ~5-10 minutes)
   ```bash
   python run_healing_tests.py quick
   ```

2. **Comprehensive Suite** (8 challenging tests - ~20-30 minutes)
   ```bash
   python run_healing_tests.py comprehensive
   ```

3. **Single Test** (run specific test)
   ```bash
   python run_healing_tests.py single planner_test_1_lcs_algorithm
   ```

4. **List Available Tests**
   ```bash
   python run_healing_tests.py list
   ```

## 📋 Test Details

### **Quick Demo Tests**

| Test | Agent Config | Problem | Expected Outcome |
|------|-------------|---------|------------------|
| **Planner Healing Demo** | BAD Planner + GOOD Executor | Edit Distance Algorithm (DP) | Planner healing triggered |
| **Executor Healing Demo** | GOOD Planner + BAD Executor | Palindrome Check (Simple) | Executor healing triggered |
| **Mixed Scenario Demo** | BAD Planner + BAD Executor | Binary Search (Medium) | Either/both healing triggered |

### **Comprehensive Test Suite**

#### **Planner Healing Tests** (Complex problems, bad planner, good executor)
1. **`planner_test_1_lcs_algorithm`**: Longest Common Subsequence with DP
2. **`planner_test_2_graph_algorithm`**: Dijkstra's Algorithm implementation  
3. **`planner_test_3_complex_data_structure`**: Trie with autocomplete and scoring

#### **Executor Healing Tests** (Simple problems, good planner, bad executor)
1. **`executor_test_1_basic_math`**: Factorial calculation (iterative)
2. **`executor_test_2_string_processing`**: Reverse words while preserving spaces
3. **`executor_test_3_list_operations`**: Find pair with target sum

#### **Mixed Complexity Tests** (Medium problems, both agents bad)
1. **`mixed_test_1_binary_tree`**: Binary Search Tree with full operations
2. **`mixed_test_2_dynamic_programming`**: 0/1 Knapsack problem

## 🔍 How It Works

### **Enhanced Multi-Agent Harness Workflow**
1. **Initial Planning**: Planner creates plan (potentially bad)
2. **Plan Validation**: Check plan quality and structure  
3. **Execution**: Executor generates code based on plan
4. **Initial Evaluation**: Critic evaluates and tests the code
5. **Direct Fix Attempt**: Try simple fixes first
6. **LLM-Based Failure Classification**: Analyze root cause
7. **Targeted Healing**: Apply healing to specific problematic agent
8. **Re-evaluation**: Test improved solution

### **LLM Failure Classification**
The `LLMFailureClassifier` analyzes:
- Task description and requirements
- Plan quality and completeness  
- Code implementation and errors
- Test case failures and patterns
- Execution errors and traces

It determines:
- **Primary failure type**: `PLANNING_FAILURE`, `EXECUTION_FAILURE`, `MIXED_FAILURE`
- **Recommended healing target**: `PLANNER`, `EXECUTOR`, `MULTIPLE`
- **Confidence level**: 0.0 to 1.0
- **Specific issues**: Detailed breakdown of problems
- **Healing recommendations**: Targeted improvement suggestions

## 📊 Expected Results

### **Success Indicators**
- **Planner Healing Tests**: Should trigger planner healing (bad plans → good plans)
- **Executor Healing Tests**: Should trigger executor healing (broken code → working code)
- **Classification Accuracy**: >70% correct identification of failure source
- **Final Scores**: Improvement after healing iterations

### **Sample Output**
```
🧪 TEST 1/8: planner_test_1_lcs_algorithm
🎯 Type: PLANNER_HEALING
📊 Complexity: HIGH
🎭 Expected Target: PLANNER

🔧 AGENT CONFIGURATION:
   🤖 Planner: BAD (vague, incomplete plans)
   🔧 Executor: GOOD (should generate working code)

📊 TEST 1 RESULTS:
   Final Status: SUCCESS_PLANNER_HEALING
   Final Score: 0.92
   Execution Time: 45.2s
   Planner Healings: 1
   Executor Healings: 0
   🤖 LLM Classifications: 1
      1. PLANNING_FAILURE → PLANNER (confidence: 0.87)
   ✅ HEALING TARGET CORRECTLY IDENTIFIED
```

## 🛠️ Customization

### **Add New Tests**
Edit `src/self_healing_agents/evaluation/comprehensive_healing_test.py`:

```python
{
    "id": "your_test_id",
    "test_type": "PLANNER_HEALING",  # or EXECUTOR_HEALING, MIXED_COMPLEXITY
    "description": "Your problem description...",
    "planner_prompt": BAD_PLANNER_PROMPT,  # or PLANNER_SYSTEM_PROMPT
    "executor_prompt": DEFAULT_EXECUTOR_SYSTEM_PROMPT,  # or bad prompt
    "expected_healing_target": "PLANNER",  # or EXECUTOR, MULTIPLE
    "complexity": "HIGH"  # or LOW, MEDIUM
}
```

### **Create New Bad Prompts**
Add to `src/self_healing_agents/prompts.py`:

```python
YOUR_BAD_PROMPT = """Your intentionally problematic prompt that will cause specific types of failures..."""
```

## 📈 Interpreting Results

### **Classification Accuracy**
- **>80%**: Excellent - System reliably identifies failure sources
- **60-80%**: Good - System mostly correct with some confusion  
- **<60%**: Needs improvement - Classification logic needs refinement

### **Healing Effectiveness**
- **High success rate on targeted tests**: System can fix specific problems
- **Score improvement after healing**: Healing actually helps
- **Appropriate healing selection**: Right agent gets healed for each problem type

### **Common Issues**
- **No healing triggered**: Problems may be too easy/hard, or prompts not bad enough
- **Wrong healing target**: Classification logic may need adjustment
- **Low final scores**: Healing may not be effective enough

## 🔧 Troubleshooting

### **Tests Not Running**
```bash
# Check LLM service configuration
echo $LLM_PROVIDER
echo $LLM_MODEL

# Test LLM connectivity
python -c "from self_healing_agents.llm_service import LLMService; LLMService('deepseek', 'deepseek-coder')"
```

### **No Healing Triggered**
- Verify bad prompts are truly problematic
- Check if problems are appropriately challenging
- Review LLM classification thresholds

### **Unexpected Results**  
- Check LLM service responses and reasoning
- Review classification history in output
- Analyze specific test failures and error patterns

## 📝 Test Reports

Test results include:
- **Final status and scores**
- **Healing breakdown** (planner vs executor healings)
- **LLM classification decisions** with reasoning
- **Execution time and iteration counts**
- **Detailed workflow phase analysis**

Use these to understand system behavior and identify areas for improvement.

---

**Happy Testing!** 🎉 Use this suite to verify that your enhanced multi-agent harness can intelligently identify and heal both planning and execution failures. 