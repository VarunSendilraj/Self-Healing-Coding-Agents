# Code Analyzer Integration with Self-Healing Agents

This document explains how to integrate the Code Execution and Analysis System with the Self-Healing Agents framework.

## Overview

The integration provides:

1. A replacement for the sandbox environment using our controlled code execution system
2. Test case generation and validation based on task descriptions
3. Detailed performance metrics and analysis recommendations
4. Seamless integration with the existing Critic component

## Integration Components

- `code_analyzer_integration.py`: Provides the `CodeAnalyzerCritic` class that bridges between our system and the agent framework
- `enhanced_harness_integration.py`: Script to patch and run the enhanced harness with our code analyzer

## How It Works

The integration works by:

1. Creating a wrapper around the `CodeAnalyzer` that matches the interface expected by the agent system
2. Monkey patching the `Critic` class in the enhanced harness to use our `CodeAnalyzerCritic`
3. Extracting test cases from task descriptions when none are provided
4. Translating between the result formats of our analyzer and the agent framework

## Installation and Usage

### Prerequisites

- Python 3.6+
- Self-Healing Agents framework installed
- Code Execution and Analysis System installed

### Integration Steps

1. Copy the `code_runner.py` and `code_analyzer.py` files to your project root
2. Create the integration files in the specified locations:
   - `src/self_healing_agents/evaluation/code_analyzer_integration.py`
   - `src/self_healing_agents/evaluation/enhanced_harness_integration.py`

3. Run the integration script:

```bash
cd /path/to/project/root
python src/self_healing_agents/evaluation/enhanced_harness_integration.py
```

## Test Case Generation

The integration can automatically generate test cases from task descriptions by:

1. Extracting example inputs and outputs using regex patterns
2. Parsing the structured examples into test case dictionaries
3. Determining the appropriate function name and parameters

For example, given a twoSum task description with examples, it will create test cases with:
- Function name: `twoSum`
- Inputs: `{"nums": [2, 7, 11, 15], "target": 9}`
- Expected output: `[0, 1]`

## Benefits

- **Safety**: Code execution is performed in a controlled environment with resource limits
- **Performance Analysis**: Get detailed metrics on execution time and memory usage
- **Recommendations**: Automatic suggestions for code optimization and error fixing
- **Detailed Reports**: Comprehensive test results for each code execution

## Customization

You can customize the integration by:

1. Modifying the `generate_test_cases_from_description` method to support more task formats
2. Adjusting the scoring algorithm in the `run` method
3. Adding additional analysis metrics to the critic report

## Troubleshooting

If you encounter issues:

1. Check that the paths are correctly set up in the integration files
2. Verify that the `code_runner.py` and `code_analyzer.py` files are in the expected locations
3. Ensure that the monkey patching is occurring before the enhanced harness is executed
4. Review the logs for detailed error messages 