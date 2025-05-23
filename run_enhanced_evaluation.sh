#!/bin/bash
# Script to run the Enhanced Evaluation Harness with direct fix capability

echo "Running Enhanced Evaluation Harness..."
echo "======================================="

# Define conda environment path - assuming standard location
CONDA_ENV_PATH="$HOME/SelfHealingAgentsV2/.conda"
PYTHON_CMD="$CONDA_ENV_PATH/bin/python"

# Check if the conda Python executable exists
if [ ! -f "$PYTHON_CMD" ]; then
    echo "Conda Python not found at $PYTHON_CMD"
    echo "Attempting to use Python from .conda in current directory..."
    
    # Try looking for Python in the local .conda directory
    if [ -d ".conda" ]; then
        PYTHON_CMD=".conda/bin/python"
        if [ ! -f "$PYTHON_CMD" ]; then
            echo "ERROR: Python executable not found at $PYTHON_CMD"
            echo "Please ensure your conda environment is properly set up."
            exit 1
        fi
    else
        echo "ERROR: No .conda directory found."
        echo "Please ensure your conda environment is properly set up."
        exit 1
    fi
fi

echo "Using Python: $PYTHON_CMD"

# Set the PYTHONPATH to include the project root
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Run the enhanced evaluation harness
$PYTHON_CMD -m src.self_healing_agents.evaluation.enhanced_harness

echo "======================================="
echo "Evaluation complete."
echo "Check enhanced_evaluation_harness.log for details."
