#!/bin/bash

echo "🧹 Cleaning up Self-Healing Agents project..."

# Remove test files
echo "🗑️  Removing test files..."
rm -f test_enhanced_multi_agent_harness.py
rm -f demonstrate_self_healing.py
rm -f test_self_healing_demo.py
rm -f test_llm_test_generation.py
rm -f test_enhanced_harness_integration.py
rm -f test_code_analyzer.py
rm -f test_harness_integration.py
rm -f test_subprocess.py
rm -f test_integration.py
rm -f test_example.py
rm -f test_sandbox_safe_twoSum.py
rm -f test_final_twoSum.py
rm -f test_enhanced_twoSum.py
rm -f test_manual.py
rm -f test_single_case.py
rm -f test_direct_fix.py

# Remove example code files
echo "🗑️  Removing example code files..."
rm -f twoSum.py
rm -f sandbox_safe_twoSum.py
rm -f final_twoSum.py
rm -f enhanced_twoSum.py
rm -f patch_sandbox.py

# Remove generated test files
echo "🗑️  Removing generated test files..."
rm -f direct_integration_test.txt
rm -f parent_process.txt
rm -f subprocess_test.txt

# Remove duplicate code runners
echo "🗑️  Removing duplicate code runners..."
rm -f code_analyzer.py
rm -f code_runner.py

# Remove logs and extra docs
echo "🗑️  Removing logs and extra documentation..."
rm -f README_INTEGRATION_FIXES.md
rm -f README_INTEGRATION.md
rm -f enhanced_evaluation_harness.log
rm -f CombinationSUm.log
rm -f sortedArray.log

# Remove task definitions (unless needed)
echo "🗑️  Removing task definition files..."
rm -f tricky_task.json
rm -f regex_task.json

# Remove shell scripts
echo "🗑️  Removing shell scripts..."
rm -f run_enhanced_evaluation.sh

# Remove unused evaluation files
echo "🗑️  Removing unused evaluation files..."
rm -f src/self_healing_agents/evaluation/quick_healing_demo.py
rm -f src/self_healing_agents/evaluation/comprehensive_healing_test.py
rm -f src/self_healing_agents/evaluation/enhanced_evaluation_harness.log
rm -f src/self_healing_agents/evaluation/enhanced_harness_integration.py
rm -f src/self_healing_agents/evaluation/code_analyzer_integration.py
rm -f src/self_healing_agents/evaluation/harness.py

# Remove unused agent components
echo "🗑️  Removing unused agent components..."
rm -f src/self_healing_agents/orchestration.py
rm -f src/self_healing_agents/prompt_modifier.py
rm -f src/self_healing_agents/agent_factory.py
rm -f src/self_healing_agents/orchestrator.py
rm -f src/self_healing_agents/tasks.py

# Remove cache files
echo "🗑️  Removing cache files..."
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Remove empty directories
echo "🗑️  Removing empty directories..."
find . -type d -empty -delete 2>/dev/null || true

echo "✅ Cleanup complete!"
echo ""
echo "📁 Core files preserved:"
echo "   - src/self_healing_agents/agents.py"
echo "   - src/self_healing_agents/llm_service.py"
echo "   - src/self_healing_agents/prompts.py"
echo "   - src/self_healing_agents/schemas.py"
echo "   - src/self_healing_agents/classifiers/"
echo "   - src/self_healing_agents/evaluation/test_llm_classifier.py"
echo "   - src/self_healing_agents/evaluation/enhanced_multi_agent_harness.py"
echo "   - src/self_healing_agents/evaluation/enhanced_harness.py"
echo ""
echo "🚀 Your project is now clean and ready to run with:"
echo "   python src/self_healing_agents/evaluation/test_llm_classifier.py" 