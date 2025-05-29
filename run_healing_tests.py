#!/usr/bin/env python3
"""
Healing Test Runner

This script provides an easy way to run different healing tests for the
enhanced multi-agent self-healing system.

Usage:
    python run_healing_tests.py quick          # Quick demo (3 targeted tests)
    python run_healing_tests.py comprehensive  # Full test suite (8 challenging tests)  
    python run_healing_tests.py single <test_id>  # Run specific test
    python run_healing_tests.py list           # List available tests
"""

import sys
import os

# Add src to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

def run_quick_demo():
    """Run the quick demonstration with 3 targeted scenarios."""
    from self_healing_agents.evaluation.quick_healing_demo import run_quick_demo
    return run_quick_demo()

def run_comprehensive_tests():
    """Run the full comprehensive test suite."""
    from self_healing_agents.evaluation.comprehensive_healing_test import run_comprehensive_healing_tests
    return run_comprehensive_healing_tests()

def run_single_test(test_id: str):
    """Run a specific test by ID."""
    from self_healing_agents.evaluation.comprehensive_healing_test import run_single_test
    return run_single_test(test_id)

def list_available_tests():
    """List all available tests."""
    from self_healing_agents.evaluation.comprehensive_healing_test import get_challenging_test_tasks
    
    print("📋 AVAILABLE TESTS:")
    print("=" * 60)
    
    test_tasks = get_challenging_test_tasks()
    
    print(f"💡 QUICK DEMO TESTS (run with: python run_healing_tests.py quick):")
    print(f"   • Planner Healing Demo: Complex DP algorithm with bad planner")
    print(f"   • Executor Healing Demo: Simple palindrome with bad executor")
    print(f"   • Mixed Scenario Demo: Binary search with both bad agents")
    
    print(f"\n🧪 COMPREHENSIVE TESTS (run with: python run_healing_tests.py comprehensive):")
    
    for i, task in enumerate(test_tasks, 1):
        test_type = task['test_type']
        complexity = task['complexity']
        expected_target = task['expected_healing_target']
        
        # Color coding
        if test_type == 'PLANNER_HEALING':
            type_color = '🧠'
        elif test_type == 'EXECUTOR_HEALING':
            type_color = '⚙️'
        else:
            type_color = '🔄'
            
        print(f"   {i:2d}. {type_color} {task['id']}")
        print(f"       Type: {test_type}")
        print(f"       Complexity: {complexity}")
        print(f"       Expected Target: {expected_target}")
        print(f"       Description: {task['description'][:80]}...")
        print()
    
    print(f"🎯 SINGLE TEST (run with: python run_healing_tests.py single <test_id>):")
    print(f"   Use any test_id from the comprehensive list above")
    print(f"   Example: python run_healing_tests.py single planner_test_1_lcs_algorithm")

def show_help():
    """Show usage help."""
    print(__doc__)

def main():
    """Main entry point for the test runner."""
    
    if len(sys.argv) < 2:
        show_help()
        return
    
    command = sys.argv[1].lower()
    
    if command == "quick":
        print("🚀 RUNNING QUICK DEMO...")
        run_quick_demo()
        
    elif command == "comprehensive":
        print("🧪 RUNNING COMPREHENSIVE TEST SUITE...")
        run_comprehensive_tests()
        
    elif command == "single":
        if len(sys.argv) < 3:
            print("❌ Please specify a test ID")
            print("Usage: python run_healing_tests.py single <test_id>")
            print("Run 'python run_healing_tests.py list' to see available tests")
            return
        test_id = sys.argv[2]
        print(f"🧪 RUNNING SINGLE TEST: {test_id}")
        run_single_test(test_id)
        
    elif command == "list":
        list_available_tests()
        
    elif command in ["help", "-h", "--help"]:
        show_help()
        
    else:
        print(f"❌ Unknown command: {command}")
        show_help()

if __name__ == "__main__":
    main() 