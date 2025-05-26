#!/usr/bin/env python3
"""
Fix the critique bug where hasattr is used instead of checking dictionary keys.
"""

import re

def fix_critique_bug():
    """Fix the bug in evolutionary_enhanced_harness.py"""
    
    file_path = "src/self_healing_agents/evaluation/evolutionary_enhanced_harness.py"
    
    # Read the file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Fix the hasattr bug - replace with dict key check
    old_pattern = r"hasattr\(critique_for_classification, 'test_results'\)"
    new_pattern = "isinstance(critique_for_classification, dict) and 'test_results' in critique_for_classification"
    
    content = re.sub(old_pattern, new_pattern, content)
    
    # Fix the attribute access bug - replace with dict key access
    old_pattern = r"critique_for_classification\.test_results"
    new_pattern = "critique_for_classification['test_results']"
    
    content = re.sub(old_pattern, new_pattern, content)
    
    # Write the file back
    with open(file_path, 'w') as f:
        f.write(content)
    
    print("✅ Fixed critique bug in evolutionary_enhanced_harness.py")
    print("   - Changed hasattr() to dict key check")
    print("   - Changed attribute access to dict key access")

if __name__ == "__main__":
    fix_critique_bug() 