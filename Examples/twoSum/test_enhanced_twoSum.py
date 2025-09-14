# Test script for enhanced_twoSum
from enhanced_twoSum import twoSum, safe_enumerate

def test_case(nums, target, expected):
    result = twoSum(nums, target)
    if sorted(result) == sorted(expected):
        print(f"✅ PASS: twoSum({nums}, {target}) = {result}")
        return True
    else:
        print(f"❌ FAIL: twoSum({nums}, {target}) = {result}, expected {expected}")
        return False

# Test cases for twoSum
print("Testing twoSum function:")
test_case([2, 7, 11, 15], 9, [0, 1])
test_case([3, 2, 4], 6, [1, 2])
test_case([3, 3], 6, [0, 1])
test_case([-1, -2, -3, -4], -5, [1, 2])
test_case([-1, 2, 3, -4], 1, [0, 1])
test_case([1000000, 2000000, 3000000], 5000000, [1, 2])

# Test the safe_enumerate function directly
print("\nTesting safe_enumerate function:")
test_list = ['a', 'b', 'c', 'd']
expected_output = [(0, 'a'), (1, 'b'), (2, 'c'), (3, 'd')]
actual_output = list(safe_enumerate(test_list))

if actual_output == expected_output:
    print(f"✅ PASS: safe_enumerate({test_list}) = {actual_output}")
else:
    print(f"❌ FAIL: safe_enumerate({test_list}) = {actual_output}, expected {expected_output}") 