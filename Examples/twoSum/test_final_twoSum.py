# Test script for final_twoSum.py
from final_twoSum import twoSum

def test_case(nums, target, expected):
    result = twoSum(nums, target)
    if sorted(result) == sorted(expected):
        print(f"✅ PASS: twoSum({nums}, {target}) = {result}")
        return True
    else:
        print(f"❌ FAIL: twoSum({nums}, {target}) = {result}, expected {expected}")
        return False

# Test cases
print("Testing twoSum function:")
test_case([2, 7, 11, 15], 9, [0, 1])
test_case([3, 2, 4], 6, [1, 2])
test_case([3, 3], 6, [0, 1])
test_case([-1, -2, -3, -4], -5, [1, 2])
test_case([-1, 2, 3, -4], 1, [0, 1])
test_case([1000000, 2000000, 3000000], 5000000, [1, 2]) 