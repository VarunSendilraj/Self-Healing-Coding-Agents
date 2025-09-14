from typing import List

def twoSum(nums: List[int], target: int) -> List[int]:
    """
    Finds indices of two numbers in nums that add up to target.
    
    Args:
        nums: List of integers
        target: Target sum
        
    Returns:
        List of two indices whose elements sum to target
    """
    # Dictionary to store complements
    num_map = {}
    
    # Use manual indexing instead of enumerate to avoid sandbox restrictions
    for i in range(len(nums)):
        num = nums[i]
        complement = target - num
        if complement in num_map:
            return [num_map[complement], i]
        num_map[num] = i
    
    # No solution found
    return [] 