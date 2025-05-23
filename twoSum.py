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
    # Check if list has at least 2 elements
    if len(nums) < 2:
        return []
    
    # Input validation - check if nums is a list
    if not isinstance(nums, list):
        return []
    
    # Input validation - check if target is an integer
    if not isinstance(target, int):
        return []
    
    # Input validation - check if all elements in nums are integers
    for num in nums:
        if not isinstance(num, int):
            return []
    
    # Dictionary to store complements
    num_map = {}
    
    # Use range instead of enumerate to avoid sandbox restrictions
    for i in range(len(nums)):
        num = nums[i]
        complement = target - num
        if complement in num_map:
            return [num_map[complement], i]
        num_map[num] = i
    
    # No solution found
    return [] 