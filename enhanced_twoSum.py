from typing import List, Tuple, Iterator, TypeVar, Any

T = TypeVar('T')

def safe_enumerate(iterable: List[T]) -> Iterator[Tuple[int, T]]:
    """
    Implementation of enumerate that works in restricted environments.
    
    Args:
        iterable: The list or iterable to enumerate
        
    Yields:
        Tuples of (index, item) for each item in the iterable
    """
    i = 0
    for item in iterable:
        yield (i, item)
        i += 1

def twoSum(nums: List[int], target: int) -> List[int]:
    """
    Finds indices of two numbers in nums that add up to target.
    Uses a custom enumerate implementation for compatibility with restricted environments.
    
    Args:
        nums: List of integers
        target: Target sum
        
    Returns:
        List of two indices whose elements sum to target
    """
    # Dictionary to store complements
    num_map = {}
    
    # Use safe_enumerate instead of built-in enumerate
    for i, num in safe_enumerate(nums):
        complement = target - num
        if complement in num_map:
            return [num_map[complement], i]
        num_map[num] = i
    
    return [] 