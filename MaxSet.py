"""
Info:
    Given a list of integers, return a subsequence of non-consecutive
    numbers (as a list) that produces the maximum possible sum.

    -   The subsequence cannot have consecutive numbers from the list
    -   If all numbers in list are negative then return empty list
    -   if sum of nums is 0 then return empty list

Analysis:
    Rules -
        1. No consecutive numbers included in the max sum
        2. max sum >= 0
    
    We don't know the max sum target
    
    We cannot just add up all the numbers because that violates rule 1
    
    We could just skip every other number? But then how do we determine which 
    number to start at?
    
    We could still skip every other number but we would have to check every number
    as our start position.

Example:
    Input: [7,2,5,8,6]
    Output: [7,5,6] (This will have sum of 18)

State:
    nums, i

Transitions:
    take - i + 2
    skip - i + 1

Base case:
    i >= len(nums) return 0

Bounds:
    i: [0:len(nums)+1]
    T(i) = T(i-c) + O(1) = O(n^2)


"""
def max_independent_set(nums):
    cache = [0] * (len(nums) + 2)
    max_set_iter_helper(nums, cache)

    # Backtrack
    ans = []
    i = 0
    while i < len(nums):
        if cache[i] == cache[i+1]:
            # We skiped the current item
            i += 1
        else:
            # This was the optimal value to take
            ans.append(nums[i])
            i += 2
    
    return ans

def max_set_iter_helper(nums, cache):
    for n in range(len(nums) - 1, -1, -1):
        skip = cache[n + 1]
        take = nums[n] + cache[n + 2] 

        cache[n] = max(skip, take)


"""
------------Brute Force Method----------------------------------------------------
def max_independent_set_brute(nums, i):
    # Base case
    if i >= len(nums):
        return 0 
    
    skip = max_independent_set_brute(nums, i + 1)
    take = nums[i] + max_independent_set_brute(nums, i + 2)
    
    # Since the max sum needs to be greater than 0
    return max(skip, take)
-----------------------------------------------------------------------------------  
-----------Memo Method-------------------------------------------------------------
def max_set_memo_helper(nums, i, cache):
    # Base Case
    if i >= len(nums):
        return 0
    
    # memo
    if cache[i] != -1:
        return cache[i]
    
    skip = max_set_memo_helper(nums, i + 1, cache)
    take = nums[i] + max_set_memo_helper(nums, i + 2, cache)

    cache[i] = max(skip, take)

    return cache[i]
"""