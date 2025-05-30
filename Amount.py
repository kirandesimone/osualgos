"""
Analysis:
    find all distinct combinations
    0/1 knapsack??
    we can sort to jump over duplicate elements and we'll avoid the
    constraint overall by doing this and the orignal element will have
    branches that either only includes one of it or the duplicates.

base case:
    If we hit our target sum or we go over the len(input_list)

choices:
    we can either take our current element or skip it 
    
constraints:
    an element can only be used the same amount of times it appears
    in the input array. 

Running Time:
    sorting takes like O(n log n) time so not that big a deal in this
    case

    T(n) = 2T(n-1) + n = O(2^n)

Space:
    we have to build the combo array as we recurse and worst case O(len(nums_list))
"""
def amountHelper(nums_list, target_sum, curr_sum, combos, combo, i):
    if curr_sum == target_sum:
        combos.append(combo.copy())
        return
    
    if i == len(nums_list):
        return
    
    new_sum = curr_sum + nums_list[i]
    # take
    if new_sum <= target_sum:
        combo.append(nums_list[i])
        amountHelper(nums_list, target_sum, new_sum, combos, combo, i + 1)
        combo.pop()
    
    # skip duplicates
    m = i
    for n in range(m, len(nums_list) - 1):
        if nums_list[n] == nums_list[n + 1]:
            i += 1
        else:
            break 

    amountHelper(nums_list, target_sum, curr_sum, combos, combo, i + 1)

    return
        
def amount(nums_list, target_sum):
    combos = []
    combo = []
    sorted_list = sorted(nums_list)
    amountHelper(sorted_list, target_sum, 0, combos, combo, 0)

    return combos