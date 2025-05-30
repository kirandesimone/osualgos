"""
Info:
    Generate the power set for a given set of integers

Analysis:
    We need to identify the powerset of the given set

    what is a powerset?
        - It's a set that contains all possible subsets of a given set
            including the empty set and the set itself
        
    Since we need all subsets of the given set we can use backtracking

Run Time:
    run time will probably be 
Goal:
    Find the powerset of the given set

Base Case:
    When do we stop? When we have a powerset


Choices/transitions:
    take the current element or skip it

Running Time:
    O(n * 2^n)
    the height of the tree is 2^n
    at each node we have to build an array of n elements
"""
def powersetHelper(input_set, subsets, subset, n, i):
    if i >= n:
        subsets.append(subset.copy())
        return
    
    # backtrack/take
    subset.append(input_set[i])
    powersetHelper(input_set, subsets, subset, n, i + 1)
    subset.pop()
    
    # skip 
    powersetHelper(input_set, subsets, subset, n, i + 1)

    return

def powerset(input_set):
    subsets = []
    subset = []
    n = len(input_set)
    powersetHelper(input_set, subsets, subset, n, 0)

    return subsets
