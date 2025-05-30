from typing import List
import heapq
from DisjointSet import DisjointSet
#--------------------------------------------------------------------------------------------------#
#                                           RECURSION
#-------------------------------------------------------------------------------------------------#

def remove_dupes(s: str) -> str:
    # One char == unique
    if len(s) <= 1:
        return s
    
    char = remove_dupes(s[1:])
    if s[0] != char[0]:
        char = s[0] + char

    print(char)
    return char


def find_smallest_number(arr: List[int]) -> int:
    # an array of size one is the smallest
    if len(arr) <= 1:
        return arr[0]
    
    small_num = find_smallest_number(arr[1:])
    if arr[0] < small_num:
        small_num = arr[0]

    print(small_num)
    return small_num


def calc_exponent_rec(base: int, exponent: int) -> int:
    if exponent == 0:
        return 1

    return base * calc_exponent_rec(base, exponent - 1)


def towers_of_hanoi(n, src: List[int], aux: List[int], dest: List[int]) -> None:
    """
    Work from the bottom up
    Do we know how to move n disks? No, but I know how to move n-1 disks.
    What if n=2? Can you move 2 disks? No, but I can move 1 disk! From source straight
    to the destination.
    """
    # base: If we pass the bottom most disk, then come out of that level of recurrsion.
    if n < 1:
        return
    
    # Go all the way down past the bottom most disk, and swap the dest peg with the aux peg,
    # since we have n > 1 disks and we need to move the n-1st disks to get to the nth disk.
    towers_of_hanoi(n-1, src, dest, aux)
    # If n > 1 then the n-1st disks will be moved to the aux peg, else we have 1 disk and it goes
    # straight to the dest peg (initial pegs stayed the way they were when we first executed the code)
    dest.append(src.pop())
    # If n > 1 then we swap the src peg with aux peg and dest stays the same since all of our n-1st
    # disks are on the aux peg, we need those to move to the dest peg. If we only have 1 disk then
    # this step doesn't matter
    towers_of_hanoi(n-1, aux, src, dest)


def evaluateString(input):
    if len(input) <= 0:
        return 0

    i = 0
    while i < len(input):
        if input[i] == '+':
            break

        i += 1

    num = int(input[0:i])
    addendum = evaluateString(input[i+1:])

    return num + addendum
        

def ternary_search(input_list, target):
    """
    Finds the target element by splitting the list into thirds
    """
    left = 0
    right = len(input_list) - 1

    while right >= left:
        subset_size = (right - left) // 3
        m1 = left + subset_size 
        m2 = right - subset_size

        # Add 1 for 1-indexing
        if input_list[m1] == target:
            return m1 + 1 

        if input_list[m2] == target:
            return m2 + 1

        if target < input[m1]:
            right = m1 - 1
        elif target > input[m2]:
            left = m2 + 1
        else:
            left = m1 + 1
            right = m2 - 1
        
    return -1

#########################################################
#               0/1 knapsack                            #
#########################################################
def maximum_profit(profit: List[int], weight: List[int], capacity: int, i: int) -> int:
    """
    Brute Force recursion method O(2^n) -
    State: capacity, i
    transitions: 
        skip = [capacity, i + 1]
        take = [capacity - cur_weight, i + 1]
    base case: if we checked all of our weights
    """
    # Base Case
    if i == len(weight):
        return 0
    
    cur_weight = weight[i]
    cur_profit = profit[i] 

    # Skip the current item
    skip_profit = maximum_profit(profit, weight, capacity, (i + 1))

    # Take the current item
    take_profit = 0
    if (capacity - cur_weight) >= 0:
        take_profit = cur_profit + maximum_profit(profit, weight, (capacity - cur_weight), (i + 1))
    
    return max(skip_profit, take_profit)


def maximum_profit_memo(profit: List[int], weight: List[int], capacity: int, i: int, cache: List[List[int]]) -> int:
    """
    TODO: Top-down with memoization method O(capacity * len(weight)) -
    """
    pass

def maximum_profit_iter(profit: List[int], weight: List[int], capacity: int, i: int, cache: List[List[int]]) -> int:
    """
    TODO: Bottom-up (tabulation) method O(capacity * len(weight)) -
    """
    pass

########################################################
#               unbounded knapsack                     #
########################################################



#######################################################
#              Longest Common Subsequence             #
#######################################################
"""
Example:
    TAGTTCCGTCAAA
    GTGTTCCCGTCAAA

    Answer - TGTTCCGTCAAA = 12

Analysis:
    DNA sequences can be different lengths
    They don't need to be contiguous
    A subsequence of a sequence is just the given sequence with 0 or more elements left out
    
State:
    DNA1, DNA2
    i1, i2

Transitions:
    if dna1_n == dna2_m, then we check the subsequence length at i1 + 1 and i2 + 1
    if dna1_n != dna2_m, then 
        1. We check the subsequence length at i1 and i2 + 1
        or
        2. We check the subsequence length at i1 + 1 and i2

Base Case:
    if dna1_n == 0 or dna2_m == 0, return 0
"""
def dna_match_brute(DNA1: str, DNA2: str, i1: int, i2: int) -> int:
    # base case
    if i1 == len(DNA1) or i2 == len(DNA2):
        return 0 
    
    # transitions 
    if DNA1[i1] == DNA2[i2]:
        return 1 + dna_match_brute(DNA1, DNA2, (i1 + 1), (i2 + 1))
    else:
        lcs1 = dna_match_brute(DNA1, DNA2, i1, (i2 + 1))
        lcs2 = dna_match_brute(DNA1, DNA2, (i1 + 1), i2)
        return max(lcs1, lcs2)

def dna_match_td():
    pass

def dn_match_bu():
    pass

#########################################################################
#                   knapsack variation                                  #
#########################################################################
def max_independent_set_brute(nums, i):
    # Base case
    if i >= len(nums):
        return 0 
    
    skip = max_independent_set_brute(nums, i + 1)
    take = nums[i] + max_independent_set_brute(nums, i + 2)
    
    # Since the max sum needs to be greater than 0
    return max(skip, take)

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

def max_set_iter_helper(nums, cache):
    for n in range(len(nums) - 1, -1, -1):
        skip = cache[n + 1]
        take = nums[n] + cache[n + 2] 

        cache[n] = max(skip, take)

    return None
#####################################################################
#                   Backtracking                                    #
#####################################################################
def construct_queen_candidates(n: int, row: int, dc: List[int], psc: List[int], nsc: List[int]) -> List[int]:
    candidates = []
    for col in range(n):
        if col in dc or (row + col) in psc or (row - col) in nsc:
            continue
            
        candidates.append(col)
    
    return candidates

def backtrack_queens(boardConfigs: List[List[str]], board: List[str], n: int, 
    row: int, dc: List[int], psc: List[int], nsc: List[int]) -> None:

    if row == n:
        boardCopy = ["".join(row) for row in board]
        boardConfigs.append(boardCopy)
        return None
    
    candidates = construct_queen_candidates(n, row, dc, psc, nsc)
    for col in candidates:
        board[row][col] = "Q"
        dc.append(col)
        psc.append(row + col)
        nsc.append(row - col)

        backtrack_queens(boardConfigs, board, n, row + 1, dc, psc, nsc)

        board[row][col] = "."
        dc.pop()
        psc.pop()
        nsc.pop()
    
    return None

def solve_n_queens(n: int) -> List[List[str]]:
    boardConfigs = []
    board = [["."]*n for _i in range(n)]
    downCandidates = []
    posSlopeCandidates = []
    negSlopCandidates = []

    backtrack_queens(boardConfigs, board, n, 0, downCandidates, posSlopeCandidates, negSlopCandidates)

    return boardConfigs

def backtrack_combo(
    combos: List[int], 
    combo:List[int], 
    candidates: List[int], 
    target: int, 
    sum: int, 
    i: int
) -> None:
    if sum == target:
        combos.append(combo.copy())
        return None

    for c in range(i, len(candidates)):
        newSum = sum + candidates[c]
        if newSum <= target:
            combo.append(candidates[c])
            # to combat against dupelicate combos or permutations of combos, 
            # we can only take the current object. If that object doesn't 
            # work then we can only take objects ahead of our current object.
            backtrack_combo(combos, combo, candidates, target, newSum, c)
            combo.pop()
    
    return None

def combination_sum(candidates: List[int], target: int) -> List[List[int]]:
    combos = []
    combo = []
    backtrack_combo(combos, combo, candidates, target, 0, 0)
    return combos

def powersetHelper(input_set: List[int], subsets: List[List[int]], subset: List[int], n: int, i: int) -> None:
    if i >= n:
        subsets.append(subset.copy())
        return
    
    subset.append(input_set[i])
    powersetHelper(input_set, subsets, subset, n, i + 1)
    subset.pop()

    powersetHelper(input_set, subsets, subset, n, i + 1)

    return 
   
def powerset(input_set: List[int]) -> List[List[int]]:
    subsets = []
    subset = []
    n = len(input_set)
    powersetHelper(input_set, subsets, subset, n, 0)
    
    return subsets

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
    
    # skip 
    m = i
    for n in range(m, len(nums_list) - 1):
        if nums_list[n] == nums_list[n + 1]:
            i += 1
        else:
            break
            
    amountHelper(nums_list, target_sum, curr_sum, combos, combo, i + 1)

    return
        
def amount(nums_list: List[int], target_sum: int):
    combos = []
    combo = []
    sorted_list = sorted(nums_list)    
    amountHelper(sorted_list, target_sum, 0, combos, combo, 0)

    return combos

#######################################################################
#                       Graphs                                        #
#######################################################################
def topological_sort(n: int, edges: List[List[int]]) -> None:
    graph = {}
    for vertex in range(n):
        graph[vertex] = []
    
    for src, dst in edges:
        graph[src].append(dst)
    
    top_order = []   
    visited = []

    # terminates early if it detects a cycle
    def dfs(src: int, cycle_path: List[int]) -> bool:
        neighbors = graph[src]
        detect_cycle = False
        cycle_path.append(src)

        for adj in neighbors:
            if adj in cycle_path:
                return True

            if adj not in visited:
                detect_cycle = dfs(adj, cycle_path)
            
            if detect_cycle:
                break
        
        visited.append(src)
        top_order.append(src)
        cycle_path.pop()

        return detect_cycle

    cycle_detected = False
    for vertex in graph:
        if vertex not in visited:
            cycle_detected = dfs(vertex, []) 
        
        if cycle_detected:
            top_order = []
            break

    top_order.reverse()
    print(top_order)

    
def dijkstra_3d(graph: List[List[int]]) -> int: 
    n = len(graph)
    m = len(graph[0])
    frontier = []
    visited = set()
    path_cost = {}
    movement = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    visited.add((0, 0))
    path_cost[0] = graph[0][0]
    heapq.heappush(frontier, (graph[0][0], 0, 0))

    max_effort = -1
    while frontier:
        w, u, v = heapq.heappop(frontier) 
        vertex = u + v

        for vert_pos, horz_pos in movement:
            next_u = u + vert_pos
            next_v = v + horz_pos
            if (
                (next_u >= 0 and next_u < n)
                and (next_v >= 0 and next_v < m)
                and ((next_u, next_v) not in visited)
            ):
                curr_w = graph[u][v]
                next_w = graph[next_u][next_v]
                # Relax the edge
                heapq.heappush(frontier, (abs(next_w - curr_w), next_u, next_v))

        if vertex not in path_cost:
            path_cost[vertex] = w
            max_effort = max(max_effort, w)

        visited.add((u, v))

        # Could break early here since once we put in the final min effort into
        # the hash table, we are done
    
    return max_effort


def prims(n: int, edges: List[List[int]]) -> int:
    # Build the graph
    graph = {}
    for v in range(n):
        graph[v] = []
    
    for u, v, w in edges:
        graph[u].append([w, v])
        graph[v].append([w, u])
    
    min_edge_pq = [] 
    MST = set()
    cost = 0

    for w, v in graph[0]:
        heapq.heappush(min_edge_pq, [w, v])

    MST.add(0)
    while min_edge_pq and len(MST) < n:
        edge_w, to_v = heapq.heappop(min_edge_pq)

        if to_v not in MST:
            MST.add(to_v)
            cost += edge_w

        for adj_edge_w, adj_v in graph[to_v]:
            if adj_v not in MST:
                heapq.heappush(min_edge_pq, [adj_edge_w, adj_v])
        
    if len(MST) < n:
        return -1
    else:
        return cost
    
def kruskals(n: int, edges: List[List[int]]) -> int:
    forest = DisjointSet(n) 
    edges_copy = edges
    edges_copy.sort(key=lambda x: x[2])
    min_span_tree_cost = 0
    print(edges_copy)

    for u, v, w in edges_copy:
        if forest.union(u, v):
            min_span_tree_cost += w
            print(f'{u}, {v}')

    if forest.numOfComponents() > 1:
        return -1

    return min_span_tree_cost

def traveling_salesman_heuristic(graph: List[List[int]]) -> List[int]:
    ## Nearest Neighbor
    n = len(graph)
    m = len(graph[0])
    u = 0
    v = 0
    path = []
    
    while len(path) < n:
        min_weight = graph[u][v]

        if min_weight == 0 or v in path:
            v = (v + 1) % m
            if v == u:
                path.append(u)
                break
            else:
                continue

        for next_v in range(m):
            curr_min_weight = min(graph[u][next_v], min_weight)
            if curr_min_weight < min_weight and curr_min_weight > 0 and next_v not in path:
                v = next_v
                min_weight = curr_min_weight

        path.append(u)
        u = v
    
    if graph[0][v] > 0:
        path.append(0) 
    
    return path 


def main():
    n = 5
    edges = [
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
    ]
    path = traveling_salesman_heuristic(edges)
    print(path)

if __name__ == "__main__":
    main()