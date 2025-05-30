import heapq
"""
Info:
    3-D puzzle defined by a 2D matrix called
    puzzle[m][n]. The value at puzzle[row][column] represents 
    the height of the cell located at the coordinates [row][column]

    starting position is at (0, 0)

    Effort is defined as the maximum absolute difference between the
    heights of two consecutive cells along the path

Goal:
    Return the maximum effort value in the path that takes the least amount
    of effort to trek overall

Method:
    BFS -
        Dijkstra's Algo since we are looking for the path with the least
        amount of effort needed. 

State:
    path_cost -> hash table
    visited -> stack/hash set
    frontier -> minheap
"""

def minEffort(puzzle):
    n = len(puzzle)
    m = len(puzzle[0])
    frontier = []
    visited = set()
    path_cost = {}
    movement = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    heapq.heappush(frontier, (puzzle[0][0], 0, 0))
    visited.add((0, 0))
    path_cost[0] = puzzle[0][0]

    min_max_effort = -1
    while frontier:
        # d for distance
        d, u, v = heapq.heappop(frontier)
        vertex = u + v

        for vert, horz in movement:
            new_u = u + vert
            new_v = v + horz
            
            if (
                (new_u >= 0 and new_u < n)
                and (new_v >= 0 and new_v < m)
                and ((new_u, new_v) not in visited)
            ):
                curr_w = puzzle[u][v]
                new_w = puzzle[new_u][new_v]
                # We push the distance between two vertices instead of pushing
                # the weight of the next valid edge
                heapq.heappush(frontier, (abs(curr_w - new_w), new_u, new_v))
        
        if vertex not in path_cost:
            path_cost[vertex] = d
            min_max_effort = max(min_max_effort, d)
        
        visited.add((u, v))
        
        # Could break early here since once we put the final min effort into
        # the path_cost hash table, we're done
    
    return min_max_effort