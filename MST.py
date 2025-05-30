import heapq
"""
Info:
    connect multiple islands with bridges
    each bridge has a cost which is the distance between islands

Goal:
    connect all the islands at the minimum total cost

Constraints:
    connected, undirected, weighted graph

Method:
    Sounds like we have to build an MST
    Prims
    Kruskals
"""

def connect_bridges(graph):
    n = len(graph)
    m = n
    frontier = []
    visited = set()
    mst = []
    edge_count = 0

    for i in range(n):
        edge_weight = graph[0][i]
        if edge_weight != 0:
            heapq.heappush(frontier, (edge_weight, 0, i))
    
    visited.add(0) 

    while frontier and edge_count < (n-1):
        weight, src, dest = heapq.heappop(frontier)
        
        if dest in visited:
            continue

        for i in range(m):
            next_weight = graph[dest][i]
            if i not in visited and next_weight != 0:
                heapq.heappush(frontier, (next_weight, dest, i))

        visited.add(dest)
        mst.append((src, dest, weight))
        edge_count += 1
    
    return mst

    
"""
def main():
    g = [
        [0, 8, 5, 0, 0, 0, 0],
        [8, 0, 10, 2, 18, 0, 0],
        [5, 10, 0, 3, 0, 16, 0],
        [0, 2, 3, 0, 12, 30, 14],
        [0, 18, 0, 12, 0, 0, 4],
        [0, 0, 16, 30, 0, 0, 26],
        [0, 0, 0, 14, 4, 26, 0]
    ]
    connect_bridges(g)


if __name__ == "__main__":
    main()
"""