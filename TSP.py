def solve_tsp(graph):
    ## Nearest Neighbor
    n = len(graph)
    m = len(graph[0])
    u = 0
    v = 0
    path = []
    
    while len(path) < n:
        min_weight = graph[u][v]

        # find a starting min weight by essentially looping through
        if min_weight == 0 or v in path:
            v = (v + 1) % m
            # If we end up back at our start then we can't go anywhere
            # so we're done
            if v == u:
                path.append(u)
                break
            else:
                continue
        
        # This can be optimized but don't care right now
        # Optimization can be to limit our choices to ones we haven't been to yet
        # rather than searching every possible edge for our current vertex
        for next_v in range(m):
            curr_min_weight = min(graph[u][next_v], min_weight)
            if curr_min_weight < min_weight and curr_min_weight > 0 and next_v not in path:
                v = next_v
                min_weight = curr_min_weight

        path.append(u)
        u = v
    
    # We only have a tour if we can make it back to the start
    if graph[0][v] > 0:
        path.append(0) 
    
    return path 