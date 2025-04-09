import networkx as nx

from math import radians, sin, cos, sqrt, asin
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in kilometers
    # Convert degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return R * c

import heapq
from math import radians, sin, cos, sqrt, asin

def a_star(graph, start, goal, h, moneyMultiplier=0):
    dist = {node: float('inf') for node in graph}
    dist[start] = 0

    prev = {node: None for node in graph}

    # Priority queue stores (f_score, node), where f_score = dist[node] + h(node, goal)
    pq = [(dist[start] + h(start, goal, graph), start)]
    previousVisits = set()
    # previousVisits.add(start)
    while pq:
        # print("pq: ", pq)
        f_score, current_node = heapq.heappop(pq)
        # print("yes")
        if(current_node in previousVisits):
            continue
        # If we've reached the goal, we can stop
        if current_node == goal:
            break

        # If the f_score is out of date, skip
        if f_score > dist[current_node] + h(current_node, goal, graph):
            continue

        # For each neighbor, check if we have found a better path
        previousVisits.add(current_node)
        # print(graph_neighbors)
        for _, neighbor, key in graph.out_edges(current_node, keys=True):
            edge=graph.edges[(current_node, neighbor, key)]
            price=edge["price"]
            weight = edge["weight"] + moneyMultiplier * price
            tentative_g_score = dist[current_node] + weight


            if tentative_g_score < dist[neighbor]:
                dist[neighbor] = tentative_g_score
                prev[neighbor] = current_node
                # Recompute f-score = g-score + heuristic
                heapq.heappush(pq, (dist[neighbor] + h(neighbor, goal, graph), neighbor))

    return dist, prev

def heuristic(u, v, G):
    """Heuristic function for A* algorithm: Haversine distance between two nodes."""
    lat1, lon1 = G.nodes[u]['pos']
    lat2, lon2 = G.nodes[v]['pos']
    distance=haversine(lat1,lon1,lat2,lon2)
    return distance



def dijkstra(graph, start, moneyMultiplier=0):
    """Dijkstra's algorithm to find the shortest path from start to all other nodes."""
    dist = {node: float('inf') for node in graph}
    dist[start] = 0

    # Keep track of the path
    prev = {node: None for node in graph}
    pq = [(0, start)]

    while pq:
        current_dist, current_node = heapq.heappop(pq)

        # If this distance is outdated (i.e., we already found a better path), skip
        if current_dist > dist[current_node]:
            continue
        
        #Dijkstra’s doesn't need a goal parameter because it calculates the shortest path to all nodes

        # Explore neighbors
        for _, neighbor, key in graph.out_edges(current_node, keys=True):
            edge=graph.edges[(current_node, neighbor, key)]

            weight = edge["weight"] + moneyMultiplier * edge["price"]
            new_dist = dist[current_node] + weight
            if new_dist < dist[neighbor]:
                dist[neighbor] = new_dist
                prev[neighbor] = current_node
                heapq.heappush(pq, (new_dist, neighbor))

    return dist, prev

def is_dominated(new_vec, existing_vecs):
    """Check if new_vec is dominated by any vector in existing_vecs."""
    for vec in existing_vecs:
        if (vec[0] <= new_vec[0] and vec[1] <= new_vec[1]):
            return True
    return False

def reconstruct_path(start, goal, pareto_front, predecessors):
    """Trace back from goal to start for each Pareto-optimal cost vector."""
    paths = []
    for (total_dist, total_money) in pareto_front:
        path = []
        current_node = goal
        current_cost = (total_dist, total_money)
        
        # Backtrack from goal to start
        while current_node is not None:
            path.append(current_node)
            pred_info = predecessors[current_node].get(current_cost, None)
            
            if pred_info is None:
                break  # Reached the start node
            
            # Move to predecessor and update current cost
            prev_node, prev_dist, prev_money = pred_info
            current_node = prev_node
            current_cost = (prev_dist, prev_money)
        
        # Reverse to get path from start to goal
        path.reverse()
        if path[0] == start:
            paths.append((path, (total_dist, total_money)))
    
    return paths

def multi_criteria_a_star(graph, start, goal, h):
    # Track non-dominated (distance, money) vectors for each node
    cost_map = {node: [] for node in graph}
    cost_map[start] = [(0, 0)]  # (distance, money)

    # Predecessor map: {node: {(distance, money): (prev_node, prev_distance, prev_money)}}
    prev = {node: {} for node in graph}
    prev[start][(0, 0)] = None

    # Priority queue: (f_distance, node, current_distance, current_money)
    pq = []
    h_start = h(start, goal, graph)
    
    heapq.heappush(pq, (0 + h_start, start, 0, 0))

    while pq:
        f_dist, current, curr_dist, curr_money = heapq.heappop(pq)

        # Skip if this cost vector is no longer in cost_map (dominated)
        if (curr_dist, curr_money) not in cost_map[current]:
            continue

        # Early exit if we reach the goal (but continue to find all Pareto-optimal paths)
        if current == goal:
            # print("Reached goal with cost: ", (curr_dist, curr_money))
            continue 

        # Explore neighbors
        for _, neighbor, key in graph.out_edges(current, keys=True):
            edge_data = graph.edges[(current, neighbor, key)]
            new_dist = curr_dist + edge_data["weight"]
            new_money = curr_money + edge_data["price"]

            # Check if this new vector is dominated by existing ones
            new_vec = (new_dist, new_money)
            if is_dominated(new_vec, cost_map[neighbor]):
                # if neighbor==goal:
                #     print("Dominated vector: ", new_vec, " for neighbor: ", neighbor)
                #     print("Existing vectors: ", cost_map[neighbor])
                #     print("---------------")
                continue

            # Add to cost_map and update predecessors
            cost_map[neighbor] = [vec for vec in cost_map[neighbor] if not (new_vec[0] <= vec[0] and new_vec[1] <= vec[1])]
            cost_map[neighbor].append(new_vec)
            prev[neighbor][new_vec] = (current, curr_dist, curr_money)

            # Calculate heuristic for neighbor
            h_neighbor = h(neighbor, goal, graph)
            f_dist_new = new_dist + h_neighbor

            heapq.heappush(pq, (f_dist_new, neighbor, new_dist, new_money))

    # Extract Pareto-optimal paths to goal
    pareto_front = cost_map.get(goal, [])
    return pareto_front, prev