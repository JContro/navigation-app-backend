import networkx as nx

def shortest_path_waypoint(start, waypoint, end, graph, weight="weight"): 
    first_leg = nx.shortest_path(G, source=start, target=waypoint)
    return first_leg