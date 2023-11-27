
import networkx as nx
import numpy as np

def create_index_map(G: nx.Graph):
    """
    Creates a mapping from matrix indices to node IDs.

    Args:
        G (nx.Graph): A NetworkX graph.

    Returns:
        dict: A dictionary where the keys are matrix indices and the values are node IDs.
    """
    id_map = {idx: node_id for idx, node_id in enumerate(G.nodes())}
    return id_map


def update_edge_weights(G: nx.Graph, adjacency_matrix: np.array.array, attribute: str = 'weight'):
    """
    Updates the weights of the edges in the graph based on the given adjacency matrix.

    Args:
        G (nx.Graph): A NetworkX graph.
        adjacency_matrix (np.array): A 2D numpy array representing the adjacency matrix of the graph.
        attribute (str, optional): The name of the edge attribute to be updated. Defaults to 'weight'.

    Returns:
        None
    """
    for i, from_node in enumerate(G.nodes()):
        for j, to_node in enumerate(G.nodes()):
            # Check if there is an edge between from_node and to_node
            if G.has_edge(from_node, to_node):
                # Assuming it's a MultiDiGraph, you might have multiple edges between the same nodes
                for key in G[from_node][to_node]:
                    # Update the weight
                    G[from_node][to_node][key][attribute] = adjacency_matrix[i, j]


                    
