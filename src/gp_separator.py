from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
from sklearn.cluster import KMeans



def separate_communities(
    graph: nx.Graph, communities: Dict[int, int]
) -> Dict[int, int]:
    """
    Separate communities in the graph using K-Means clustering.

    This method iterates through each community and applies K-Means to split
    it into two sub-communities based on node embeddings derived from random walks.

    Args:
        communities: Dictionary mapping nodes to community IDs

    Returns:
        Dictionary mapping nodes to new community IDs after separation
    """
    new_communities = []
    for community_id in set(communities.values()):
        # Get nodes in the current community
        community = [node for node, comm in communities.items() if comm == community_id]

        if len(community) <= 1:
            continue

        # Split the community into two sub-communities
        C1, C2 = split_one_community(graph, community)
        new_communities.append(C1)
        new_communities.append(C2)
        new_communities = [c for c in new_communities if c]
    # Create a new mapping of nodes to community IDs
    new_community_mapping = {}
    for i, community in enumerate(new_communities):
        for node in community:
            new_community_mapping[node] = i
    return new_community_mapping


def split_one_community(
    graph: nx.Graph, community: List[int]
) -> Tuple[List[int], List[int]]:
    """
    Split a community _C in graph G into two subcommunities using K-Means on
    embeddings derived from random walk and cosine normalization.

    Parameters:
        G : networkx.Graph
            The input graph
        _C : list/set
            Nodes belonging to a community

    Returns:
        tuple: (C1, C2) two node clusters (lists)
    """
    g = graph.subgraph(community)
    N = g.number_of_nodes()

    if N <= 1:
        return community, []

    # Adjacency matrix and degree
    A = nx.to_numpy_array(g, dtype=float)
    D_diag = np.sum(A, axis=1)
    if np.any(D_diag == 0):  # avoid division by zero
        return community, []

    d = np.sum(D_diag)
    P = A / D_diag[:, None]  # Transition matrix
    P_t = np.linalg.matrix_power(P, 10)  # P^t

    phi = D_diag / d
    phi_norm = phi / np.linalg.norm(phi)

    P_t_norm = normalize(P_t)
    D_vectors = P_t_norm - phi_norm.reshape(1, -1)  # Embedding

    # Use K-Means to split the D_vectors space
    try:
        labels = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(D_vectors)
    except Exception as e:
        print(f"Error during KMeans clustering: {e}")
        return community, []

    C1 = [community[i] for i in range(len(community)) if labels[i] == 0]
    C2 = [community[i] for i in range(len(community)) if labels[i] == 1]

    return C1, C2


def normalize(matrix: np.ndarray):
    """
    Perform cosine normalization on a matrix.

    Parameters:
        matrix : numpy.ndarray
            Input matrix to normalize

    Returns:
        numpy.ndarray: Normalized matrix
    """
    norms = np.linalg.norm(matrix, axis=1)
    norms[norms == 0] = 1.0  # Avoid division by zero
    return matrix / norms[:, None]
