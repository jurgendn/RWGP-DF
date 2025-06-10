from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from src.community_info import CommunityUtils
from pyinstrument import Profiler

def separate_communities(
    graph: nx.Graph, communities: Dict[int, int]
) -> Dict[int, int]:
    """
    Separate communities in the graph using K-Means clustering.

    Only updates communities that are actually split (i.e., where modularity improves).
    Unchanged communities retain their original IDs.

    Args:
        communities: Dictionary mapping nodes to community IDs

    Returns:
        Dictionary mapping nodes to new community IDs after separation
    """
    # Get initial communities as sets
    p = Profiler()
    with p:
        comm_dict = CommunityUtils.get_communities_as_sets(communities)
        origin_communities = [list(comm) for comm in comm_dict.values()]
        current_modularity = nx.algorithms.community.quality.modularity(
            G=graph,
            communities=origin_communities,
        )
        # Map node to current community id
        node_to_comm = {node: cid for cid, nodes in comm_dict.items() for node in nodes}
        # Track next available community id
        next_comm_id = max(comm_dict.keys()) + 1 if comm_dict else 0
        updated = True
        while updated:
            updated = False
            for comm_id, community in list(comm_dict.items()):
                if len(community) <= 1:
                    continue
                C1, C2 = split_one_community(graph, list(community))
                if not C1 or not C2:
                    continue
                candidate_communities = [nodes if cid != comm_id else C1 for cid, nodes in comm_dict.items() if cid != comm_id]
                candidate_communities.append(C1)
                candidate_communities.append(C2)
                candidate_modularity = nx.algorithms.community.quality.modularity(
                    G=graph,
                    communities=candidate_communities,
                )
                if candidate_modularity > current_modularity:
                    # Update mapping: assign new id to C2, keep C1 as comm_id
                    for node in C1:
                        node_to_comm[node] = comm_id
                    for node in C2:
                        node_to_comm[node] = next_comm_id
                    # Update comm_dict
                    comm_dict[comm_id] = set(C1)
                    comm_dict[next_comm_id] = set(C2)
                    next_comm_id += 1
                    current_modularity = candidate_modularity
                    updated = True
                    break  # Restart after any change
    # p.print()
    return node_to_comm
    


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
    if len(community) > 200:
        try:
            kmeans = KMeans(n_clusters=2, n_init=1, max_iter=50, random_state=0)
            # Use MiniBatchKMeans for large communities
            labels = kmeans.fit_predict(D_vectors)
            C1 = [community[i] for i in range(len(community)) if labels[i] == 0]
            C2 = [community[i] for i in range(len(community)) if labels[i] == 1]
            return C1, C2
        except Exception as e:
            print(f"Error during KMeans clustering: {e}")
            return community, []
    else:
        return community, []



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
