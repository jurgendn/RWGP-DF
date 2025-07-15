from typing import Any, Dict, List, Set, Tuple

import networkx as nx

from src.community_info import CommunityUtils


def compute_modularity(graph: nx.Graph, community: List[int] | Set[int]) -> float:
    q = 0
    all_links = graph.number_of_edges()
    subgraph = graph.subgraph(community)
    links_in_C = subgraph.number_of_edges()
    links_to_C = len(graph.edges(community))
    q += (links_in_C / all_links) - (links_to_C / all_links) ** 2
    return q


def delta_modularity_add(
    graph: nx.Graph,
    community: List[int] | Set[int],
    node: int,
    degrees: Dict[int, int],
    total_edges: int,
) -> float:
    community_set = set(community)
    d_node = degrees[node]
    sum_A_nj = sum(1 for j in community_set if graph.has_edge(node, j))
    sum_kj = sum(degrees[j] for j in community_set)
    delta_q = (sum_A_nj / total_edges) - (d_node * sum_kj) / (
        2 * total_edges * total_edges
    )
    return delta_q


def delta_modularity_keep(
    graph: nx.Graph,
    community: List[int] | Set[int],
    node: int,
    degrees: Dict[int, int],
    total_edges: int,
) -> float:
    community_set = set(community)
    d_node = degrees[node]
    l_iC = sum(1 for j in community_set if graph.has_edge(node, j))
    sum_kj = sum(degrees[j] for j in community_set)
    delta_q = (l_iC / total_edges) - (d_node * sum_kj) / (4 * total_edges * total_edges)
    return delta_q

# Main algorithm: GP-DF
def split_community(
    graph: nx.Graph, community: List[int] | Set[int], steps: int = 20
) -> Tuple[List[int], List[int]]:
    nodes = list(community)
    # index = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)
    # Create adjacency matrix more efficiently using vectorized operations
    subgraph = graph.subgraph(nodes)
    adjacency_matrix = nx.adjacency_matrix(subgraph, nodelist=nodes).toarray()
    degrees = adjacency_matrix.sum(axis=1)
    P = (adjacency_matrix.T / degrees).T
    P_t0 = P[0, :].copy()
    for _ in range(steps):
        P_t0 = P_t0 @ P
    threshold = degrees / degrees.sum()

    V1 = [nodes[i] for i in range(n) if P_t0[i] >= threshold[i]]
    V2 = [nodes[i] for i in range(n) if P_t0[i] < threshold[i]]
    return V1, V2


def adjust_communities(
    graph: nx.Graph,
    community1: List[int] | Set[int],
    community2: List[int] | Set[int],
    degrees: Dict[int, int],
    total_edges: int,
) -> Tuple[int, List[int], List[int]]:
    # Use sets for O(1) membership testing and removal
    C1 = set(community1)
    C2 = set(community2)

    # Batch collect nodes to move
    moves_to_C2 = []
    moves_to_C1 = []

    # Single pass through each community to determine moves
    for node in C1:
        delta_keep = delta_modularity_keep(
            graph=graph,
            community=C1,
            node=node,
            degrees=degrees,
            total_edges=total_edges,
        )
        delta_add = delta_modularity_add(
            graph=graph,
            community=C2,
            node=node,
            degrees=degrees,
            total_edges=total_edges,
        )
        if delta_keep < delta_add:
            moves_to_C2.append(node)
    for node in C2:
        delta_keep = delta_modularity_keep(
            graph=graph,
            community=C2,
            node=node,
            degrees=degrees,
            total_edges=total_edges,
        )
        delta_add = delta_modularity_add(
            graph=graph,
            community=C1,
            node=node,
            degrees=degrees,
            total_edges=total_edges,
        )
        if delta_keep < delta_add:
            moves_to_C1.append(node)

    # Execute all moves
    C1.difference_update(moves_to_C2)  # Remove all nodes moving to C2
    C2.update(moves_to_C2)  # Add all nodes moving to C2

    C2.difference_update(moves_to_C1)  # Remove all nodes moving to C1
    C1.update(moves_to_C1)  # Add all nodes moving to C1

    total_adjustments = len(moves_to_C2) + len(moves_to_C1)

    return total_adjustments, list(C1), list(C2)


def validate_division(
    graph: nx.Graph,
    community_idx: int,
    community1: List[int],
    community2: List[int],
    original_community: Dict[Any, Any],
):
    # Check if the new communities are the same as the original
    if set(community1) == set(original_community) or set(community2) == set(
        original_community
    ):
        return original_community, True

    tmp_communities = original_community.copy()
    tmp_communities = update_community(
        tmp_communities,
        community_id=community_idx,
        community_1=community1,
        community_2=community2,
    )
    list_new_communities = list(
        CommunityUtils.get_communities_as_sets(tmp_communities).values()
    )
    list_original_communities = list(
        CommunityUtils.get_communities_as_sets(original_community).values()
    )
    new_q = nx.algorithms.community.modularity(
        G=graph, communities=list_new_communities
    )
    original_q = nx.algorithms.community.modularity(
        G=graph, communities=list_original_communities
    )
    delta_q = new_q - original_q
    if delta_q > 0:
        print(f"Original Q: {original_q}, New Q: {new_q}")
        return tmp_communities, True
    return original_community, False


def update_community(
    partition: Dict[int, int],
    community_id: int,
    community_1: List,
    community_2: List,
) -> Dict[int, int]:
    """
    Update the partition with the new communities.
    """
    num_community = max(partition.values()) + 1
    for node in community_1:
        partition[node] = community_id
    for node in community_2:
        partition[node] = community_id + num_community
    # Reindex the partition to increment community IDs
    return partition


def separate_communities_v2_full(
    graph: nx.Graph,
    communities: Dict[int, int],
    full_communities: Dict[int, int] | None = None,
):
    if full_communities is None:
        original_communities = {}
        partition = {}
    else:
        original_communities = full_communities.copy()
        partition = full_communities.copy()


    for community_id in set(communities.values()):
        nodes_in_community = [
            node for node, comm_id in communities.items() if comm_id == community_id
        ]

        if len(nodes_in_community) > 1:
            C1, C2 = split_community(graph, nodes_in_community)
            _, is_improved = validate_division(
                graph=graph,
                community_idx=community_id,
                community1=C1,
                community2=C2,
                original_community=original_communities,
            )
            if is_improved:
                original_communities = update_community(partition, community_id, C1, C2)
    partition = {node: original_communities[node] for node in partition.keys()}
    return partition
