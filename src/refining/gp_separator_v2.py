import networkx as nx
import numpy as np
from typing import Dict, List, Tuple
from pyinstrument import Profiler

def separate_communities_v2(graph: nx.Graph, communities: Dict[int, int]) -> Dict[int, int]:
    """
    Splits a graph into communities based on modularity optimization.

    Args:
        graph (nx.Graph): The input graph.
        communities (Dict[int, int]): Initial community assignments for nodes.

    Returns:
        Dict[int, int]: Updated community assignments for nodes.
    """
    def compute_modularity(graph: nx.Graph, community: List[int]) -> float:
        """
        Computes the modularity of a given community within the graph.

        Args:
            graph (nx.Graph): The input graph.
            community (List[int]): List of nodes in the community.

        Returns:
            float: Modularity value of the community.
        """
        q = 0
        all_links = graph.number_of_edges()
        subgraph = graph.subgraph(community)
        links_in_C = subgraph.number_of_edges()
        links_to_C = len(graph.edges(community))
        q += (links_in_C / all_links) - (links_to_C / all_links) ** 2
        return q

    def delta_modularity_add(graph: nx.Graph, community: List[int], node: int, degrees: Dict[int, int], total_edges: int) -> float:
        """
        Calculates the change in modularity when adding a node to a community.

        Args:
            graph (nx.Graph): The input graph.
            community (List[int]): List of nodes in the community.
            node (int): Node to be added.
            degrees (Dict[int, int]): Node degrees.
            total_edges (int): Total number of edges in the graph.

        Returns:
            float: Change in modularity.
        """
        community_set = set(community)
        d_node = degrees[node]
        sum_A_nj = sum(1 for j in community_set if graph.has_edge(node, j))
        sum_kj = sum(degrees[j] for j in community_set)
        delta_q = (sum_A_nj / total_edges) - (d_node * sum_kj) / (2 * total_edges * total_edges)
        return delta_q

    def delta_modularity_keep(graph: nx.Graph, community: List[int], node: int, degrees: Dict[int, int], total_edges: int) -> float:
        """
        Calculates the change in modularity when keeping a node in its current community.

        Args:
            graph (nx.Graph): The input graph.
            community (List[int]): List of nodes in the community.
            node (int): Node to be kept.
            degrees (Dict[int, int]): Node degrees.
            total_edges (int): Total number of edges in the graph.

        Returns:
            float: Change in modularity.
        """
        community_set = set(community)
        d_node = degrees[node]
        l_iC = sum(1 for j in community_set if graph.has_edge(node, j))
        sum_kj = sum(degrees[j] for j in community_set)
        delta_q = (l_iC / total_edges) - (d_node * sum_kj) / (4 * total_edges * total_edges)
        return delta_q

    def split_community(graph: nx.Graph, community: List[int], steps: int = 5) -> Tuple[List[int], List[int]]:
        """
        Splits a community into two sub-communities using a random walk-based approach.

        Args:
            graph (nx.Graph): The input graph.
            community (List[int]): List of nodes in the community to be split.
            steps (int, optional): Number of random walk steps. Defaults to 5.

        Returns:
            Tuple[List[int], List[int]]: Two sub-communities resulting from the split.
        """
        nodes = list(community)
        index = {node: i for i, node in enumerate(nodes)}
        n = len(nodes)
        adjacency_matrix = np.zeros((n, n))
        for u in nodes:
            for v in graph[u]:
                if v in index:
                    i = index[u]
                    j = index[v]
                    adjacency_matrix[i, j] = 1
        degrees = adjacency_matrix.sum(axis=1)
        P = (adjacency_matrix.T / degrees).T
        P_t0 = P[0, :].copy()
        for _ in range(steps):
            P_t0 = P_t0 @ P
        threshold = degrees / degrees.sum()
        V1 = [nodes[i] for i in range(n) if P_t0[i] >= threshold[i]]
        V2 = [nodes[i] for i in range(n) if P_t0[i] < threshold[i]]
        return V1, V2

    def adjust_communities(graph: nx.Graph, community1: List[int], community2: List[int], degrees: Dict[int, int], total_edges: int) -> Tuple[int, List[int], List[int]]:
        """
        Adjusts two sub-communities by moving nodes to optimize modularity.

        Args:
            graph (nx.Graph): The input graph.
            community1 (List[int]): List of nodes in the first sub-community.
            community2 (List[int]): List of nodes in the second sub-community.
            degrees (Dict[int, int]): Node degrees.
            total_edges (int): Total number of edges in the graph.

        Returns:
            Tuple[int, List[int], List[int]]: Number of adjustments made and the updated sub-communities.
        """
        C1 = community1.copy()
        C2 = community2.copy()
        adjustments = 0
        moved_to_C2 = [i for i in C1 if delta_modularity_keep(graph, C1, i, degrees, total_edges) < delta_modularity_add(graph, C2, i, degrees, total_edges)]
        for i in moved_to_C2:
            C1.remove(i)
            C2.append(i)
            adjustments += 1
        moved_to_C1 = [i for i in C2 if delta_modularity_keep(graph, C2, i, degrees, total_edges) < delta_modularity_add(graph, C1, i, degrees, total_edges)]
        for i in moved_to_C1:
            C2.remove(i)
            C1.append(i)
            adjustments += 1
        return adjustments, C1, C2

    def refine_partition(graph: nx.Graph, community: List[int], degrees: Dict[int, int], total_edges: int) -> Tuple[List[int], List[int]]:
        """
        Refines a given community by splitting it into two sub-communities based on modularity optimization.

        Args:
            graph (nx.Graph): The input graph.
            community (List[int]): List of nodes in the community to be refined.
            degrees (Dict[int, int]): Node degrees.
            total_edges (int): Total number of edges in the graph.

        Returns:
            Tuple[List[int], List[int]]: Two sub-communities resulting from the refinement.
        """
        C1, C2 = split_community(graph, community)
        adjustments = -1
        iteration = 0
        while adjustments != 0 or iteration == 2:
            adjustments, C1, C2 = adjust_communities(graph, C1, C2, degrees, total_edges)
            iteration += 1
        return C1, C2

    def validate_division(graph: nx.Graph, community1: List[int], community2: List[int], original_community: List[int]) -> bool:
        """
        Validates whether the division of a community is acceptable based on modularity.

        Args:
            graph (nx.Graph): The input graph.
            community1 (List[int]): First sub-community.
            community2 (List[int]): Second sub-community.
            original_community (List[int]): Original community before division.

        Returns:
            bool: True if the division is valid, False otherwise.
        """
        if set(community1) == set(original_community) or set(community2) == set(original_community):
            return True
        return (compute_modularity(graph, community1) + compute_modularity(graph, community2) - compute_modularity(graph, original_community)) <= 0

    def recursive_divide(graph: nx.Graph, community: List[int], degrees: Dict[int, int], total_edges: int, partition: Dict[int, int]):
        """
        Recursively divides a community into sub-communities.

        Args:
            graph (nx.Graph): The input graph.
            community (List[int]): List of nodes in the community to be divided.
            degrees (Dict[int, int]): Node degrees.
            total_edges (int): Total number of edges in the graph.
            partition (Dict[int, int]): Community assignments for nodes.
        """
        C1, C2 = refine_partition(graph, community, degrees, total_edges)
        is_valid_division = validate_division(graph, C1, C2, community)
        if is_valid_division or len(C1) == 0 or len(C2) == 0:
            new_community_id = max(partition.values(), default=0) + 1
            for node in community:
                partition[node] = new_community_id
        else:
            recursive_divide(graph, C1, degrees, total_edges, partition)
            recursive_divide(graph, C2, degrees, total_edges, partition)

    # Begin algorithm
    node_list = list(graph.nodes)
    degrees = {node: graph.degree[node] for node in graph.nodes}
    total_edges = graph.number_of_edges()
    partition = {}
    recursive_divide(graph, node_list, degrees, total_edges, partition)
    return partition
