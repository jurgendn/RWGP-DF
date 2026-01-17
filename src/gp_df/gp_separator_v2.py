from typing import Dict, List, Tuple

import networkx as nx


def separate_communities_v2(
    graph: nx.Graph,
    communities: Dict[int, int],
    full_communities: Dict[int, List[int]] | None = None,
) -> Dict[int, int]:
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


    def adjust_communities(graph: nx.Graph, community1: List[int], community2: List[int], 
                        degrees: Dict[int, int], total_edges: int) -> Tuple[int, List[int], List[int]]:
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
        # Use sets for O(1) membership testing and removal
        C1 = set(community1)
        C2 = set(community2)
        
        # Batch collect nodes to move
        moves_to_C2 = []
        moves_to_C1 = []
        
        # Single pass through each community to determine moves
        for node in C1:
            delta_keep = delta_modularity_keep(graph, C1, node, degrees, total_edges)
            delta_add = delta_modularity_add(graph, C2, node, degrees, total_edges)
            if delta_keep < delta_add:
                moves_to_C2.append(node)
        
        for node in C2:
            delta_keep = delta_modularity_keep(graph, C2, node, degrees, total_edges)
            delta_add = delta_modularity_add(graph, C1, node, degrees, total_edges)
            if delta_keep < delta_add:
                moves_to_C1.append(node)
        
        # Execute all moves
        C1.difference_update(moves_to_C2)  # Remove all nodes moving to C2
        C2.update(moves_to_C2)             # Add all nodes moving to C2
        
        C2.difference_update(moves_to_C1)  # Remove all nodes moving to C1
        C1.update(moves_to_C1)             # Add all nodes moving to C1
        
        total_adjustments = len(moves_to_C2) + len(moves_to_C1)
        
        return total_adjustments, list(C1), list(C2)

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
        # Assume C = {C1, C2}
        # Compute modularity for C1, C2, and original community
        # Delta Q = Q(C1) + Q(C2) - Q(C)
        # Keep if Delta Q > 0
        delta_q = (compute_modularity(graph, community1) + compute_modularity(graph, community2) - compute_modularity(graph, original_community))
        if delta_q > 0:
            print(
            f"Modularity 1: {compute_modularity(graph, community1)}",
            f"Modularity 2: {compute_modularity(graph, community2)}",
            f"Modularity original: {compute_modularity(graph, original_community)}",
        )
            return True
        return False

    # Begin algorithm
    partition = {node: comm_id for node, comm_id in communities.items()}

    for community_id in set(communities.values()):
        nodes_in_community = [node for node, comm_id in communities.items() if comm_id == community_id]
        if len(nodes_in_community) > 1:
            C1, C2 = split_community(graph, nodes_in_community)
            if validate_division(graph, C1, C2, nodes_in_community):

                print(
                    f"Community {community_id} split into {len(C1)} and {len(C2)} nodes."
                )
                for node in C1:
                    partition[node] = community_id
                for node in C2:
                    partition[node] = community_id + 1

    return partition
