import networkx as nx
from typing import Dict, List, Tuple

def split_community(graph: nx.Graph, community: List[int], steps: int = 5) -> Tuple[List[int], List[int]]:
    """
    Splits a community into two sub-communities using a random walk-based approach.
    This is the exact algorithm from v2, moved to module level for reuse.
    """
    nodes = list(community)
    n = len(nodes)
    
    if n <= 1:
        return nodes, []
    if n == 2:
        return [nodes[0]], [nodes[1]]
        
    # Create adjacency matrix more efficiently using vectorized operations
    subgraph = graph.subgraph(nodes)
    adjacency_matrix = nx.adjacency_matrix(subgraph, nodelist=nodes).toarray()
    degrees_matrix = adjacency_matrix.sum(axis=1)
    
    # Avoid division by zero
    degrees_matrix[degrees_matrix == 0] = 1e-10
    
    P = (adjacency_matrix.T / degrees_matrix).T
    P_t0 = P[0, :].copy()
    for _ in range(steps):
        P_t0 = P_t0 @ P
    threshold = degrees_matrix / degrees_matrix.sum()

    V1 = [nodes[i] for i in range(n) if P_t0[i] >= threshold[i]]
    V2 = [nodes[i] for i in range(n) if P_t0[i] < threshold[i]]
    
    # Ensure both communities are non-empty
    if not V1:
        V1 = [nodes[0]]
        V2 = nodes[1:]
    elif not V2:
        V2 = [nodes[-1]]
        V1 = nodes[:-1]
        
    return V1, V2

def separate_communities_v3(
    graph: nx.Graph,
    communities: Dict[int, int],
    full_communities: Dict[int, int] | None = None,
) -> Dict[int, int]:
    """
    Optimized version of the community separation algorithm from v2.
    Maintains the original algorithm structure while adding performance optimizations.

    Args:
        graph (nx.Graph): The input graph.
        communities (Dict[int, int]): Initial community assignments for nodes.

    Returns:
        Dict[int, int]: Updated community assignments for nodes.
    """

    # Pre-compute expensive operations once
    total_edges = graph.number_of_edges()
    if total_edges == 0:
        return communities.copy()
    
    # Cache degrees for performance
    degrees = {}
    for node in graph.nodes():
        degrees[node] = len(list(graph.neighbors(node)))

    def compute_modularity(graph: nx.Graph, community: List[int]) -> float:
        """
        Computes the modularity of a given community within the graph.
        """
        if not community:
            return 0.0
        
        all_links = graph.number_of_edges()
        if all_links == 0:
            return 0.0
            
        subgraph = graph.subgraph(community)
        links_in_C = subgraph.number_of_edges()
        links_to_C = len(graph.edges(community))
        q = (links_in_C / all_links) - (links_to_C / all_links) ** 2
        return q

    def delta_modularity_add(graph: nx.Graph, community: List[int], node: int, degrees: Dict[int, int], total_edges: int) -> float:
        """
        Calculates the change in modularity when adding a node to a community.
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
        """
        community_set = set(community)
        d_node = degrees[node]
        l_iC = sum(1 for j in community_set if graph.has_edge(node, j))
        sum_kj = sum(degrees[j] for j in community_set)
        delta_q = (l_iC / total_edges) - (d_node * sum_kj) / (4 * total_edges * total_edges)
        return delta_q

    def adjust_communities(graph: nx.Graph, community1: List[int], community2: List[int], 
                          degrees: Dict[int, int], total_edges: int) -> Tuple[int, List[int], List[int]]:
        """
        Adjusts two sub-communities by moving nodes to optimize modularity.
        """
        # Use sets for O(1) membership testing and removal
        C1 = set(community1)
        C2 = set(community2)
        
        # Batch collect nodes to move
        moves_to_C2 = []
        moves_to_C1 = []
        
        # Single pass through each community to determine moves
        for node in list(C1):  # Convert to list to avoid modification during iteration
            delta_keep = delta_modularity_keep(graph, list(C1), node, degrees, total_edges)
            delta_add = delta_modularity_add(graph, list(C2), node, degrees, total_edges)
            if delta_keep < delta_add:
                moves_to_C2.append(node)
        
        for node in list(C2):  # Convert to list to avoid modification during iteration
            delta_keep = delta_modularity_keep(graph, list(C2), node, degrees, total_edges)
            delta_add = delta_modularity_add(graph, list(C1), node, degrees, total_edges)
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
        """
        C1, C2 = split_community(graph, community)
        adjustments = -1
        iteration = 0
        max_iterations = 10  # Add limit to prevent infinite loops
        
        while (adjustments != 0 or iteration < 2) and iteration < max_iterations:
            adjustments, C1, C2 = adjust_communities(graph, C1, C2, degrees, total_edges)
            iteration += 1
        return C1, C2

    def validate_division(graph: nx.Graph, community1: List[int], community2: List[int], original_community: List[int]) -> bool:
        """
        Validates whether the division of a community is acceptable based on modularity.
        """
        if set(community1) == set(original_community) or set(community2) == set(original_community):
            return True
        
        if not community1 or not community2:
            return False
            
        try:
            orig_mod = compute_modularity(graph, original_community)
            new_mod = compute_modularity(graph, community1) + compute_modularity(graph, community2)
            return new_mod >= orig_mod
        except Exception:
            return False

    # Begin algorithm - same structure as v2
    partition = {node: comm_id for node, comm_id in communities.items()}

    for community_id in set(communities.values()):
        nodes_in_community = [node for node, comm_id in communities.items() if comm_id == community_id]
        if len(nodes_in_community) > 1:
            C1, C2 = split_community(graph, nodes_in_community)
            if validate_division(graph, C1, C2, nodes_in_community):
                for node in C1:
                    partition[node] = community_id
                for node in C2:
                    partition[node] = community_id + 1

    return partition

def separate_communities_v3_ultra_fast(graph: nx.Graph, communities: Dict[int, int]) -> Dict[int, int]:
    """
    Extreme performance version - trades some accuracy for maximum speed.
    Use this when you need the fastest possible execution.
    
    Args:
        graph (nx.Graph): The input graph.
        communities (Dict[int, int]): Initial community assignments for nodes.

    Returns:
        Dict[int, int]: Updated community assignments for nodes.
    """
    
    # Immediate early exits
    if not communities or len(communities) <= 1:
        return communities.copy()
    
    total_edges = graph.number_of_edges()
    if total_edges == 0:
        return communities.copy()
    
    # Pre-compute adjacency list once - fastest NetworkX access
    adj_list = {node: list(graph.neighbors(node)) for node in graph.nodes()}
    
    # Group communities with size filter
    community_groups = {}
    for node, comm_id in communities.items():
        if comm_id not in community_groups:
            community_groups[comm_id] = []
        community_groups[comm_id].append(node)
    
    # Filter out communities not worth processing
    viable_communities = {k: v for k, v in community_groups.items() 
                         if 3 <= len(v) <= 50}  # Sweet spot for processing
    
    if not viable_communities:
        return communities.copy()
    
    result = communities.copy()
    next_id = max(communities.values()) + 1
    
    for comm_id, nodes in viable_communities.items():
        # Use the same matrix-based split as the main function
        high_degree, low_degree = split_community(graph, nodes)
        
        # Ensure non-empty splits
        if not high_degree:
            high_degree = [nodes[0]]
            low_degree = nodes[1:]
        elif not low_degree:
            low_degree = [nodes[-1]]
            high_degree = nodes[:-1]
        
        # Super-fast validation using edge density heuristic
        # Count edges between the two groups
        cross_edges = 0
        total_possible = len(high_degree) * len(low_degree)
        
        # Sample only first few nodes for speed
        sample_size = min(3, len(high_degree))
        for node in high_degree[:sample_size]:
            cross_edges += sum(1 for neighbor in adj_list[node] if neighbor in low_degree)
        
        # Extrapolate cross-edge density
        if sample_size > 0:
            estimated_cross_edges = cross_edges * len(high_degree) // sample_size
            cross_density = estimated_cross_edges / max(total_possible, 1)
            
            # Accept split if cross-density is low (good separation)
            if cross_density < 0.5:  # Very permissive threshold for speed
                for node in high_degree:
                    result[node] = comm_id
                for node in low_degree:
                    result[node] = next_id
                next_id += 1
    
    return result
