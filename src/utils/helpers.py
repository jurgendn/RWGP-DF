from typing import Any, Dict, List, Tuple

import networkx as nx
import numpy as np

from src.components.dataset import TemporalChanges


def capture_graph_statistics(
    graph: nx.Graph,
    initial_communities_dict: Dict[int, int],
    temporal_changes: List[TemporalChanges],
) -> Dict[str, Any]:
    """
    Capture comprehensive graph statistics for experiment logging.
    
    Args:
        graph: NetworkX graph object
        initial_communities_dict: Dictionary mapping nodes to community IDs
        temporal_changes: List of temporal changes (optional)
    
    Returns:
        Dictionary containing graph statistics
    """
    stats = {}
    
    # Basic graph properties
    stats["num_nodes"] = graph.number_of_nodes()
    stats["num_edges"] = graph.number_of_edges()
    stats["density"] = nx.density(graph)
    stats["is_connected"] = nx.is_connected(graph)
    
    # Degree statistics
    degrees = [d for n, d in graph.degree()]
    stats["avg_degree"] = np.mean(degrees)
    stats["std_degree"] = np.std(degrees)
    stats["min_degree"] = min(degrees)
    stats["max_degree"] = max(degrees)
    stats["median_degree"] = np.median(degrees)
    
    # Community statistics
    num_communities = len(set(initial_communities_dict.values()))
    stats["num_initial_communities"] = num_communities
    
    # Calculate community sizes
    community_sizes = {}
    for node, community_id in initial_communities_dict.items():
        community_sizes[community_id] = community_sizes.get(community_id, 0) + 1
    
    community_size_values = list(community_sizes.values())
    stats["avg_community_size"] = np.mean(community_size_values)
    stats["std_community_size"] = np.std(community_size_values)
    stats["min_community_size"] = min(community_size_values)
    stats["max_community_size"] = max(community_size_values)
    stats["median_community_size"] = np.median(community_size_values)
    
    # Calculate initial modularity
    communities_list = []
    for community_id in range(num_communities):
        community_nodes = [node for node, cid in initial_communities_dict.items() if cid == community_id]
        if community_nodes:
            communities_list.append(set(community_nodes))
    
    if communities_list:
        stats["initial_modularity"] = nx.algorithms.community.modularity(graph, communities_list)
    else:
        stats["initial_modularity"] = 0.0
    
    # Graph connectivity metrics
    if nx.is_connected(graph):
        stats["diameter"] = nx.diameter(graph)
        stats["average_shortest_path_length"] = nx.average_shortest_path_length(graph)
        stats["radius"] = nx.radius(graph)
    else:
        # For disconnected graphs, compute for largest component
        largest_cc = max(nx.connected_components(graph), key=len)
        largest_subgraph = graph.subgraph(largest_cc)
        stats["diameter"] = nx.diameter(largest_subgraph)
        stats["average_shortest_path_length"] = nx.average_shortest_path_length(largest_subgraph)
        stats["radius"] = nx.radius(largest_subgraph)
        stats["num_connected_components"] = nx.number_connected_components(graph)
        stats["largest_component_size"] = len(largest_cc)
        stats["largest_component_fraction"] = len(largest_cc) / stats["num_nodes"]
    
    # Clustering metrics
    stats["average_clustering"] = nx.average_clustering(graph)
    stats["transitivity"] = nx.transitivity(graph)
    
    # Centrality metrics (sample for large graphs)
    if stats["num_nodes"] <= 1000:  # Only compute for smaller graphs due to computational cost
        betweenness = nx.betweenness_centrality(graph)
        closeness = nx.closeness_centrality(graph)
        eigenvector = nx.eigenvector_centrality(graph, max_iter=1000)
        
        stats["avg_betweenness_centrality"] = np.mean(list(betweenness.values()))
        stats["max_betweenness_centrality"] = max(betweenness.values())
        stats["avg_closeness_centrality"] = np.mean(list(closeness.values()))
        stats["max_closeness_centrality"] = max(closeness.values())
        stats["avg_eigenvector_centrality"] = np.mean(list(eigenvector.values()))
        stats["max_eigenvector_centrality"] = max(eigenvector.values())
    
    # Temporal change statistics (if provided)
    if temporal_changes:
        stats["num_temporal_steps"] = len(temporal_changes)
        
        total_deletions = sum(len(change.deletions) for change in temporal_changes)
        total_insertions = sum(len(change.insertions) for change in temporal_changes)
        
        stats["total_edge_deletions"] = total_deletions
        stats["total_edge_insertions"] = total_insertions
        stats["avg_deletions_per_step"] = total_deletions / len(temporal_changes)
        stats["avg_insertions_per_step"] = total_insertions / len(temporal_changes)
        
        # Change intensity relative to original graph
        stats["deletion_intensity"] = total_deletions / stats["num_edges"] if stats["num_edges"] > 0 else 0
        stats["insertion_intensity"] = total_insertions / stats["num_edges"] if stats["num_edges"] > 0 else 0
        
        # Step-wise change statistics
        deletions_per_step = [len(change.deletions) for change in temporal_changes]
        insertions_per_step = [len(change.insertions) for change in temporal_changes]
        
        stats["std_deletions_per_step"] = np.std(deletions_per_step)
        stats["max_deletions_per_step"] = max(deletions_per_step) if deletions_per_step else 0
        stats["std_insertions_per_step"] = np.std(insertions_per_step)
        stats["max_insertions_per_step"] = max(insertions_per_step) if insertions_per_step else 0
    
    # Additional network properties
    try:
        # Assortativity (degree correlation)
        stats["degree_assortativity"] = nx.degree_assortativity_coefficient(graph)
    except:
        stats["degree_assortativity"] = None
    
    # Edge connectivity (for smaller graphs)
    if stats["num_nodes"] <= 500:
        try:
            stats["edge_connectivity"] = nx.edge_connectivity(graph)
            stats["node_connectivity"] = nx.node_connectivity(graph)
        except:
            stats["edge_connectivity"] = None
            stats["node_connectivity"] = None

    return stats


def generate_plot_filename(mode: str, dataset_config: Dict[str, Any]) -> str:
    """
    Generate plot filename based on mode and dataset configuration.

    Args:
        mode: The benchmark mode ('batch' or 'window_frame').
        dataset_config: Dictionary containing dataset configuration parameters.

    Returns:
        Generated filename for the plot.

    Raises:
        ValueError: If mode is not supported.
        KeyError: If required configuration keys are missing.
    """
    try:
        initial_fraction = dataset_config['initial_fraction']
        load_full_nodes = dataset_config['load_full_nodes']
    except KeyError as e:
        raise KeyError(f"Missing required configuration key: {e}")

    load_type = 'load_full_nodes' if load_full_nodes else 'load_partial_nodes'
    common_suffix = f"initial_fraction_{initial_fraction}_{load_type}_benchmark_plot.png"

    if mode == "batch":
        try:
            batch_range = dataset_config['batch_range']
            return f"batch_range_{batch_range}_{common_suffix}"
        except KeyError:
            raise KeyError("Missing 'batch_range' key for batch mode")
    elif mode == "window_frame":
        try:
            step_size = dataset_config['step_size']
            window_size = dataset_config['window_size']
            return f"window_size_{window_size}_step_size_{step_size}_{common_suffix}"
        except KeyError:
            raise KeyError("Missing 'step_size' key for window_frame mode")
    else:
        raise ValueError(
            f"Unsupported mode: '{mode}'. Expected 'batch' or 'window_frame'."
        )
    
def make_lfr_dataset(
    n: int,
    tau1: float,
    tau2: float,
    mu: float,
    min_degree: int | None = None,
    max_degree: int | None = None,
    min_community: int | None = None,
    max_community: int | None = None,
    seed: int = 42,
) -> nx.Graph:
    # Set default values if not provided
    if min_degree is None:
        min_degree = max(1, int(np.log(n)))
    if max_degree is None:
        max_degree = max(min_degree + 1, int(n / 10))
    if min_community is None:
        min_community = max(10, int(n / 100))
    if max_community is None:
        max_community = max(min_community + 1, int(n / 5))

    # Ensure parameters are valid
    min_degree = max(1, min_degree)
    max_degree = max(min_degree + 1, min(max_degree, n - 1))
    min_community = max(3, min_community)  # Communities need at least 3 nodes
    max_community = max(min_community + 1, min(max_community, n))

    # Validate mixing parameter
    if not 0 <= mu <= 1:
        raise ValueError("Mixing parameter mu must be between 0 and 1")

    # Validate tau parameters
    if tau1 < 2 or tau1 > 3:
        print(f"Warning: tau1={tau1} is outside typical range [2,3]")
    if tau2 < 1 or tau2 > 2:
        print(f"Warning: tau2={tau2} is outside typical range [1,2]")

    # Generate LFR benchmark graph
    graph = nx.LFR_benchmark_graph(
        n=n,
        tau1=tau1,
        tau2=tau2,
        mu=mu,
        min_degree=min_degree,
        max_degree=max_degree,
        min_community=min_community,
        max_community=max_community,
        seed=seed,
    )

    # Convert to simple graph if multigraph
    if isinstance(graph, nx.MultiGraph):
        graph = nx.Graph(graph)

    # Ensure node labels are integers starting from 0
    graph = nx.convert_node_labels_to_integers(graph, first_label=0, ordering="sorted")
    return graph

def make_gaussian_dataset(
    n: int,
    s: int,
    v: float,
    p_in: float,
    p_out: float,
    seed: int = 42,
):
    """
    Generate a Gaussian random partition graph with deterministic output if seed is set.

    Args:
        n: Total number of nodes.
        s: Mean size of communities.
        v: Variance of community sizes.
        p_in: Probability of edges within communities.
        p_out: Probability of edges between communities.
        seed: Random seed for reproducibility (default: 42).

    Returns:
        A NetworkX graph object.
    """
    graph = nx.gaussian_random_partition_graph(
        n=n, s=s, v=v, p_in=p_in, p_out=p_out, seed=seed
    )
    graph = nx.convert_node_labels_to_integers(graph, first_label=0, ordering="sorted")
    return graph

