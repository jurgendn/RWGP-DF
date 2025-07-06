import asyncio
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager, Process
from typing import Dict, List, Optional, Tuple

import networkx as nx

# ==================== MULTIPROCESSING VERSION ====================

def compute_modularity_mp(graph_data: Tuple, community: List[int]) -> float:
    """Multiprocessing-safe modularity computation."""
    # Reconstruct graph from data
    nodes, edges = graph_data
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    
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

def split_community_mp(graph_data: Tuple, community: List[int], steps: int = 5) -> Tuple[List[int], List[int]]:
    """Multiprocessing-safe community splitting."""
    nodes, edges = graph_data
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    
    nodes = list(community)
    n = len(nodes)
    
    if n <= 1:
        return nodes, []
    
    subgraph = graph.subgraph(nodes)
    adjacency_matrix = nx.adjacency_matrix(subgraph, nodelist=nodes).toarray()
    degrees = adjacency_matrix.sum(axis=1)
    degrees[degrees == 0] = 1
    
    P = (adjacency_matrix.T / degrees).T
    P_t0 = P[0, :].copy()
    
    for _ in range(steps):
        P_t0 = P_t0 @ P
        
    threshold = degrees / degrees.sum()

    V1 = [nodes[i] for i in range(n) if P_t0[i] >= threshold[i]]
    V2 = [nodes[i] for i in range(n) if P_t0[i] < threshold[i]]
    
    if not V1 and V2:
        V1 = [V2.pop()]
    elif not V2 and V1:
        V2 = [V1.pop()]
        
    return V1, V2

def process_community_division(args):
    """Worker function for multiprocessing community division."""
    graph_data, community, degrees, total_edges = args
    
    try:
        # Split community
        C1, C2 = split_community_mp(graph_data, community)
        
        # Validate division
        mod_original = compute_modularity_mp(graph_data, community)
        mod_c1 = compute_modularity_mp(graph_data, C1)
        mod_c2 = compute_modularity_mp(graph_data, C2)
        
        is_valid_division = (set(C1) == set(community) or set(C2) == set(community) or 
                           (mod_c1 + mod_c2 - mod_original) <= 0)
        
        if is_valid_division or len(C1) == 0 or len(C2) == 0:
            return 'assign', community
        else:
            return 'split', (C1, C2)
            
    except Exception:
        return 'assign', community

def separate_communities_v3(graph: nx.Graph, communities: Dict[int, int], 
                                          num_processes: Optional[int] = None) -> Dict[int, int]:
    """
    Multiprocessing version of community detection algorithm.
    
    Args:
        graph (nx.Graph): The input graph.
        communities (Dict[int, int]): Initial community assignments for nodes.
        num_processes (int): Number of processes to use. If None, uses CPU count.

    Returns:
        Dict[int, int]: Updated community assignments for nodes.
    """
    
    if num_processes is None:
        num_processes = mp.cpu_count()
    
    # Prepare graph data for multiprocessing (graphs aren't directly serializable)
    graph_data = (list(graph.nodes), list(graph.edges))
    degrees = {node: graph.degree[node] for node in graph.nodes}
    total_edges = graph.number_of_edges()
    
    # Initialize work queue and results
    work_queue = [list(graph.nodes)]
    partition = {}
    next_community_id = 1
    
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        while work_queue:
            # Prepare batch of work
            current_batch = []
            batch_size = min(len(work_queue), num_processes * 2)  # Process in batches
            
            for _ in range(batch_size):
                if work_queue:
                    community = work_queue.pop(0)
                    if len(community) <= 1:
                        # Assign single nodes directly
                        for node in community:
                            partition[node] = next_community_id
                        next_community_id += 1
                    else:
                        current_batch.append((graph_data, community, degrees, total_edges))
            
            if not current_batch:
                continue
            
            # Submit batch to process pool
            futures = [executor.submit(process_community_division, args) for args in current_batch]
            
            # Process results
            for future in as_completed(futures):
                try:
                    action, data = future.result()
                    
                    if action == 'assign':
                        community = data
                        for node in community:
                            partition[node] = next_community_id
                        next_community_id += 1
                    
                    elif action == 'split':
                        C1, C2 = data
                        if C1:
                            work_queue.append(C1)
                        if C2:
                            work_queue.append(C2)
                            
                except Exception as e:
                    print(f"Error processing community: {e}")
                    # Skip this community or handle error as needed
    
    return partition


def separate_communities_v2_multiprocessing_manager(graph: nx.Graph, communities: Dict[int, int], 
                                                  num_processes: Optional[int] = None) -> Dict[int, int]:
    """
    Alternative multiprocessing version using Manager for shared state.
    """
    
    if num_processes is None:
        num_processes = mp.cpu_count()
    
    def worker_process(work_queue, result_queue, graph_data, degrees, total_edges):
        """Worker process function."""
        while True:
            try:
                community = work_queue.get(timeout=1)
                if community is None:  # Poison pill
                    break
                
                result = process_community_division((graph_data, community, degrees, total_edges))
                result_queue.put(result)
                work_queue.task_done()
                
            except Exception:
                break
    
    # Setup multiprocessing components
    manager = Manager()
    work_queue = manager.Queue()
    result_queue = manager.Queue()
    
    # Prepare data
    graph_data = (list(graph.nodes), list(graph.edges))
    degrees = {node: graph.degree[node] for node in graph.nodes}
    total_edges = graph.number_of_edges()
    
    # Initialize with root community
    work_queue.put(list(graph.nodes))
    
    # Start worker processes
    processes = []
    for _ in range(num_processes):
        p = Process(target=worker_process, 
                   args=(work_queue, result_queue, graph_data, degrees, total_edges))
        p.start()
        processes.append(p)
    
    # Process results
    partition = {}
    next_community_id = 1
    
    try:
        while True:
            try:
                action, data = result_queue.get(timeout=1)
                
                if action == 'assign':
                    community = data
                    for node in community:
                        partition[node] = next_community_id
                    next_community_id += 1
                
                elif action == 'split':
                    C1, C2 = data
                    if C1:
                        work_queue.put(C1)
                    if C2:
                        work_queue.put(C2)
                        
            except Exception:
                if work_queue.empty() and result_queue.empty():
                    break
    
    finally:
        # Cleanup
        for _ in range(num_processes):
            work_queue.put(None)  # Poison pills
        
        for p in processes:
            p.join()
    
    return partition