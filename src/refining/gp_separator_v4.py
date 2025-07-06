from collections import deque
from typing import Dict, List, Tuple

import networkx as nx


def separate_communities_v4(graph: nx.Graph, communities: Dict[int, int] = None) -> Dict[int, int]:
    """
    Ultra-fast simplified community detection algorithm.
    Removes expensive operations while maintaining core functionality.
    """
    if not graph.nodes:
        return {}
    
    # Quick adjacency lookup
    adj = {node: set(graph.neighbors(node)) for node in graph.nodes}
    nodes = list(graph.nodes)
    
    def simple_split(community_nodes: List[int]) -> Tuple[List[int], List[int]]:
        """Ultra-simple community splitting based on connectivity."""
        if len(community_nodes) <= 2:
            return community_nodes, []
        
        # Find the node with highest degree as seed
        seed = max(community_nodes, key=lambda n: len(adj[n] & set(community_nodes)))
        
        # BFS-like expansion from seed
        visited = set()
        queue = deque([seed])
        group1 = []
        
        # Take roughly half the nodes that are most connected to seed
        target_size = len(community_nodes) // 2
        
        while queue and len(group1) < target_size:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            group1.append(node)
            
            # Add unvisited neighbors in community
            neighbors = [n for n in adj[node] if n in community_nodes and n not in visited]
            queue.extend(neighbors)
        
        # Remaining nodes go to group2
        group2 = [n for n in community_nodes if n not in group1]
        
        return group1, group2 if group2 else community_nodes[:len(community_nodes)//2], community_nodes[len(community_nodes)//2:]
    
    def should_split(community_nodes: List[int]) -> bool:
        """Simple heuristic: split if community is large and has low internal density."""
        if len(community_nodes) <= 3:
            return False
        
        community_set = set(community_nodes)
        internal_edges = 0
        possible_edges = len(community_nodes) * (len(community_nodes) - 1) // 2
        
        for node in community_nodes:
            internal_edges += len(adj[node] & community_set)
        
        internal_edges //= 2  # Each edge counted twice
        density = internal_edges / possible_edges if possible_edges > 0 else 0
        
        # Split if density is low and community is large enough
        return density < 0.3 and len(community_nodes) > 5
    
    # Iterative splitting using queue
    partition = {}
    community_queue = deque([nodes])
    community_id = 0
    
    while community_queue:
        current_community = community_queue.popleft()
        
        if not should_split(current_community):
            # Assign community ID to all nodes
            community_id += 1
            for node in current_community:
                partition[node] = community_id
        else:
            # Split and add to queue
            group1, group2 = simple_split(current_community)
            if group1 and group2:
                community_queue.append(group1)
                community_queue.append(group2)
            else:
                # Couldn't split effectively, keep as one community
                community_id += 1
                for node in current_community:
                    partition[node] = community_id

    return partition