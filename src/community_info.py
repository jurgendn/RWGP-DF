"""
Core data structures and utilities for Dynamic Frontier Louvain algorithm.

This module contains the shared data structures and utility functions used by both
synchronous and asynchronous implementations of the Dynamic Frontier Louvain algorithm.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Set, Tuple

import networkx as nx
import numpy as np
from sklearn.cluster import KMeans


@dataclass
class CommunityInfo:
    """Stores community information including weighted degrees and total edge weights"""

    vertex_degrees: Dict[int, float]  # K^t: weighted degree of each vertex
    community_weights: Dict[int, float]  # Σ^t: total edge weight of each community
    community_assignments: Dict[int, int]  # C^t: community assignment for each vertex


class CommunityUtils:
    """Utility functions for community detection operations"""

    @staticmethod
    def calculate_weighted_degrees(graph: nx.Graph) -> Dict[int, float]:
        """
        Calculate weighted degree for each node.

        Args:
            graph: NetworkX graph

        Returns:
            Dictionary mapping node to weighted degree
        """
        degrees = {}
        for node in graph.nodes():
            degree = 0.0
            for neighbor in graph.neighbors(node):
                weight = graph[node][neighbor].get("weight", 1.0)
                degree += weight
            degrees[node] = degree
        return degrees

    @staticmethod
    def calculate_community_weights(
        graph: nx.Graph, communities: Dict[int, int]
    ) -> Dict[int, float]:
        """
        Calculate total edge weight for each community.

        Args:
            graph: NetworkX graph
            communities: Community assignments

        Returns:
            Dictionary mapping community ID to total edge weight
        """
        weights = defaultdict(float)
        vertex_degrees = CommunityUtils.calculate_weighted_degrees(graph)

        for node, community_id in communities.items():
            weights[community_id] += vertex_degrees.get(node, 0.0)

        return dict(weights)

    @staticmethod
    def get_neighbor_communities(
        graph: nx.Graph, node: int, communities: Dict[int, int]
    ) -> Dict[int, float]:
        """
        Get communities of neighboring nodes and their connection weights.

        Args:
            graph: NetworkX graph
            node: Target node
            communities: Community assignments

        Returns:
            Dictionary mapping community ID to total connection weight
        """
        neighbor_communities = defaultdict(float)

        for neighbor in graph.neighbors(node):
            neighbor_community = communities.get(neighbor, neighbor)
            weight = graph[node][neighbor].get("weight", 1.0)
            neighbor_communities[neighbor_community] += weight

        return neighbor_communities

    @staticmethod
    def calculate_modularity_gain(
        graph: nx.Graph,
        node: int,
        current_community: int,
        target_community: int,
        communities: Dict[int, int],
        vertex_degrees: Dict[int, float],
        community_weights: Dict[int, float],
        total_edge_weight: float,
    ) -> float:
        """
        Calculate change in modularity for moving node to target community.

        Args:
            graph: NetworkX graph
            node: Node to move
            current_community: Current community ID
            target_community: Target community ID
            communities: Community assignments
            vertex_degrees: Vertex degrees
            community_weights: Community weights
            total_edge_weight: Total edge weight in graph

        Returns:
            Delta modularity value (positive values indicate improvement)
        """
        if current_community == target_community or total_edge_weight == 0:
            return 0.0

        # Get connection weights to current and target communities
        neighbor_communities = CommunityUtils.get_neighbor_communities(
            graph, node, communities
        )
        k_i_to_c = neighbor_communities.get(target_community, 0.0)
        k_i_to_d = neighbor_communities.get(current_community, 0.0)

        k_i = vertex_degrees.get(node, 0.0)
        sigma_c = community_weights.get(target_community, 0.0)
        sigma_d = community_weights.get(current_community, 0.0)

        # Delta-modularity formula (Equation 2 from Louvain paper)
        delta_q = (1.0 / total_edge_weight) * (k_i_to_c - k_i_to_d) - (
            k_i / (2.0 * total_edge_weight**2)
        ) * (k_i + sigma_c - sigma_d)

        return delta_q

    @staticmethod
    def calculate_modularity(graph: nx.Graph, communities: Dict[int, int]) -> float:
        """
        Calculate modularity of current community structure.

        Args:
            graph: NetworkX graph
            communities: Community assignments

        Returns:
            Modularity score
        """
        # vertex_degrees = CommunityUtils.calculate_weighted_degrees(graph)
        # total_edge_weight = sum(vertex_degrees.values()) / 2

        # if total_edge_weight == 0:
        #     return 0.0

        # modularity = 0.0

        # for edge in graph.edges(data=True):
        #     node1, node2 = edge[0], edge[1]
        #     weight = edge[2].get("weight", 1.0)

        #     # Kronecker delta: 1 if same community, 0 otherwise
        #     if communities.get(node1, node1) == communities.get(node2, node2):
        #         modularity += weight - (
        #             vertex_degrees.get(node1, 0) * vertex_degrees.get(node2, 0)
        #         ) / (2.0 * total_edge_weight)

        # return modularity / (2.0 * total_edge_weight)

        # This is a more efficient way to calculate modularity using NetworkX
        # Convert from {node_id: community_id} to {community_id: set(node_ids)}
        community_sets = list(
            CommunityUtils.get_communities_as_sets(communities=communities).values()
        )
        return nx.algorithms.community.modularity(
            G=graph, communities=community_sets
        )

    @staticmethod
    def get_communities_as_sets(communities: Dict[int, int]) -> Dict[int, Set[Any]]:
        """
        Get communities as a dictionary of sets.

        Args:
            communities: Community assignments

        Returns:
            Dictionary mapping community IDs to sets of nodes
        """
        community_sets = defaultdict(set)
        for node, community_id in communities.items():
            community_sets[community_id].add(node)
        return dict(community_sets)

    @staticmethod
    def initialize_communities_with_networkx(graph: nx.Graph) -> Dict[int, int]:
        """
        Initialize communities using a simple approach.

        Args:
            graph: NetworkX graph

        Returns:
            Dictionary mapping nodes to community IDs
        """
        # Initialize communities - start with each node in its own community
        # This avoids the NetworkX ArrayLike iteration issues
        # Use networkx's built-in community detection
        # communities_lv: List[Set[int]]
        communities = {}
        communities_lv = nx.algorithms.community.louvain_communities(graph)
        for community_id, community in enumerate(communities_lv):  # type: ignore
            for node in community:
                communities[node] = community_id
        return communities

        # communities = {}
        # for node_id, node in enumerate(graph.nodes()):
        #     communities[node] = node_id

        # return communities
