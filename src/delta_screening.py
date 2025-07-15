from collections import defaultdict
from time import time
from typing import Any, Dict, List, Optional, Text, Tuple

import networkx as nx
import numpy as np

from src.community_info import CommunityUtils
from src.components.factory import IntermediateResults


class DeltaScreeningLouvain:
    """
    Delta-screening Louvain algorithm implementation for community detection
    in dynamic networks.

    This class implements a delta-screening approach that uses modularity-based
    scoring to determine which vertices are affected by network changes,
    processing only those vertices instead of all vertices.
    """

    def __init__(
        self,
        graph: nx.Graph,
        initial_communities: Optional[Dict[Any, int]] = None,
        tolerance: float = 1e-2,
        max_iterations: int = 20,
        verbose: bool = True,
    ) -> None:
        """
        Initialize Delta-screening Louvain algorithm.

        Args:
            graph: NetworkX graph (undirected)
            tolerance: Convergence tolerance for local-moving phase
            max_iterations: Maximum iterations per local-moving phase
            verbose: Whether to print progress information
        """
        self.__shortname__ = "delta"
        self.graph = graph.copy()
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.nodes = list(graph.nodes())
        self.node_to_idx = {node: idx for idx, node in enumerate(self.nodes)}

        # Convert to undirected if needed
        if isinstance(graph, nx.DiGraph):
            self.graph = graph.to_undirected()

        # Initialize communities using NetworkX's Louvain algorithm
        if initial_communities is not None:
            communities_dict = initial_communities
        else:
            communities_dict = CommunityUtils.initialize_communities_with_networkx(
                self.graph
            )

        # Convert to community assignment array
        self.community = np.zeros(len(self.nodes), dtype=int)
        for node, community_id in communities_dict.items():
            if node in self.node_to_idx:
                node_idx = self.node_to_idx[node]
                self.community[node_idx] = community_id

        # Calculate initial weighted degrees and community weights
        self.weighted_degree = self._calculate_weighted_degrees()
        self.community_weights = self._calculate_community_weights()
        self.total_edge_weight = sum(self.weighted_degree) / 2

        # Affected vertices tracking (delta-screening specific)
        self.affected = np.zeros(len(self.nodes), dtype=bool)
        self.previous_community = None

    def _calculate_weighted_degrees(self) -> np.ndarray:
        """
        Calculate weighted degree for each node.

        Returns:
            Array of weighted degrees for each node
        """
        degrees = np.zeros(len(self.nodes), dtype=float)
        for i, node in enumerate(self.nodes):
            degree = 0.0
            for neighbor in self.graph.neighbors(node):
                weight = self.graph[node][neighbor].get("weight", 1.0)
                degree += weight
            degrees[i] = degree
        return degrees

    def _calculate_community_weights(self) -> Dict[int, float]:
        """
        Calculate total edge weight for each community.

        Returns:
            Dictionary mapping community ID to total edge weight
        """
        weights = defaultdict(float)
        for i, node in enumerate(self.nodes):
            community_id = self.community[i]
            weights[community_id] += self.weighted_degree[i]
        return dict(weights)

    def _get_neighbor_communities(self, node_idx: int) -> Dict[int, float]:
        """
        Get communities of neighboring nodes and their connection weights.

        Args:
            node_idx: Index of the node

        Returns:
            Dictionary mapping community ID to total connection weight
        """
        node = self.nodes[node_idx]
        neighbor_communities = defaultdict(float)

        for neighbor in self.graph.neighbors(node):
            if neighbor in self.node_to_idx:
                neighbor_idx = self.node_to_idx[neighbor]
                neighbor_community = self.community[neighbor_idx]
                weight = self.graph[node][neighbor].get("weight", 1.0)
                neighbor_communities[neighbor_community] += weight

        return neighbor_communities

    def _calculate_delta_modularity(
        self, node_idx: int, target_community: int
    ) -> float:
        """
        Calculate change in modularity for moving node to target community.

        Args:
            node_idx: Index of the node to move
            target_community: Target community ID

        Returns:
            Delta modularity value (positive values indicate improvement)
        """
        current_community = self.community[node_idx]

        if current_community == target_community:
            return 0.0

        # Get connection weights to current and target communities
        neighbor_communities = self._get_neighbor_communities(node_idx)
        k_i_to_c = neighbor_communities.get(target_community, 0.0)
        k_i_to_d = neighbor_communities.get(current_community, 0.0)

        k_i = self.weighted_degree[node_idx]
        sigma_c = self.community_weights.get(target_community, 0.0)
        sigma_d = self.community_weights.get(current_community, 0.0)

        # Delta-modularity formula (Equation 2 from Louvain paper)
        delta_q = (1.0 / self.total_edge_weight) * (k_i_to_c - k_i_to_d) - (
            k_i / (2.0 * self.total_edge_weight**2)
        ) * (k_i + sigma_c - sigma_d)

        return delta_q

    def _move_node(self, node_idx: int, new_community: int) -> None:
        """
        Move node to new community and update community weights.

        Args:
            node_idx: Index of the node to move
            new_community: Target community ID
        """
        old_community = self.community[node_idx]

        if old_community == new_community:
            return

        # Update community weights
        node_degree = self.weighted_degree[node_idx]
        self.community_weights[old_community] -= node_degree
        self.community_weights[new_community] = (
            self.community_weights.get(new_community, 0.0) + node_degree
        )

        # Update community assignment
        self.community[node_idx] = new_community

        # Mark neighbors as affected (for next iteration)
        self._mark_neighbors_affected(node_idx)

    def _get_current_connection_weight(self, node_idx: int, target_community: int) -> float:
        """Get current connection weight from node to target community."""
        neighbor_communities = self._get_neighbor_communities(node_idx)
        return neighbor_communities.get(target_community, 0.0)

    def _mark_neighbors_affected(self, node_idx: int) -> None:
        """Mark all neighbors of a node as affected."""
        node = self.nodes[node_idx]
        for neighbor in self.graph.neighbors(node):
            if neighbor in self.node_to_idx:
                neighbor_idx = self.node_to_idx[neighbor]
                self.affected[neighbor_idx] = True

    def _identify_affected_vertices(
        self, 
        edge_deletions: Optional[List[Tuple]] = None,
        edge_insertions: Optional[List[Tuple]] = None
    ) -> None:
        """
        Identify vertices affected by the batch update using delta-screening.
        
        Implementation follows the Delta-screening approach from:
        "DF Louvain: Fast Incrementally Expanding Approach for Community Detection on Dynamic Graphs"
        
        Key principles from the paper:
        1. For edge deletions: Only mark vertices as affected if the deletion 
           occurs within the same community (may cause community split)
        2. For edge insertions: Only consider cross-community insertions, 
           find the one with highest modularity gain potential
        3. Mark neighbors of affected vertices to handle cascading effects
        
        Args:
            edge_deletions: List of edges to delete
            edge_insertions: List of edges to insert
        """
        self.affected.fill(False)

        if self.previous_community is None:
            # First run, mark all as affected
            self.affected.fill(True)
            return

        # Process edge deletions - only for edges within same community
        if edge_deletions:
            for edge in edge_deletions:
                if len(edge) == 2:
                    node1, node2 = edge
                else:
                    node1, node2 = edge[0], edge[1]

                if node1 in self.node_to_idx and node2 in self.node_to_idx:
                    idx1, idx2 = self.node_to_idx[node1], self.node_to_idx[node2]
                    # According to paper: only mark if deletion is within same community
                    if self.previous_community[idx1] == self.previous_community[idx2]:
                        self.affected[idx1] = True
                        self.affected[idx2] = True
                        # Mark neighbors as affected
                        self._mark_neighbors_affected(idx1)
                        self._mark_neighbors_affected(idx2)

        # Process edge insertions - according to paper's delta-screening approach
        if edge_insertions:
            # Group insertions by source vertex (as per paper's Algorithm 3)
            insertion_groups = defaultdict(list)
            for edge in edge_insertions:
                if len(edge) == 3:
                    node1, node2, weight = edge
                else:
                    node1, node2 = edge[0], edge[1]
                    weight = 1.0

                if node1 in self.node_to_idx and node2 in self.node_to_idx:
                    idx1, idx2 = self.node_to_idx[node1], self.node_to_idx[node2]
                    insertion_groups[idx1].append((idx2, weight))
                    if idx1 != idx2:  # For undirected graphs
                        insertion_groups[idx2].append((idx1, weight))

            # For each source vertex, find highest delta-modularity cross-community insertion
            for source_idx, targets in insertion_groups.items():
                source_community = int(self.previous_community[source_idx])
                best_delta_modularity = -float('inf')
                best_target_community = None
                
                # Find insertion with highest delta-modularity among cross-community connections
                for target_idx, weight in targets:
                    target_community = int(self.previous_community[target_idx])
                    if target_community != source_community:
                        # Calculate potential delta-modularity gain (simplified)
                        delta_q_gain = weight / self.total_edge_weight  # Primary gain factor
                        
                        if delta_q_gain > best_delta_modularity:
                            best_delta_modularity = delta_q_gain
                            best_target_community = target_community

                # Mark as affected if beneficial cross-community insertion found
                if best_target_community is not None and best_delta_modularity > 0:
                    self.affected[source_idx] = True
                    self._mark_neighbors_affected(source_idx)

    def _is_affected_function(self, node_idx: int) -> bool:
        """Check if node is affected (delta-screening specific)"""
        return self.affected[node_idx]

    def _in_affected_range_function(self, node_idx: int) -> bool:
        """Check if node is in affected range (always True for delta-screening)"""
        return True
    def louvain_move(
        self, lambda_functions: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Perform local-moving phase of Louvain algorithm.

        This implements the local moving phase where only affected nodes
        (identified by delta-screening) are processed to maximize modularity.
        The algorithm continues until convergence or max_iterations is reached.

        Args:
            lambda_functions: Dictionary with 'is_affected' and 'in_affected_range'
                             callback functions. If None, uses delta-screening functions.

        Returns:
            Number of iterations performed
        """
        if lambda_functions is None:
            # Default: use delta-screening functions
            is_affected = self._is_affected_function
            in_affected_range = self._in_affected_range_function
        else:
            is_affected = lambda_functions.get(
                "is_affected", self._is_affected_function
            )
            in_affected_range = lambda_functions.get(
                "in_affected_range", self._in_affected_range_function
            )

        num_iterations = 0

        for iteration in range(self.max_iterations):
            total_delta_modularity = 0.0
            moved_nodes = 0

            # Process only affected nodes (delta-screening)
            for node_idx in range(len(self.nodes)):
                if not in_affected_range(node_idx) or not is_affected(node_idx):
                    continue

                current_community = self.community[node_idx]

                # Find neighboring communities
                neighbor_communities = self._get_neighbor_communities(node_idx)
                
                # Find best community to move to
                best_community = current_community
                max_delta_modularity = 0.0

                for community in neighbor_communities:
                    if community != current_community:
                        delta_q = self._calculate_delta_modularity(node_idx, community)
                        if delta_q > max_delta_modularity:
                            max_delta_modularity = delta_q
                            best_community = community

                # Move node if beneficial
                if best_community != current_community and max_delta_modularity > 0:
                    self._move_node(node_idx, best_community)
                    total_delta_modularity += max_delta_modularity
                    moved_nodes += 1

            num_iterations += 1

            if self.verbose:
                print(
                    f"Iteration {num_iterations}: moved {moved_nodes} nodes, "
                    f"ΔQ = {total_delta_modularity:.6f}"
                )

            # Check for convergence
            if total_delta_modularity <= self.tolerance:
                if self.verbose:
                    print("Converged!")
                break

        return num_iterations

    def _add_new_nodes(self, new_nodes: List[Any]) -> None:
        """
        Add new nodes to the DeltaScreeningLouvain data structures.

        This method handles the addition of new nodes that appear in temporal
        changes but weren't in the original graph.

        Args:
            new_nodes: List of new node IDs to add
        """
        for node in new_nodes:
            if node not in self.node_to_idx:
                # Add node to NetworkX graph first
                self.graph.add_node(node)

                # Update our data structures
                node_idx = len(self.nodes)
                self.nodes.append(node)
                self.node_to_idx[node] = node_idx

                # Extend arrays for the new node
                # Each new node starts in its own community
                self.community = np.append(
                    self.community, node_idx
                )

                self.weighted_degree = np.append(self.weighted_degree, 0.0)
                self.affected = np.append(self.affected, False)

                # Initialize community weight for the new community
                self.community_weights[node_idx] = 0.0

    def apply_batch_update(
        self,
        edge_deletions: Optional[List[Tuple]] = None,
        edge_insertions: Optional[List[Tuple]] = None,
    ) -> None:
        """
        Apply batch updates and identify affected vertices using delta-screening.

        This method handles dynamic updates to the graph by processing batches
        of edge deletions and insertions, while using delta-screening to
        identify which vertices are affected by these changes.

        Args:
            edge_deletions: List of edges to delete [(node1, node2), ...]
            edge_insertions: List of edges to insert [(node1, node2, weight), ...]
        """
        # Store previous community state for delta-screening
        self.previous_community = self.community.copy()

        # Identify affected vertices using delta-screening
        self._identify_affected_vertices(edge_deletions, edge_insertions)

        # Process edge deletions
        if edge_deletions:
            for edge in edge_deletions:
                if len(edge) == 2:
                    node1, node2 = edge
                else:
                    node1, node2 = edge[0], edge[1]

                if self.graph.has_edge(node1, node2):
                    # Check if both nodes exist in our mapping
                    if node1 in self.node_to_idx and node2 in self.node_to_idx:
                        idx1, idx2 = self.node_to_idx[node1], self.node_to_idx[node2]

                        # Remove edge and update degrees
                        weight = self.graph[node1][node2].get("weight", 1.0)
                        self.graph.remove_edge(node1, node2)
                        self.weighted_degree[idx1] -= weight
                        self.weighted_degree[idx2] -= weight
                        self.total_edge_weight -= weight
                    else:
                        # Just remove the edge if nodes not in our mapping
                        self.graph.remove_edge(node1, node2)

        # Process edge insertions
        if edge_insertions:
            for edge in edge_insertions:
                if len(edge) == 3:
                    node1, node2, weight = edge
                else:
                    node1, node2 = edge[0], edge[1]
                    weight = 1.0

                # Handle new nodes that weren't in the original graph
                new_nodes = []
                for node in [node1, node2]:
                    if node not in self.node_to_idx:
                        new_nodes.append(node)

                if new_nodes:
                    self._add_new_nodes(new_nodes)

                # Now we can safely access both nodes
                idx1, idx2 = self.node_to_idx[node1], self.node_to_idx[node2]

                # Add edge and update degrees
                self.graph.add_edge(node1, node2, weight=weight)
                self.weighted_degree[idx1] += weight
                self.weighted_degree[idx2] += weight
                self.total_edge_weight += weight

        # Update community weights
        self.community_weights = self._calculate_community_weights()

    def run(
        self,
        edge_deletions: Optional[List[Tuple]] = None,
        edge_insertions: Optional[List[Tuple]] = None,
    ) -> Dict[Text, IntermediateResults]:
        """
        Run complete Delta-screening Louvain algorithm.

        This method applies batch updates, identifies affected vertices using
        delta-screening, and processes only those vertices to update community
        structures efficiently.

        Args:
            edge_deletions: List of edges to delete
            edge_insertions: List of edges to insert

        Returns:
            Dictionary mapping nodes to community IDs
        """
        # Start timing the algorithm
        start_time = time()

        # Apply batch update and identify affected vertices
        if edge_deletions is not None or edge_insertions is not None:
            self.apply_batch_update(edge_deletions, edge_insertions)

        # Define lambda functions for affected vertices (delta-screening)
        lambda_functions = {
            "is_affected": self._is_affected_function,
            "in_affected_range": self._in_affected_range_function,
        }

        # Run local-moving phase (only on affected vertices)
        self.louvain_move(lambda_functions)

        # Build current community assignment
        community_assignment = {self.nodes[i]: self.community[i] for i in range(len(self.nodes))}
        
        # Calculate runtime and modularity
        runtime = time() - start_time
        modularity = self.get_modularity()
        affected_count = np.sum(self.affected)
        res = IntermediateResults(
            runtime=runtime,
            modularity=modularity,
            affected_nodes=affected_count,
        )

        return {"Delta Screening": res}

    def get_modularity(self) -> float:
        """
        Calculate modularity of current community structure.

        Modularity is a measure of the quality of a community structure.
        Higher values indicate better community structures.

        Returns:
            Modularity score
        """
        communities_dict = {
            self.nodes[i]: self.community[i] for i in range(len(self.nodes))
        }
        return CommunityUtils.calculate_modularity(self.graph, communities_dict)

    def get_communities(self) -> Dict[int, set]:
        """
        Get communities as a dictionary of sets.

        Returns:
            Dictionary mapping community IDs to sets of nodes
        """
        communities = defaultdict(set)
        for i, node in enumerate(self.nodes):
            communities[self.community[i]].add(node)
        return dict(communities)

    def get_affected_nodes(self) -> List[Any]:
        """
        Get list of affected nodes (delta-screening specific).

        Returns:
            List of nodes marked as affected by delta-screening
        """
        return [self.nodes[i] for i in range(len(self.nodes)) if self.affected[i]]
