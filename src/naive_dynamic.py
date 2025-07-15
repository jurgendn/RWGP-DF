from collections import defaultdict
from time import time
from typing import Any, Callable, Dict, List, Optional, Text, Tuple

import networkx as nx
import numpy as np

from src.community_info import CommunityUtils
from src.components.factory import IntermediateResults


class NaiveDynamicLouvain:
    """
    Naive-dynamic Louvain algorithm implementation for community detection
    in dynamic networks.

    This class implements a naive approach that processes all vertices 
    regardless of edge deletions and insertions, without using the 
    dynamic frontier optimization.
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
        Initialize Naive-dynamic Louvain algorithm.

        Args:
            graph: NetworkX graph (undirected)
            tolerance: Convergence tolerance for local-moving phase
            max_iterations: Maximum iterations per local-moving phase
            verbose: Whether to print progress information
        """
        self.__shortname__ = "naive"
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

    def _calculate_weighted_degrees(self) -> np.ndarray:
        """
        Calculate weighted degree for each node.

        Returns:
            Array of weighted degrees for each node
        """
        degrees = np.zeros(len(self.nodes))
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

    def _is_affected_function(self, node_idx: int) -> bool:
        """Check if node is affected (always True for naive approach)"""
        return True

    def _in_affected_range_function(self, node_idx: int) -> bool:
        """Check if node is in affected range (always True for naive approach)"""
        return True

    def louvain_move(
        self, lambda_functions: Optional[Dict[str, Callable[[int], bool]]] = None
    ) -> int:
        """
        Perform local-moving phase of Louvain algorithm.

        This implements the local moving phase where ALL nodes are processed
        (naive approach) to maximize modularity. The algorithm continues until
        convergence or max_iterations is reached.

        Args:
            lambda_functions: Dictionary with 'is_affected' and 'in_affected_range'
                             callback functions. For naive approach, all nodes are processed.

        Returns:
            Number of iterations performed
        """
        if lambda_functions is None:
            # Default: process all nodes (naive approach)
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

            # Process nodes according to lambda functions (ALL nodes for naive approach)
            for node_idx in range(len(self.nodes)):
                if not in_affected_range(node_idx) or not is_affected(node_idx):
                    continue

                current_community = self.community[node_idx]

                # Find best community for this node
                neighbor_communities = self._get_neighbor_communities(node_idx)
                best_community = current_community
                max_delta_modularity = 0.0

                for community in neighbor_communities.keys():
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
        Add new nodes to the NaiveDynamicLouvain data structures.

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

                # Initialize community weight for the new community
                self.community_weights[node_idx] = 0.0
    def apply_batch_update(
        self,
        edge_deletions: Optional[List[Tuple]] = None,
        edge_insertions: Optional[List[Tuple]] = None,
    ) -> None:
        """
        Apply batch updates to the graph.

        This method handles dynamic updates to the graph by processing batches
        of edge deletions and insertions. Unlike the Dynamic Frontier approach,
        this does not track affected vertices as all vertices will be processed.
        
        Follows the auxiliary information update approach from the paper.

        Args:
            edge_deletions: List of edges to delete [(node1, node2), ...]
            edge_insertions: List of edges to insert [(node1, node2, weight), ...]
        """
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

                        # Get edge weight before removal
                        weight = self.graph[node1][node2].get("weight", 1.0)
                        
                        # Remove edge and incrementally update degrees
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

                # Add edge and incrementally update degrees
                self.graph.add_edge(node1, node2, weight=weight)
                self.weighted_degree[idx1] += weight
                self.weighted_degree[idx2] += weight
                self.total_edge_weight += weight

        # Update community weights using auxiliary information
        # This is more efficient than recalculating from scratch
        self.community_weights = self._calculate_community_weights()

        # Update community weights
        if self.verbose and (edge_deletions or edge_insertions):
            try:
                nx_community = nx.algorithms.community.louvain_communities(
                    self.graph, weight="weight"
                )
                modularity = nx.algorithms.community.modularity(
                    self.graph, nx_community, weight="weight"
                )
                print(f"Current NetworkX modularity: {modularity:.6f}")
            except Exception as e:
                if self.verbose:
                    print(f"Could not calculate NetworkX modularity: {e}")

    def run(
        self,
        edge_deletions: Optional[List[Tuple]] = None,
        edge_insertions: Optional[List[Tuple]] = None,
    ) -> Dict[Text, IntermediateResults]:
        """
        Run complete Naive-dynamic Louvain algorithm.

        This method applies batch updates and then processes ALL vertices
        to update community structures, without using dynamic frontier optimization.
        Following Algorithm 2 from the paper.

        Args:
            edge_deletions: List of edges to delete
            edge_insertions: List of edges to insert

        Returns:
            Dictionary mapping nodes to community IDs
        """
        # Start timing the algorithm
        start_time = time()

        # Apply batch update
        if edge_deletions or edge_insertions:
            self.apply_batch_update(edge_deletions, edge_insertions)

        # Define lambda functions for naive approach (all vertices are processed)
        lambda_functions = {
            "is_affected": self._is_affected_function,
            "in_affected_range": self._in_affected_range_function,
        }

        # Run local-moving phase with lambda functions
        self.louvain_move(lambda_functions)

        # Build current community assignment
        community_assignment = {self.nodes[i]: self.community[i] for i in range(len(self.nodes))}
        
        # Calculate runtime and modularity
        runtime = time() - start_time
        modularity = self.get_modularity()
        res = IntermediateResults(
            runtime=runtime,
            modularity=modularity,
            affected_nodes=len(self.nodes),  # All nodes processed
        )

        return {"Naive Dynamic Louvain": res}

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
