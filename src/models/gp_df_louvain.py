"""
Synchronous Dynamic Frontier Louvain algorithm implementation.

This module provides the synchronous implementation of the Dynamic Frontier Louvain
algorithm for community detection in dynamic networks.
"""
from collections import defaultdict
from time import time
from typing import Any, Callable, Dict, List, Literal, Optional, Text, Tuple

import networkx as nx
import numpy as np

from src.components.dataset import SelectiveSampler
from src.components.factory import (
    IntermediateResults,
    MethodDynamicResults,
)
from src.gp_df import separate_communities
from src.models.community_info import CommunityUtils
from src.models.base import LouvainMixin

class GPDynamicFrontierLouvain(LouvainMixin):
    def __init__(
        self,
        graph: nx.Graph,
        initial_communities: Optional[Dict[Any, int]] = None,
        sampler_type: Literal["selective", "full"] = "selective",
        tolerance: float = 1e-2,
        max_iterations: int = 20,
        refine_version: str = "v2",
        verbose: bool = True,
    ) -> None:
        super().__init__(
            graph=graph,
            initial_communities=initial_communities,
            sampler_type=sampler_type,
            tolerance=tolerance,
            max_iterations=max_iterations,
            verbose=verbose,
        )

        self.__shortname__ = "gpdf"
        self.refine_version = refine_version

    def _move_node(self, node_idx: int, new_community: int) -> None:
        old_community = self.community[node_idx]
        if old_community == new_community:
            return
        node_degree = self.weighted_degree[node_idx]
        self.community_weights[old_community] -= node_degree
        self.community_weights[new_community] = (
            self.community_weights.get(new_community, 0.0) + node_degree
        )
        self.community[node_idx] = new_community
        node = self.nodes[node_idx]
        for neighbor in self.graph.neighbors(node):
            if neighbor in self.node_to_idx:
                neighbor_idx = self.node_to_idx[neighbor]
                self.affected[neighbor_idx] = True

    def _is_affected_function(self, node_idx: int) -> bool:
        return self.affected[node_idx]

    def _in_affected_range_function(self, node_idx: int) -> bool:
        return True

    def louvain_move(
        self, lambda_functions: Optional[Dict[str, Callable[[int], bool]]] = None
    ) -> int:
        if lambda_functions is None:
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
        for _ in range(self.max_iterations):
            total_delta_modularity = 0.0
            moved_nodes = 0
            for node_idx in range(len(self.nodes)):
                if not in_affected_range(node_idx) or not is_affected(node_idx):
                    continue
                current_community = self.community[node_idx]
                neighbor_communities = self._get_neighbor_communities(node_idx)
                best_community = current_community
                max_delta_modularity = 0.0
                for community in neighbor_communities.keys():
                    if community != current_community:
                        delta_q = self._calculate_delta_modularity(node_idx, community)
                        if delta_q > max_delta_modularity:
                            max_delta_modularity = delta_q
                            best_community = community
                if best_community != current_community and max_delta_modularity > 0:
                    self._move_node(node_idx, best_community)
                    total_delta_modularity += max_delta_modularity
                    moved_nodes += 1
            num_iterations += 1
            if total_delta_modularity <= self.tolerance:
                if self.verbose:
                    print("Converged!")
                break
        return num_iterations

    def apply_batch_update(
        self,
        edge_deletions: Optional[List[Tuple]] = None,
        edge_insertions: Optional[List[Tuple]] = None,
    ) -> None:
        self.affected.fill(False)

        # Process edge deletions
        if edge_deletions:
            for edge in edge_deletions:
                if len(edge) == 2:
                    node1, node2 = edge
                else:
                    node1, node2 = edge[0], edge[1]

                if self.graph.has_edge(node1, node2):
                    if node1 in self.node_to_idx and node2 in self.node_to_idx:
                        idx1, idx2 = self.node_to_idx[node1], self.node_to_idx[node2]
                        if self.community[idx1] == self.community[idx2]:
                            self.affected[idx1] = True
                            self.affected[idx2] = True
                        weight = self.graph[node1][node2].get("weight", 1.0)
                        self.graph.remove_edge(node1, node2)
                        self.weighted_degree[idx1] -= weight
                        self.weighted_degree[idx2] -= weight
                        self.total_edge_weight -= weight
                    else:
                        self.graph.remove_edge(node1, node2)

        if edge_insertions:
            for edge in edge_insertions:
                if len(edge) == 3:
                    node1, node2, weight = edge
                else:
                    node1, node2 = edge[0], edge[1]
                    weight = 1.0
                new_nodes = []
                for node in [node1, node2]:
                    if node not in self.node_to_idx:
                        new_nodes.append(node)
                if new_nodes:
                    self._add_new_nodes(new_nodes)
                idx1, idx2 = self.node_to_idx[node1], self.node_to_idx[node2]
                if self.community[idx1] != self.community[idx2]:
                    self.affected[idx1] = True
                    self.affected[idx2] = True
                self.graph.add_edge(node1, node2, weight=weight)
                self.weighted_degree[idx1] += weight
                self.weighted_degree[idx2] += weight
                self.total_edge_weight += weight

        self.community_weights = self._calculate_community_weights()

    def run(
        self,
        edge_deletions: List[Tuple],
        edge_insertions: List[Tuple],
    ) -> Dict[Text, IntermediateResults]:
        df_results = MethodDynamicResults()
        gp_df_results = MethodDynamicResults()
        if self.sampler_type == "selective":
            edge_deletions = self.sampler.sample(
                num_samples=len(edge_deletions),
            )
        start_time = time()
        if edge_deletions is not None or edge_insertions is not None:
            self.apply_batch_update(edge_deletions, edge_insertions)
        lambda_functions = {
            "is_affected": self._is_affected_function,
            "in_affected_range": self._in_affected_range_function,
        }

        self.louvain_move(lambda_functions)
        df_runtime = time() - start_time

        df_results = IntermediateResults(
            runtime=df_runtime,
            modularity=self.get_modularity(),
            affected_nodes=len(self.get_affected_nodes()),
        )
        new_community = {self.nodes[i]: self.community[i] for i in range(len(self.nodes))}
        # Refine communities using GP - DF Louvain
        separated = self.refine_communities(
            new_community,
            edge_deletions=edge_deletions,
            edge_insertions=edge_insertions,
        )
        gp_runtime = time() - start_time
        updated_community = new_community.copy()
        updated_community.update(separated)
        for node, community_id in updated_community.items():
            if node in self.node_to_idx:
                node_idx = self.node_to_idx[node]
                self.community[node_idx] = community_id

        self.sampler.update_communities(updated_community)
        gp_df_results = IntermediateResults(
            runtime=gp_runtime,
            modularity=self.get_modularity(),
            affected_nodes=len(self.get_affected_nodes()),
        )
        return {"Dynamic Frontier Louvain": df_results, "GP - Dynamic Frontier Louvain": gp_df_results}

    def refine_communities(
        self,
        new_community: Dict[int, int],
        edge_deletions: List[Tuple],
        edge_insertions: List[Tuple],
    ) -> Dict[Any, int]:

        affected_nodes = set(self.get_affected_nodes())
        affected_communities = []

        for inserted_edge in edge_insertions:
            if len(inserted_edge) == 2:
                node1, node2 = inserted_edge
            else:
                node1, node2 = inserted_edge[0], inserted_edge[1]
            if new_community.get(node1) != new_community.get(node2):
                affected_communities.append(new_community.get(node1))
                affected_communities.append(new_community.get(node2))
        for deleted_edge in edge_deletions:
            if new_community.get(deleted_edge[0]) == new_community.get(deleted_edge[1]):
                # If the deleted edge connects two different communities, mark them as affected
                affected_communities.append(new_community.get(deleted_edge[0]))
                affected_communities.append(new_community.get(deleted_edge[1]))
        
        affected_comm_ids = set(
            self.community[self.node_to_idx[node]] for node in affected_nodes
        )
        affected_communities = set(affected_communities)
        affected_subgraph_nodes = [node for node in self.nodes if self.community[self.node_to_idx[node]] in affected_comm_ids]

        affected_communities = {node: new_community[node] for node in affected_subgraph_nodes}
        if len(affected_communities) == 0:
            return new_community

        for node in affected_subgraph_nodes:
            node_idx = self.node_to_idx[node]
            self.affected[node_idx] = True
        affected_nodes = self.get_affected_nodes()

        separated = separate_communities[self.refine_version](
            self.graph,
            communities=affected_communities,
            full_communities=new_community,
        )
        return separated
