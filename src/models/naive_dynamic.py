from time import time
from typing import Any, Callable, Dict, List, Literal, Optional, Text, Tuple

import networkx as nx

from src.components.factory import IntermediateResults
from src.models.base import LouvainMixin


class NaiveDynamicLouvain(LouvainMixin):
    def __init__(
        self,
        graph: nx.Graph,
        initial_communities: Optional[Dict[Any, int]] = None,
        sampler_type: Literal["selective", "full"] = "selective",
        tolerance: float = 1e-2,
        max_iterations: int = 20,
        verbose: bool = True,
        num_communities_range: Tuple[int, int] = (1, 5),
    ) -> None:
        super().__init__(
            graph=graph,
            initial_communities=initial_communities,
            sampler_type=sampler_type,
            tolerance=tolerance,
            max_iterations=max_iterations,
            verbose=verbose,
            num_communities_range=num_communities_range,
        )
        self.__shortname__ = "naive"

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

    def _is_affected_function(self, node_idx: int) -> bool:
        return True

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
        if edge_deletions:
            for edge in edge_deletions:
                if len(edge) == 2:
                    node1, node2 = edge
                else:
                    node1, node2 = edge[0], edge[1]
                if self.graph.has_edge(node1, node2):
                    if node1 in self.node_to_idx and node2 in self.node_to_idx:
                        idx1, idx2 = self.node_to_idx[node1], self.node_to_idx[node2]
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
        if self.sampler_type == "selective":
            edge_deletions = self.sampler.sample(
                num_samples=len(edge_deletions),
            )
        if edge_deletions or edge_insertions:
            self.apply_batch_update(edge_deletions, edge_insertions)
        lambda_functions = {
            "is_affected": self._is_affected_function,
            "in_affected_range": self._in_affected_range_function,
        }
        start_time = time()
        self.louvain_move(lambda_functions)
        runtime = time() - start_time
        community_assignment = {self.nodes[i]: self.community[i] for i in range(len(self.nodes))}
        self.sampler.update_communities(community_assignment)
        self.sampler.update_graph(self.graph)
        modularity = self.get_modularity()
        res = IntermediateResults(
            runtime=runtime,
            modularity=modularity,
            affected_nodes=len(self.nodes),
            num_communities= len(set(self.community)),
        )
        return {"Naive Dynamic Louvain": res}
