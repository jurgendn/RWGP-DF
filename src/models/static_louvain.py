from time import time
from typing import Any, Dict, List, Literal, Optional, Text, Tuple

import networkx as nx
from networkx.algorithms.community import modularity

from src.components.factory import IntermediateResults
from src.models.base import LouvainMixin


class StaticLouvain(LouvainMixin):
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
        self.__shortname__ = "Static Louvain"

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
                        weight = self.graph[node1][node2].get("weight", 1.0)
                        self.graph.remove_edge(node1, node2)
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
                self.graph.add_edge(node1, node2, weight=weight)

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
        start_time = time()
        communities = nx.algorithms.community.louvain_communities(self.graph, seed=42)
        runtime = time() - start_time
        for community_id, community_nodes in enumerate(communities):  # type: ignore
            for node in community_nodes:
                self.community[self.node_to_idx[node]] = community_id
        self.sampler.update_communities(
            {self.nodes[i]: self.community[i] for i in range(len(self.nodes))}
        )
        self.sampler.update_graph(self.graph)
        modularity_score = modularity(self.graph, communities)
        res = IntermediateResults(
            runtime=runtime,
            modularity=modularity_score,
            affected_nodes=len(self.nodes),  # All nodes processed
            num_communities=len(communities),  # type: ignore
        )
        return {"Static Louvain": res}
