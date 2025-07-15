from time import time
from typing import Any, Dict, List, Optional, Text, Tuple, Literal

import networkx as nx
import numpy as np
from networkx.algorithms.community import modularity

from src.models.community_info import CommunityUtils
from src.components.factory import IntermediateResults

from src.components.dataset import SelectiveSampler
class StaticLouvain:
    def __init__(
        self,
        graph: nx.Graph,
        initial_communities: Optional[Dict[Any, int]] = None,
        sampler_type: Literal["selective", "full"] = "selective",
        tolerance: float = 1e-2,
        max_iterations: int = 20,
        verbose: bool = True,
    ) -> None:
        self.__shortname__ = "Static Louvain"
        self.sampler_type = sampler_type
        self.graph = graph.copy()
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.nodes = list(graph.nodes())
        self.node_to_idx = {node: idx for idx, node in enumerate(self.nodes)}

        if isinstance(graph, nx.DiGraph):
            self.graph = graph.to_undirected()

        if initial_communities is not None:
            communities_dict = initial_communities
        else:
            communities_dict = CommunityUtils.initialize_communities_with_networkx(
                self.graph
            )

        self.community = np.zeros(len(self.nodes), dtype=int)
        for node, community_id in communities_dict.items():
            if node in self.node_to_idx:
                node_idx = self.node_to_idx[node]
                self.community[node_idx] = community_id
        self.sampler = SelectiveSampler(self.graph, communities_dict)

    def _add_new_nodes(self, new_nodes: List[Any]) -> None:
        for node in new_nodes:
            if node not in self.node_to_idx:
                self.graph.add_node(node)
                node_idx = len(self.nodes)
                self.nodes.append(node)
                self.node_to_idx[node] = node_idx
                self.community = np.append(self.community, node_idx)

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

        # Process edge insertions
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
                num_communities=np.random.randint(2, 3),
            )
        if edge_deletions or edge_insertions:
            self.apply_batch_update(edge_deletions, edge_insertions)
        
        start_time = time()
        communities = nx.algorithms.community.louvain_communities(self.graph, seed=42)
        runtime = time() - start_time
        for community_id, community_nodes in enumerate(communities): # type: ignore
            for node in community_nodes:
                self.community[self.node_to_idx[node]] = community_id
        self.sampler.update_communities(
            {self.nodes[i]: self.community[i] for i in range(len(self.nodes))}
        )
        modularity_score = modularity(self.graph, communities)
        res = IntermediateResults(
            runtime=runtime,
            modularity=modularity_score,
            affected_nodes=len(self.nodes),  # All nodes processed
        )

        return {"Static Louvain": res}
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