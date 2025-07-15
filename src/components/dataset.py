from typing import Dict, List, Text, Tuple

import networkx as nx
import numpy as np
from pydantic import BaseModel


class TemporalChanges(BaseModel):
    deletions: List[Tuple]
    insertions: List[Tuple]

class SelectiveSampler:
    def __init__(self, G: nx.Graph, communities: Dict[int, int]):
        self.G = G
        self.communities = communities

        self.num_static_edges = G.number_of_edges()

    def update_communities(self, communities: Dict[int, int]):
        self.communities = communities

    def __is_eligible_to_remove(
        self, u: int, v: int, target_communities: List[int] | np.ndarray
    ) -> bool:
        """Check if an edge can be removed based on community membership."""
        return (
            self.communities[u] in target_communities
            and self.communities[v] in target_communities
            and self.communities[u] == self.communities[v]
        )

    def sample(self, num_samples: int, num_communities: int) -> List[Tuple]:
        """Sampling edges from the graph based on community membership.
        Randomly selects edges within community among the specified number of communities.

        Args:
            num_samples (int): _description_
            num_communities (int): _description_

        Returns:
            List[Tuple]: _description_
        """        
        shuffled_edges = np.random.permutation(list(self.G.edges()))
        communities_list = list(self.communities.values())

        selected_communities = np.random.choice(
            communities_list, size=num_communities, replace=False
        )
        sampled_edges = []
        for edge in shuffled_edges:
            u, v = edge
            if self.__is_eligible_to_remove(u, v, selected_communities) is True:
                sampled_edges.append(edge)
                if len(sampled_edges) >= num_samples:
                    break
        return sampled_edges
    


class DataLoader:
    def __init__(
        self,
        dataset_path: Text,
        graph: nx.Graph,
        temporal_edges: List[TemporalChanges],
        initial_communities: Dict[int, int] | None = None,
    ):
        self.G = graph
        self.temporal_edges = temporal_edges
        self.dataset_path = dataset_path
        self.communities = self.init_communities(initial_communities)
        self.sampler = SelectiveSampler(self.G, self.communities)

    def init_communities(self, initial_communities: Dict[int, int] | None = None):
        if initial_communities is not None:
            communities = initial_communities
        else:
            nx_communities = nx.algorithms.community.louvain_communities(self.G)
            communities = {
                node: i
                for i, community in enumerate(nx_communities)  # type: ignore
                for node in community
            }
        return communities

    def update_communities(self, communities: Dict[int, int]):
        """Update the communities in the sampler."""
        self.communities = communities
        self.sampler.update_communities(communities)
    
    def __getitem__(self, item: int) -> Tuple[nx.Graph, Dict[int, int]]:
        """Get the graph and communities for a specific item."""
        return self.G, self.communities