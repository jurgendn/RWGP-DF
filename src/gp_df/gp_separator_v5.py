from typing import Dict, List, Set, Iterable, Tuple, Optional
import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import issparse

# ---------- helpers: volumes, cuts, ΔQγ ----------

def total_edge_weight(G: nx.Graph, weight: Optional[str] = "weight") -> float:
    return float(G.size(weight=weight))  # m

def degree_map(G: nx.Graph, nodes: Iterable[int], weight: Optional[str] = "weight") -> Dict[int, float]:
    return dict(G.degree(nodes, weight=weight))  # type: ignore

def volume(G: nx.Graph, nodes: Iterable[int], deg: Dict[int, float]) -> float:
    return float(sum(deg[n] for n in nodes))

def cut_weight_between(G: nx.Graph, S1: Set[int], S2: Set[int], weight: Optional[str] = "weight") -> float:
    # iterate neighbors of the smaller side for speed
    if len(S2) < len(S1):
        S1, S2 = S2, S1
    s2 = S2  # already a set
    e = 0.0
    for u in S1:
        for v, data in G[u].items():
            if v in s2:
                w = data.get(weight, 1.0) if weight is not None else 1.0
                e += float(w)
    return e

def delta_q_split(G: nx.Graph,
                  S1: Set[int], S2: Set[int],
                  deg: Dict[int, float],
                  gamma: float = 1.0,
                  weight: Optional[str] = "weight") -> float:
    m = total_edge_weight(G, weight=weight)
    if m <= 0:
        return 0.0
    two_m = 2.0 * m
    e12 = cut_weight_between(G, S1, S2, weight=weight)
    vol1 = volume(G, S1, deg)
    vol2 = volume(G, S2, deg)
    return - e12 / two_m + gamma * (vol1 * vol2) / (two_m * two_m)

# ---------- random-walk proposer (sparse) ----------

def rw_bipartition_sparse(G: nx.Graph, community: Iterable[int], steps: int = 10) -> Tuple[List[int], List[int]]:
    nodes = list(community)
    if len(nodes) <= 1:
        return nodes, []
    S = G.subgraph(nodes)
    A = nx.adjacency_matrix(S, nodelist=nodes)  # CSR
    if not issparse(A):
        A = csr_matrix(A)
    deg_arr = np.array(A.sum(axis=1)).ravel()
    # avoid division by zero
    safe_deg = deg_arr.copy()
    safe_deg[safe_deg == 0] = 1.0
    # row-stochastic P = D^{-1} A
    Dinv = csr_matrix((1.0 / safe_deg, (np.arange(len(nodes)), np.arange(len(nodes)))), shape=A.shape)
    P = Dinv @ A
    # start from node 0 distribution
    p = P.getrow(0).toarray().ravel()
    for _ in range(steps):
        p = p @ P  # dense vec × sparse mat
        if not isinstance(p, np.ndarray):
            p = p.toarray().ravel()
    # stationary threshold ~ degree proportion
    thresh = deg_arr / deg_arr.sum() if deg_arr.sum() > 0 else np.full_like(deg_arr, 1.0 / len(nodes))
    V1 = [nodes[i] for i in range(len(nodes)) if p[i] >= thresh[i]]
    V2 = [nodes[i] for i in range(len(nodes)) if p[i] <  thresh[i]]
    # ensure both sides non-empty
    if not V1 or not V2:
        mid = len(nodes) // 2
        V1, V2 = nodes[:mid], nodes[mid:]
    return V1, V2

# ---------- refinement class ----------

class RWGPRefinement:
    def __init__(self,
                 num_walks: int = 10,
                 resolution: float = 1.0,
                 weight: Optional[str] = "weight",
                 min_size: int = 10,
                 eps: float = 1e-9):
        self.num_walks = num_walks
        self.gamma = resolution
        self.weight = weight
        self.min_size = min_size
        self.eps = eps

    def _accept(self, G: nx.Graph, S: Set[int], S1: Set[int], S2: Set[int], deg: Dict[int, float]) -> bool:
        if len(S1) < self.min_size or len(S2) < self.min_size:
            return False
        dq = delta_q_split(G, S1, S2, deg=deg, gamma=self.gamma, weight=self.weight)
        return dq > self.eps

    def _update_partition_with_split(self, part: Dict[int, int], cid: int, S1: Iterable[int], S2: Iterable[int]) -> Dict[int, int]:
        new_id = max(part.values(), default=-1) + 1
        for v in S1:
            part[v] = cid
        for v in S2:
            part[v] = new_id
        return part

    def forward(self, G: nx.Graph, original: Dict[int, int], rounds: int = 1) -> Dict[int, int]:
        part = original.copy()
        for _ in range(rounds):
            changed = False
            # build communities
            comm: Dict[int, Set[int]] = {}
            for v, c in part.items():
                comm.setdefault(c, set()).add(v)
            # precompute degrees once (weighted)
            deg = degree_map(G, part.keys(), weight=self.weight)
            for cid, S in list(comm.items()):
                if len(S) < 2 * self.min_size:
                    continue
                C1, C2 = rw_bipartition_sparse(G, S, steps=self.num_walks)
                S1, S2 = set(C1), set(C2)
                if self._accept(G, S, S1, S2, deg):
                    part = self._update_partition_with_split(part, cid, S1, S2)
                    changed = True
            if not changed:
                break
        return part

# ---------- entrypoint ----------

def separate_communities_v5(
    graph: nx.Graph,
    communities: Dict[int, int],
    full_communities: Optional[Dict[int, int]] = None,
) -> Dict[int, int]:
    method = RWGPRefinement(num_walks=10, resolution=1.0, weight="weight", min_size=10, eps=1e-9)
    return method.forward(graph, communities, rounds=2)
