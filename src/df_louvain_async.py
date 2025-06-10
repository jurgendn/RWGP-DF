"""
Asynchronous Dynamic Frontier Louvain algorithm implementation.

This module provides the asynchronous implementation of the Dynamic Frontier Louvain
algorithm for community detection in dynamic networks with parallel processing capabilities.
"""

import asyncio
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx

from src.community_info import CommunityInfo, CommunityUtils


class AsyncDynamicFrontierLouvain:
    """
    Asynchronous implementation of Dynamic Frontier Louvain algorithm for community detection
    in dynamic networks with changing edge structures.

    This class provides an async version of the DFLouvain algorithm that can process
    multiple vertices in parallel and handle batch updates efficiently using asyncio.
    """

    def __init__(
        self,
        tolerance: float = 1e-2,
        max_iterations: int = 20,
        max_passes: int = 10,
        verbose: bool = True,
    ):
        """
        Initialize Async Dynamic Frontier Louvain algorithm.

        Args:
            tolerance: Convergence tolerance for local-moving phase
            max_iterations: Maximum iterations per local-moving phase
            max_passes: Maximum number of passes in the algorithm
            verbose: Whether to print progress information
        """
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.max_passes = max_passes
        self.tolerance_decline_factor = 10
        self.verbose = verbose

    async def dynamic_frontier_louvain(
        self,
        graph: nx.Graph,
        edge_deletions: Optional[List[Tuple[int, int]]] = None,
        edge_insertions: Optional[List[Tuple[int, int, float]]] = None,
        previous_communities: Optional[Dict[int, int]] = None,
        previous_vertex_degrees: Optional[Dict[int, float]] = None,
        previous_community_weights: Optional[Dict[int, float]] = None,
    ) -> CommunityInfo:
        """
        Main async DF Louvain algorithm

        Args:
            graph: Current graph snapshot
            edge_deletions: List of (u, v) edges to delete
            edge_insertions: List of (u, v, weight) edges to insert
            previous_communities: Previous community assignments
            previous_vertex_degrees: Previous weighted degrees
            previous_community_weights: Previous community total weights
        """

        # Initialize if no previous state
        if previous_communities is None:
            return await self._initial_community_detection(graph)

        # Step 1: Mark initial affected vertices
        affected_vertices = await self._mark_initial_affected_vertices(
            edge_deletions or [], edge_insertions or [], previous_communities
        )

        # Step 2: Update auxiliary information incrementally
        updated_info = await self._update_weights_incrementally(
            graph,
            edge_deletions or [],
            edge_insertions or [],
            previous_communities,
            previous_vertex_degrees or {},
            previous_community_weights or {},
        )

        # Step 3: Run parallel Louvain with incremental affected marking
        final_communities = await self._parallel_louvain_with_df(
            graph, updated_info, affected_vertices
        )

        return final_communities

    async def _initial_community_detection(self, graph: nx.Graph) -> CommunityInfo:
        """Run initial community detection on static graph"""

        # Initialize each node in its own community
        communities = {node: node for node in graph.nodes()}
        vertex_degrees = await self._compute_vertex_degrees_async(graph)
        community_weights = await self._compute_community_weights_async(
            graph, communities
        )

        initial_info = CommunityInfo(vertex_degrees, community_weights, communities)

        # Run Louvain algorithm
        all_affected = set(graph.nodes())
        result = await self._parallel_louvain_with_df(graph, initial_info, all_affected)

        return result

    async def _mark_initial_affected_vertices(
        self,
        edge_deletions: List[Tuple[int, int]],
        edge_insertions: List[Tuple[int, int, float]],
        previous_communities: Dict[int, int],
    ) -> Set[int]:
        """Mark vertices affected by batch updates"""

        async def process_deletions():
            affected = set()
            for u, v in edge_deletions:
                # Mark vertices if they belong to same community
                if previous_communities.get(u) == previous_communities.get(v):
                    affected.add(u)
                    affected.add(v)
            return affected

        async def process_insertions():
            affected = set()
            for u, v, _ in edge_insertions:
                # Mark vertices if they belong to different communities
                if previous_communities.get(u) != previous_communities.get(v):
                    affected.add(u)
                    affected.add(v)
            return affected

        # Process deletions and insertions in parallel
        deletion_task = asyncio.create_task(process_deletions())
        insertion_task = asyncio.create_task(process_insertions())

        deletion_affected, insertion_affected = await asyncio.gather(
            deletion_task, insertion_task
        )

        return deletion_affected | insertion_affected

    async def _update_weights_incrementally(
        self,
        graph: nx.Graph,
        edge_deletions: List[Tuple[int, int]],
        edge_insertions: List[Tuple[int, int, float]],
        previous_communities: Dict[int, int],
        previous_vertex_degrees: Dict[int, float],
        previous_community_weights: Dict[int, float],
    ) -> CommunityInfo:
        """Incrementally update vertex degrees and community weights"""

        # Copy previous information
        vertex_degrees = previous_vertex_degrees.copy()
        community_weights = previous_community_weights.copy()
        community_assignments = previous_communities.copy()

        async def apply_deletions():
            for u, v in edge_deletions:
                if graph.has_edge(u, v):
                    weight = graph[u][v].get("weight", 1.0)
                    vertex_degrees[u] = vertex_degrees.get(u, 0) - weight
                    vertex_degrees[v] = vertex_degrees.get(v, 0) - weight

                    # Update community weights
                    u_comm = community_assignments.get(u)
                    v_comm = community_assignments.get(v)
                    if u_comm is not None:
                        community_weights[u_comm] = (
                            community_weights.get(u_comm, 0) - weight
                        )
                    if v_comm is not None:
                        community_weights[v_comm] = (
                            community_weights.get(v_comm, 0) - weight
                        )

        async def apply_insertions():
            for u, v, weight in edge_insertions:
                vertex_degrees[u] = vertex_degrees.get(u, 0) + weight
                vertex_degrees[v] = vertex_degrees.get(v, 0) + weight

                # Update community weights
                u_comm = community_assignments.get(u, u)
                v_comm = community_assignments.get(v, v)
                if u_comm == v_comm:
                    community_weights[u_comm] = (
                        community_weights.get(u_comm, 0) + 2 * weight
                    )
                else:
                    community_weights[u_comm] = (
                        community_weights.get(u_comm, 0) + weight
                    )
                    community_weights[v_comm] = (
                        community_weights.get(v_comm, 0) + weight
                    )

        # Apply changes in parallel
        await asyncio.gather(
            asyncio.create_task(apply_deletions()),
            asyncio.create_task(apply_insertions()),
        )

        return CommunityInfo(vertex_degrees, community_weights, community_assignments)

    async def _parallel_louvain_with_df(
        self, graph: nx.Graph, community_info: CommunityInfo, initial_affected: Set[int]
    ) -> CommunityInfo:
        """Run Louvain algorithm with dynamic frontier approach"""

        current_graph = graph.copy()
        current_communities = community_info.community_assignments.copy()
        affected_vertices = initial_affected.copy()

        for pass_num in range(self.max_passes):
            if self.verbose:
                print(
                    f"Pass {pass_num + 1}: Processing {len(affected_vertices)} affected vertices"
                )

            # Local moving phase with incremental affected marking
            iterations, communities_changed = await self._local_moving_phase_async(
                current_graph, current_communities, community_info, affected_vertices
            )

            if iterations <= 1:
                if self.verbose:
                    print(f"Converged after {iterations} iteration(s)")
                break

            # Check if significant community reduction occurred
            old_communities = len(set(current_communities.values()))

            # Aggregation phase
            current_graph, current_communities = await self._aggregation_phase_async(
                current_graph, current_communities
            )

            new_communities = len(set(current_communities.values()))

            if self.verbose:
                print(
                    f"Communities reduced from {old_communities} to {new_communities}"
                )

            # Update tolerance for next pass
            self.tolerance /= self.tolerance_decline_factor

            # Mark all vertices as affected for next pass
            affected_vertices = set(current_graph.nodes())

        # Update final community info
        final_vertex_degrees = await self._compute_vertex_degrees_async(current_graph)
        final_community_weights = await self._compute_community_weights_async(
            current_graph, current_communities
        )

        return CommunityInfo(
            final_vertex_degrees, final_community_weights, current_communities
        )

    async def _local_moving_phase_async(
        self,
        graph: nx.Graph,
        communities: Dict[int, int],
        community_info: CommunityInfo,
        affected_vertices: Set[int],
    ) -> Tuple[int, bool]:
        """Async local moving phase with vertex pruning and incremental marking"""

        total_modularity_gain = 0
        iteration = 0
        unprocessed_vertices = affected_vertices.copy()

        for iteration in range(self.max_iterations):
            if not unprocessed_vertices:
                break

            # Process vertices in parallel chunks
            vertex_chunks = self._chunk_vertices(
                list(unprocessed_vertices), chunk_size=100
            )
            chunk_tasks = [
                asyncio.create_task(
                    self._process_vertex_chunk(
                        chunk, graph, communities, community_info
                    )
                )
                for chunk in vertex_chunks
            ]

            chunk_results = await asyncio.gather(*chunk_tasks)

            # Merge results
            iteration_gain = 0
            newly_affected = set()
            for moves, gain, affected in chunk_results:
                # Apply moves
                for vertex, new_community in moves.items():
                    communities[vertex] = new_community
                iteration_gain += gain
                newly_affected.update(affected)

            total_modularity_gain += iteration_gain

            # Update unprocessed vertices for next iteration
            unprocessed_vertices = newly_affected

            # Check convergence
            if iteration_gain < self.tolerance:
                break

        return iteration + 1, total_modularity_gain > self.tolerance

    async def _process_vertex_chunk(
        self,
        vertices: List[int],
        graph: nx.Graph,
        communities: Dict[int, int],
        community_info: CommunityInfo,
    ) -> Tuple[Dict[int, int], float, Set[int]]:
        """Process a chunk of vertices for community optimization"""

        moves = {}
        total_gain = 0
        newly_affected = set()

        for vertex in vertices:
            best_community, gain = await self._find_best_community_async(
                vertex, graph, communities, community_info
            )

            current_community = communities.get(vertex, vertex)
            if best_community != current_community and gain > 0:
                moves[vertex] = best_community
                total_gain += gain

                # Mark neighbors as potentially affected
                for neighbor in graph.neighbors(vertex):
                    newly_affected.add(neighbor)

        return moves, total_gain, newly_affected

    async def _find_best_community_async(
        self,
        vertex: int,
        graph: nx.Graph,
        communities: Dict[int, int],
        community_info: CommunityInfo,
    ) -> Tuple[int, float]:
        """Find the best community for a vertex based on modularity gain"""

        current_community = communities.get(vertex, vertex)
        best_community = current_community
        best_gain = 0

        # Get neighboring communities
        neighbor_communities = defaultdict(float)
        for neighbor in graph.neighbors(vertex):
            neighbor_community = communities.get(neighbor, neighbor)
            weight = graph[vertex][neighbor].get("weight", 1.0)
            neighbor_communities[neighbor_community] += weight

        # Calculate modularity gain for each neighboring community
        for community, edge_weight_to_comm in neighbor_communities.items():
            if community != current_community:
                gain = await self._calculate_modularity_gain_async(
                    vertex,
                    current_community,
                    community,
                    edge_weight_to_comm,
                    community_info,
                    graph,
                )

                if gain > best_gain:
                    best_gain = gain
                    best_community = community

        return best_community, best_gain

    async def _calculate_modularity_gain_async(
        self,
        vertex: int,
        from_community: int,
        to_community: int,
        edge_weight_to_comm: float,
        community_info: CommunityInfo,
        graph: nx.Graph,
    ) -> float:
        """Calculate modularity gain for moving vertex between communities"""

        total_weight = (
            sum(
                graph.get_edge_data(u, v, {}).get("weight", 1.0)
                for u, v in graph.edges()
            )
            / 2
        )  # Undirected graph

        if total_weight == 0:
            return 0

        vertex_degree = community_info.vertex_degrees.get(vertex, 0)
        from_comm_weight = community_info.community_weights.get(from_community, 0)
        to_comm_weight = community_info.community_weights.get(to_community, 0)

        # Calculate edge weight from vertex to its current community (excluding self-loops)
        edge_weight_to_from = sum(
            graph.get_edge_data(vertex, neighbor, {}).get("weight", 1.0)
            for neighbor in graph.neighbors(vertex)
            if community_info.community_assignments.get(neighbor) == from_community
            and neighbor != vertex
        )

        # Modularity gain calculation
        gain = (1 / (2 * total_weight)) * (
            edge_weight_to_comm - edge_weight_to_from
        ) - (vertex_degree / (4 * total_weight * total_weight)) * (
            to_comm_weight - from_comm_weight + vertex_degree
        )

        return gain

    async def _aggregation_phase_async(
        self, graph: nx.Graph, communities: Dict[int, int]
    ) -> Tuple[nx.Graph, Dict[int, int]]:
        """Aggregate communities into super-vertices"""

        # Create super-graph where each community becomes a vertex
        super_graph = nx.Graph()
        community_mapping = {}

        # Map communities to new vertex IDs
        unique_communities = list(set(communities.values()))
        for i, comm in enumerate(unique_communities):
            community_mapping[comm] = i
            super_graph.add_node(i)

        # Add edges between super-vertices
        edge_weights = defaultdict(float)

        for u, v, data in graph.edges(data=True):
            # Ensure both nodes have community assignments
            if u not in communities:
                communities[u] = u  # Assign node to its own community
            if v not in communities:
                communities[v] = v  # Assign node to its own community

            # Get community IDs, creating new mappings if needed
            u_comm_id = communities[u]
            v_comm_id = communities[v]

            if u_comm_id not in community_mapping:
                new_id = len(community_mapping)
                community_mapping[u_comm_id] = new_id
                super_graph.add_node(new_id)

            if v_comm_id not in community_mapping:
                new_id = len(community_mapping)
                community_mapping[v_comm_id] = new_id
                super_graph.add_node(new_id)

            u_comm = community_mapping[u_comm_id]
            v_comm = community_mapping[v_comm_id]
            weight = data.get("weight", 1.0)

            if u_comm != v_comm:
                edge_weights[(u_comm, v_comm)] += weight
            else:
                # Self-loop in community
                edge_weights[(u_comm, u_comm)] += weight

        # Add weighted edges to super-graph
        for (u, v), weight in edge_weights.items():
            if u == v:
                super_graph.add_edge(u, v, weight=weight * 2)  # Self-loop
            else:
                super_graph.add_edge(u, v, weight=weight)

        # Create new community assignments (each super-vertex in its own community)
        new_communities = {node: node for node in super_graph.nodes()}

        return super_graph, new_communities

    async def _compute_vertex_degrees_async(self, graph: nx.Graph) -> Dict[int, float]:
        """Compute weighted degrees of all vertices"""
        degrees = {}

        async def compute_chunk_degrees(vertices):
            chunk_degrees = {}
            for vertex in vertices:
                degree = sum(
                    graph.get_edge_data(vertex, neighbor, {}).get("weight", 1.0)
                    for neighbor in graph.neighbors(vertex)
                )
                chunk_degrees[vertex] = degree
            return chunk_degrees

        # Process vertices in parallel chunks
        vertex_chunks = self._chunk_vertices(list(graph.nodes()), chunk_size=100)
        tasks = [
            asyncio.create_task(compute_chunk_degrees(chunk)) for chunk in vertex_chunks
        ]

        chunk_results = await asyncio.gather(*tasks)

        # Merge results
        for chunk_degrees in chunk_results:
            degrees.update(chunk_degrees)

        return degrees

    async def _compute_community_weights_async(
        self, graph: nx.Graph, communities: Dict[int, int]
    ) -> Dict[int, float]:
        """Compute total edge weights for each community"""
        community_weights = defaultdict(float)

        for u, v, data in graph.edges(data=True):
            weight = data.get("weight", 1.0)
            u_comm = communities.get(u, u)
            v_comm = communities.get(v, v)

            if u_comm == v_comm:
                # Internal edge
                community_weights[u_comm] += weight * 2
            else:
                # External edge
                community_weights[u_comm] += weight
                community_weights[v_comm] += weight

        return dict(community_weights)

    def _chunk_vertices(self, vertices: List[int], chunk_size: int) -> List[List[int]]:
        """Split vertices into chunks for parallel processing"""
        return [
            vertices[i : i + chunk_size] for i in range(0, len(vertices), chunk_size)
        ]

    async def get_modularity_async(
        self, graph: nx.Graph, communities: Dict[int, int]
    ) -> float:
        """Calculate modularity score asynchronously"""
        return CommunityUtils.calculate_modularity(graph, communities)
