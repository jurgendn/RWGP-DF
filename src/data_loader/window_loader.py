from typing import List, Tuple

import networkx as nx

from src.components.dataset import TemporalChanges


def _collect_edges_by_date(data):
    """
    Helper to group edges by date (YYYYMMDD) from timestamp (assumed to be UNIX seconds).
    Returns a list of (date, [(node1, node2, timestamp), ...]) sorted by date.
    """
    from datetime import datetime
    edges_by_date = {}
    for node1, node2, timestamp in data:
        # Convert timestamp to date string (YYYYMMDD)
        date = datetime.fromtimestamp(timestamp).strftime('%Y%m%d')
        if date not in edges_by_date:
            edges_by_date[date] = []
        edges_by_date[date].append((node1, node2, timestamp))
    # Return sorted by date
    return sorted(edges_by_date.items())

def load_txt_dataset_with_timestamps(
    file_path: str,
    source_idx: int,
    target_idx: int,
    timestamp_idx: int,
    window_size: int,
    step_size: int,
    initial_fraction: float,
    max_steps: int | None = None,
    load_full_nodes: bool = True,
) -> Tuple[nx.Graph, List[TemporalChanges]]:

    data = []
    full_nodes = set()
    with open(file_path, "r") as f:
        for line in f:
            if line.strip() and not line.startswith("//"):
                parts = line.strip().split()
                if len(parts) >= 3:
                    node1, node2, timestamp = (
                        parts[source_idx],
                        parts[target_idx],
                        int(parts[timestamp_idx]),
                    )
                    data.append((node1, node2, timestamp))
                    full_nodes.add(node1)
                    full_nodes.add(node2)
    date_batches = _collect_edges_by_date(data)
    num_dates = len(date_batches)
    # Initial graph: first initial_fraction% of dates
    split_point = int(num_dates * initial_fraction)
    initial_edges = []
    for i in range(split_point):
        initial_edges.extend(date_batches[i][1])
    G = nx.Graph()

    if load_full_nodes is True:
        # Ensure all nodes from initial edges are added to the graph
        for node in full_nodes:
            if not G.has_node(node):
                G.add_node(node)

    for node1, node2, _ in initial_edges:
        if G.has_edge(node1, node2):
            G[node1][node2]["weight"] += 1
        else:
            G.add_edge(node1, node2, weight=1)
    # Sliding windows for the rest
    graphs = []
    for i in range(split_point, num_dates - window_size + 1, step_size):
        window_edges = []
        for j in range(i, i + window_size):
            window_edges.extend(date_batches[j][1])
        Gw = nx.Graph()
        for node1, node2, _ in window_edges:
            if Gw.has_edge(node1, node2):
                Gw[node1][node2]["weight"] += 1
            else:
                Gw.add_edge(node1, node2, weight=1)
        graphs.append(Gw)
    temporal_changes = []
    prev_edges = set(G.edges())
    for Gw in graphs:
        curr_edges = set(Gw.edges())
        insertions = [(u, v, Gw[u][v]["weight"]) for u, v in curr_edges - prev_edges]
        deletions = [(u, v) for u, v in prev_edges - curr_edges]
        temporal_changes.append({"deletions": deletions, "insertions": insertions})
        prev_edges = curr_edges
    if max_steps is not None:
        temporal_changes = temporal_changes[:max_steps]
    return G, temporal_changes
