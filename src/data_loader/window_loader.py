"""
Data loading utilities for DFLouvain benchmarks.
"""

from typing import Dict, List, Tuple

import networkx as nx
import pandas as pd


def load_graph_with_sliding_window(
    file_path: str,
    window_size: int,
    step_size: int,
    initial_fraction: float,
) -> List[nx.Graph]:
    """
    Load a sequence of graphs using a sliding window over edge timestamps.

    Args:
        file_path: Path to edge list file (format: node1 node2 timestamp)
        window_size: Number of edges in each window
        step_size: Number of edges to slide the window

    Returns:
        List of NetworkX graphs, one per window
    """
    # Read and sort edges by timestamp
    data = []
    with open(file_path, "r") as f:
        for line in f:
            if line.strip() and not line.startswith("//"):
                parts = line.strip().split()
                if len(parts) >= 3:
                    node1, node2, timestamp = int(parts[0]), int(parts[1]), int(parts[2])
                    data.append((node1, node2, timestamp))
    data.sort(key=lambda x: x[2])

    graphs = []
    for i in range(0, len(data) - window_size + 1, step_size):
        window_edges = data[i : i + window_size]
        G = nx.Graph()
        for node1, node2, _ in window_edges:
            if G.has_edge(node1, node2):
                G[node1][node2]["weight"] += 1
            else:
                G.add_edge(node1, node2, weight=1)
        graphs.append(G)
    return graphs


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

def load_sx_mathoverflow_sliding_window(
    file_path: str,
    window_size: int,
    step_size: int,
    initial_fraction: float,
    max_steps: int | None = None,
    load_full_nodes: bool = True,
) -> Tuple[nx.Graph, List[Dict]]:
    """
    Load MathOverflow dataset as a sequence of graphs using a sliding window over DATES (not raw timestamps).
    The initial graph is built from the first 50% of the time duration (dates), ensuring node coverage.
    Returns the initial graph and a list of temporal changes (insertions/deletions).
    """
    data = []
    full_nodes = set()
    with open(file_path, "r") as f:
        for line in f:
            if line.strip() and not line.startswith("//"):
                parts = line.strip().split()
                if len(parts) >= 3:
                    node1, node2, timestamp = int(parts[0]), int(parts[1]), int(parts[2])
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

def load_college_msg_sliding_window(
    file_path: str,
    window_size: int,
    step_size: int,
    initial_fraction: float,
    max_steps: int | None = None,
    load_full_nodes: bool = True,
) -> Tuple[nx.Graph, List[Dict]]:
    """
    Load CollegeMsg dataset as a sequence of graphs using a sliding window over DATES (not raw timestamps).
    The initial graph is built from the first 50% of the time duration (dates), ensuring node coverage.
    Returns the initial graph and a list of temporal changes (insertions/deletions).
    """
    data = []
    full_nodes = set()
    with open(file_path, "r") as f:
        for line in f:
            if line.strip() and not line.startswith("//"):
                parts = line.strip().split()
                if len(parts) >= 3:
                    node1, node2, timestamp = int(parts[0]), int(parts[1]), int(parts[2])
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

def load_bitcoin_sliding_window(
    file_path: str,
    window_size: int,
    step_size: int,
    initial_fraction: float,
    max_steps: int | None = None,
    load_full_nodes: bool = True,
) -> Tuple[nx.Graph, List[Dict]]:
    """
    Load Bitcoin dataset as a sequence of graphs using a sliding window over DATES (not raw timestamps).
    The initial graph is built from the first 50% of the time duration (dates), ensuring node coverage.
    Returns the initial graph and a list of temporal changes (insertions/deletions).
    """
    df = pd.read_csv(
        file_path, header=None, names=["source", "target", "rating", "timestamp"]
    )
    df = df.sort_values("timestamp")
    data = [
        (int(row["source"]), int(row["target"]), int(row["timestamp"]))
        for _, row in df.iterrows()
    ]
    
    full_nodes = set(df["source"]).union(set(df["target"]))
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

