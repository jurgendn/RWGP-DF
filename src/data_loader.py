"""
Data loading utilities for DFLouvain benchmarks.
"""

from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd


def load_sx_mathoverflow_dataset(
    file_path: str, batch_range: float = 1e-3, initial_fraction: float = 0.3
) -> Tuple[nx.Graph, List[Dict]]:
    """
    Load the College Message dataset and prepare it for dynamic analysis.

    Format: node1 node2 timestamp

    Args:
        file_path: Path to CollegeMsg.txt

    Returns:
        Tuple of (initial_graph, temporal_changes)
    """
    # Read the data
    data = []
    with open(file_path, "r") as f:
        for line in f:
            if line.strip() and not line.startswith("//"):
                parts = line.strip().split()
                if len(parts) >= 3:
                    node1, node2, timestamp = (
                        int(parts[0]),
                        int(parts[1]),
                        int(parts[2]),
                    )
                    data.append((node1, node2, timestamp))

    # Sort by timestamp
    data.sort(key=lambda x: x[2])

    # Create initial graph with first 30% of edges
    split_point = int(len(data) * initial_fraction)
    initial_edges = data[:split_point]

    # Add all nodes to graph
    G = nx.Graph()

    for node1, node2, _ in initial_edges:
        if G.has_edge(node1, node2):
            G[node1][node2]["weight"] += 1
        else:
            G.add_edge(node1, node2, weight=1)

    # Create temporal changes for remaining edges
    remaining_edges = data[split_point:]
    batch_size = int(split_point * batch_range)  # 10 time steps

    temporal_changes = []
    for i in range(0, len(remaining_edges), batch_size):
        batch = remaining_edges[i : i + batch_size]
        insertions = [(node1, node2, 1) for node1, node2, _ in batch]
        temporal_changes.append({"deletions": [], "insertions": insertions})

    return G, temporal_changes


def load_college_msg_dataset(
    file_path: str, batch_range: float = 1e-3, initial_fraction: float = 0.3
) -> Tuple[nx.Graph, List[Dict]]:
    """
    Load the College Message dataset and prepare it for dynamic analysis.

    Format: node1 node2 timestamp

    Args:
        file_path: Path to CollegeMsg.txt

    Returns:
        Tuple of (initial_graph, temporal_changes)
    """
    # Read the data
    data = []
    with open(file_path, "r") as f:
        for line in f:
            if line.strip() and not line.startswith("//"):
                parts = line.strip().split()
                if len(parts) >= 3:
                    node1, node2, timestamp = (
                        int(parts[0]),
                        int(parts[1]),
                        int(parts[2]),
                    )
                    data.append((node1, node2, timestamp))

    # Sort by timestamp
    data.sort(key=lambda x: x[2])

    # Create initial graph with first 30% of edges
    split_point = int(len(data) * initial_fraction)
    initial_edges = data[:split_point]

    # Add all nodes to graph
    G = nx.Graph()
    # G.add_nodes_from(
    #     set(node1 for node1, _, _ in data).union(set(node2 for _, node2, _ in data))
    # )

    for node1, node2, _ in initial_edges:
        if G.has_edge(node1, node2):
            G[node1][node2]["weight"] += 1
        else:
            G.add_edge(node1, node2, weight=1)

    # Create temporal changes for remaining edges
    remaining_edges = data[split_point:]
    batch_size = int(split_point * batch_range)  # 10 time steps

    temporal_changes = []
    for i in range(0, len(remaining_edges), batch_size):
        batch = remaining_edges[i : i + batch_size]
        insertions = [(node1, node2, 1) for node1, node2, _ in batch]
        temporal_changes.append({"deletions": [], "insertions": insertions})

    return G, temporal_changes


def load_bitcoin_dataset(
    file_path: str, batch_range: float = 1e-3, initial_fraction: float = 0.3
) -> Tuple[nx.Graph, List[Dict]]:
    """
    Load Bitcoin datasets (Alpha or OTC) and prepare them for dynamic analysis.

    Format: source,target,rating,timestamp

    Args:
        file_path: Path to soc-sign-bitcoinalpha.csv or soc-sign-bitcoinotc.csv

    Returns:
        Tuple of (initial_graph, temporal_changes)
    """
    # Read the CSV data
    df = pd.read_csv(
        file_path, header=None, names=["source", "target", "rating", "timestamp"]
    )

    # Sort by timestamp
    df = df.sort_values("timestamp")

    # Create initial graph with first 40% of edges
    split_point = len(df) // 5 * 2
    initial_data = df.iloc[:split_point]

    # Add all nodes to graph
    G = nx.Graph()
    # G.add_nodes_from(set(df["source"]).union(set(df["target"])))
    for _, row in initial_data.iterrows():
        source, target, rating = (
            int(row["source"]),
            int(row["target"]),
            float(row["rating"]),
        )
        # Use absolute rating as weight (trust/distrust both create connections)
        weight = abs(rating)

        if G.has_edge(source, target):
            G[source][target]["weight"] += weight
        else:
            G.add_edge(source, target, weight=weight)

    # Create temporal changes for remaining edges
    split_point = int(len(df) * initial_fraction)
    remaining_data = df.iloc[split_point:]
    batch_size = int(split_point * batch_range)

    temporal_changes = []
    for i in range(0, len(remaining_data), batch_size):
        batch = remaining_data.iloc[i : i + batch_size]
        insertions = []
        deletions = []

        for _, row in batch.iterrows():
            source, target, rating = (
                int(row["source"]),
                int(row["target"]),
                float(row["rating"]),
            )
            weight = abs(rating)

            # Simulate some edge deletions (negative ratings could be seen as broken trust)
            if (
                rating < 0 and np.random.random() <= 1
            ):  # 30% chance to delete on negative rating
                if (source, target) not in deletions:
                    deletions.append((source, target))
            else:
                insertions.append((source, target, weight))

        temporal_changes.append({"deletions": deletions, "insertions": insertions})

    return G, temporal_changes


def create_synthetic_dynamic_graph(
    num_nodes: int = 100, initial_edges: int = 200, time_steps: int = 10
) -> Tuple[nx.Graph, List[Dict]]:
    """
    Create a synthetic dynamic graph for testing purposes.

    Args:
        num_nodes: Number of nodes in the graph
        initial_edges: Number of initial edges
        time_steps: Number of temporal changes to generate

    Returns:
        Tuple of (initial_graph, temporal_changes)
    """
    # Create initial random graph
    G = nx.erdos_renyi_graph(
        num_nodes, initial_edges / (num_nodes * (num_nodes - 1) / 2)
    )

    # Add weights
    for edge in G.edges():
        G[edge[0]][edge[1]]["weight"] = np.random.exponential(1.0)

    # Generate temporal changes
    temporal_changes = []
    all_possible_edges = [
        (i, j) for i in range(num_nodes) for j in range(i + 1, num_nodes)
    ]
    existing_edges = list(G.edges())

    for _ in range(time_steps):
        # Random deletions (10-20% of existing edges)
        num_deletions = np.random.randint(
            len(existing_edges) // 10, len(existing_edges) // 5
        )
        deletions = np.random.choice(
            len(existing_edges),
            size=min(num_deletions, len(existing_edges)),
            replace=False,
        )
        deletion_edges = [existing_edges[i] for i in deletions]

        # Random insertions (15-25 new edges)
        possible_new_edges = [
            edge for edge in all_possible_edges if not G.has_edge(*edge)
        ]
        num_insertions = np.random.randint(15, 26)
        if len(possible_new_edges) >= num_insertions:
            insertion_indices = np.random.choice(
                len(possible_new_edges), size=num_insertions, replace=False
            )
            insertion_edges = [
                (
                    possible_new_edges[i][0],
                    possible_new_edges[i][1],
                    np.random.exponential(1.0),
                )
                for i in insertion_indices
            ]
        else:
            insertion_edges = []

        temporal_changes.append(
            {"deletions": deletion_edges, "insertions": insertion_edges}
        )

        # Update existing edges list for next iteration
        for edge in deletion_edges:
            if edge in existing_edges:
                existing_edges.remove(edge)
        for edge in insertion_edges:
            existing_edges.append((edge[0], edge[1]))

    return G, temporal_changes


def load_graph_with_sliding_window(
    file_path: str, window_size: int, step_size: int
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
        date = datetime.utcfromtimestamp(timestamp).strftime('%Y%m%d')
        if date not in edges_by_date:
            edges_by_date[date] = []
        edges_by_date[date].append((node1, node2, timestamp))
    # Return sorted by date
    return sorted(edges_by_date.items())

def load_sx_mathoverflow_sliding_window(
    file_path: str, window_size: int, step_size: int
) -> Tuple[nx.Graph, List[Dict]]:
    """
    Load MathOverflow dataset as a sequence of graphs using a sliding window over DATES (not raw timestamps).
    The initial graph is built from the first 50% of the time duration (dates), ensuring node coverage.
    Returns the initial graph and a list of temporal changes (insertions/deletions).
    """
    data = []
    with open(file_path, "r") as f:
        for line in f:
            if line.strip() and not line.startswith("//"):
                parts = line.strip().split()
                if len(parts) >= 3:
                    node1, node2, timestamp = int(parts[0]), int(parts[1]), int(parts[2])
                    data.append((node1, node2, timestamp))
    date_batches = _collect_edges_by_date(data)
    num_dates = len(date_batches)
    split_point = int(num_dates**0.75)
    # Initial graph: first 50% of dates
    initial_edges = []
    for i in range(split_point):
        initial_edges.extend(date_batches[i][1])
    G = nx.Graph()
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
    return G, temporal_changes

def load_college_msg_sliding_window(
    file_path: str, window_size: int, step_size: int
) -> Tuple[nx.Graph, List[Dict]]:
    """
    Load CollegeMsg dataset as a sequence of graphs using a sliding window over DATES (not raw timestamps).
    The initial graph is built from the first 50% of the time duration (dates), ensuring node coverage.
    Returns the initial graph and a list of temporal changes (insertions/deletions).
    """
    data = []
    with open(file_path, "r") as f:
        for line in f:
            if line.strip() and not line.startswith("//"):
                parts = line.strip().split()
                if len(parts) >= 3:
                    node1, node2, timestamp = int(parts[0]), int(parts[1]), int(parts[2])
                    data.append((node1, node2, timestamp))
    date_batches = _collect_edges_by_date(data)
    num_dates = len(date_batches)
    split_point = int(num_dates*0.85)
    # Initial graph: first 50% of dates
    initial_edges = []
    for i in range(split_point):
        initial_edges.extend(date_batches[i][1])
    G = nx.Graph()
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
    return G, temporal_changes

def load_bitcoin_sliding_window(
    file_path: str, window_size: int, step_size: int
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
    date_batches = _collect_edges_by_date(data)
    num_dates = len(date_batches)
    split_point = num_dates // 2
    # Initial graph: first 50% of dates
    initial_edges = []
    for i in range(split_point):
        initial_edges.extend(date_batches[i][1])
    G = nx.Graph()
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
    return G, temporal_changes
