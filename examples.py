"""
Example usage of the Dynamic Frontier Louvain algorithm.

This script demonstrates both synchronous and asynchronous implementations
on the provided datasets with various configurations.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add src directory to path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

from src.data_loader import (
    create_synthetic_dynamic_graph,
    load_bitcoin_dataset,
    load_college_msg_dataset,
)
from src.df_louvain import DynamicFrontierLouvain


def run_dynamic_louvain_with_edge_changes(
    df_louvain: DynamicFrontierLouvain, edge_changes_by_time: List[Dict]
) -> List[Tuple]:
    """
    Run Dynamic Frontier Louvain with edge changes over time.

    This function demonstrates how to use the DynamicFrontierLouvain class
    to track community evolution in a dynamic network with edge changes.

    Args:
        df_louvain: DynamicFrontierLouvain instance
        edge_changes_by_time: List of dicts, each with 'deletions' and 'insertions' keys

    Returns:
        List of (communities, modularity) after each time step
    """
    results = []

    print("Initial graph:")
    print(f"Nodes: {list(df_louvain.graph.nodes())}")
    print(f"Edges: {list(df_louvain.graph.edges())}")
    print(f"Initial modularity: {df_louvain.get_modularity():.4f}")

    # Initial community detection
    print("\n=== Initial Community Detection ===")
    communities = df_louvain.run_dynamic_frontier_louvain()
    print(f"Communities: {communities}")
    print(f"Modularity: {df_louvain.get_modularity():.4f}")
    results.append((communities.copy(), df_louvain.get_modularity()))

    # Apply dynamic updates over time
    for t, changes in enumerate(edge_changes_by_time):
        deletions = changes.get("deletions", [])
        insertions = changes.get("insertions", [])
        print(f"\n=== Time step {t + 1}: Applying Dynamic Updates ===")
        print(f"Deleting edges: {deletions}")
        print(f"Inserting edges: {insertions}")

        communities = df_louvain.run_dynamic_frontier_louvain(deletions, insertions)
        print(f"Updated communities: {communities}")
        print(f"Modularity: {df_louvain.get_modularity():.4f}")
        print(f"Affected vertices: {df_louvain.get_affected_nodes()}")
        results.append((communities.copy(), df_louvain.get_modularity()))

    return results


def example_college_msg():
    """Example using the College Message dataset."""
    print("=" * 60)
    print("EXAMPLE: College Message Dataset")
    print("=" * 60)

    dataset_path = current_dir / "dataset" / "CollegeMsg.txt"
    timestep = 20  # Number of time steps to simulate

    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        return

    # Load the dataset
    print("Loading College Message dataset...")
    G, temporal_changes = load_college_msg_dataset(str(dataset_path), timestep)
    print(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Temporal changes: {len(temporal_changes)} time steps")

    # Initialize DFLouvain
    df_louvain = DynamicFrontierLouvain(G, tolerance=1e-3, verbose=True)

    # Use only first few temporal changes for demo
    demo_changes = temporal_changes[:3]

    # Run the dynamic algorithm
    print("\nRunning Dynamic Frontier Louvain...")
    results = run_dynamic_louvain_with_edge_changes(df_louvain, demo_changes)

    print(f"\nFinal results: {len(results)} time steps processed")
    for i, (communities, modularity) in enumerate(results):
        print(
            f"Step {i}: Modularity = {modularity:.4f}, Communities = {len(set(communities.values()))}"
        )


def example_bitcoin():
    """Example using the Bitcoin Alpha dataset."""
    print("\n" + "=" * 60)
    print("EXAMPLE: Bitcoin Alpha Dataset")
    print("=" * 60)

    timestep = 20  # Number of time steps to simulate
    dataset_path = current_dir / "dataset" / "soc-sign-bitcoinalpha.csv"

    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        return

    # Load the dataset
    print("Loading Bitcoin Alpha dataset...")
    G, temporal_changes = load_bitcoin_dataset(str(dataset_path), timestep)
    print(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Temporal changes: {len(temporal_changes)} time steps")

    # Initialize DFLouvain
    df_louvain = DynamicFrontierLouvain(G, tolerance=1e-3, verbose=True)

    # Use only first few temporal changes for demo
    demo_changes = temporal_changes[:2]

    # Run the dynamic algorithm
    print("\nRunning Dynamic Frontier Louvain...")
    results = run_dynamic_louvain_with_edge_changes(df_louvain, demo_changes)

    print(f"\nFinal results: {len(results)} time steps processed")
    for i, (communities, modularity) in enumerate(results):
        print(
            f"Step {i}: Modularity = {modularity:.4f}, Communities = {len(set(communities.values()))}"
        )


def example_synthetic():
    """Example using synthetic data."""
    print("\n" + "=" * 60)
    print("EXAMPLE: Synthetic Dataset")
    print("=" * 60)

    # Create synthetic graph
    print("Creating synthetic dynamic graph...")
    G, temporal_changes = create_synthetic_dynamic_graph(
        num_nodes=50, initial_edges=80, time_steps=3
    )
    print(f"Created graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Temporal changes: {len(temporal_changes)} time steps")

    # Initialize DFLouvain
    df_louvain = DynamicFrontierLouvain(G, tolerance=1e-3, verbose=True)

    # Run the dynamic algorithm
    print("\nRunning Dynamic Frontier Louvain...")
    results = run_dynamic_louvain_with_edge_changes(df_louvain, temporal_changes)

    print(f"\nFinal results: {len(results)} time steps processed")
    for i, (communities, modularity) in enumerate(results):
        print(
            f"Step {i}: Modularity = {modularity:.4f}, Communities = {len(set(communities.values()))}"
        )


def main():
    """Main function to run all examples."""
    # Run examples
    example_college_msg()
    example_bitcoin()
    example_synthetic()

    print("\n" + "=" * 60)
    print("Examples complete!")
    print("Run 'python run_benchmarks.py' for comprehensive benchmarking.")
    print("=" * 60)


if __name__ == "__main__":
    main()
