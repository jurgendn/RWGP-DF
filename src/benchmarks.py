"""
Benchmarking utilities for Dynamic Frontier Louvain algorithm.
"""

import time
import warnings
from collections import defaultdict
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .data_loader import (
    create_synthetic_dynamic_graph,
    load_bitcoin_dataset,
    load_college_msg_dataset,
    load_sx_mathoverflow_dataset,
)
from .df_louvain import (
    AsyncDynamicFrontierLouvain,
    DynamicFrontierLouvain,
    GPDynamicFrontierLouvain,
)

warnings.filterwarnings("ignore")


def _safe_convert_nx_communities(nx_communities) -> List[List[int]]:
    """
    Safely convert NetworkX communities to list of lists format.

    Args:
        nx_communities: NetworkX communities object (can be various types)

    Returns:
        List of communities, where each community is a list of node IDs
    """
    communities_list = []
    try:
        # Check if it's iterable and not a string/bytes
        if hasattr(nx_communities, "__iter__") and not isinstance(
            nx_communities, (str, bytes)
        ):
            # Convert to list first to avoid type issues during iteration
            try:
                communities_iter = list(nx_communities)
            except (TypeError, ValueError):
                return []

            for community in communities_iter:
                try:
                    if hasattr(community, "__iter__") and not isinstance(
                        community, (str, bytes)
                    ):
                        # Try to convert community to list
                        try:
                            community_list = list(community)
                            communities_list.append(community_list)
                        except (TypeError, ValueError):
                            # If can't convert, wrap the community
                            communities_list.append([community])
                    else:
                        # Single node community
                        communities_list.append([community])
                except (TypeError, AttributeError, ValueError):
                    # Skip problematic communities
                    continue

        return communities_list
    except (TypeError, AttributeError, ValueError):
        # Return empty list if anything goes wrong
        return []


class DFLouvainBenchmark:
    """
    Benchmarking class for Dynamic Frontier Louvain algorithm.

    This class provides comprehensive benchmarking capabilities including:
    - Performance comparison with NetworkX Louvain
    - Modularity tracking over time
    - Runtime analysis
    - Community stability metrics
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize the benchmark.

        Args:
            verbose: Whether to print detailed progress information
        """
        self.verbose = verbose
        self.results = {}

    def benchmark_dataset(
        self,
        dataset_name: str,
        dataset_path: str,
        dataset_type: str = "college_msg",
        batch_range: float = 1e-3,
        initial_fraction: float = 0.3,
    ) -> Dict[str, Any]:
        """
        Run comprehensive benchmark on a dataset.

        Args:
            dataset_name: Name identifier for the dataset
            dataset_path: Path to the dataset file
            dataset_type: Type of dataset ('college_msg', 'bitcoin', or 'synthetic')

        Returns:
            Dictionary containing benchmark results
        """
        if self.verbose:
            print(f"\n{'=' * 60}")
            print(f"BENCHMARKING: {dataset_name}")
            print(f"{'=' * 60}")

        # Load dataset
        if dataset_type == "college_msg":
            G, temporal_changes = load_college_msg_dataset(
                dataset_path, batch_range, initial_fraction
            )
        elif dataset_type == "bitcoin":
            G, temporal_changes = load_bitcoin_dataset(
                dataset_path, batch_range, initial_fraction
            )
        elif dataset_type == "sx-mathoverflow":
            G, temporal_changes = load_sx_mathoverflow_dataset(
                dataset_path, batch_range, initial_fraction
            )
        elif dataset_type == "synthetic":
            G, temporal_changes = create_synthetic_dynamic_graph()
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

        if self.verbose:
            print(
                f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
            )
            print(f"Temporal changes: {len(temporal_changes)} time steps")

        # Run benchmarks
        results = {
            "dataset_info": {
                "name": dataset_name,
                "nodes": G.number_of_nodes(),
                "edges": G.number_of_edges(),
                "time_steps": len(temporal_changes),
            }
        }

        # 1. Static comparison benchmark
        results["static_comparison"] = self._benchmark_static_comparison(G)

        # 2. Dynamic performance benchmark
        results["dynamic_performance"] = self._benchmark_dynamic_performance(
            G, temporal_changes
        )

        # 3. Community stability analysis
        results["community_stability"] = self._analyze_community_stability(
            G, temporal_changes
        )

        # 4. Scalability analysis
        results["scalability"] = self._analyze_scalability(G, temporal_changes)

        self.results[dataset_name] = results

        if self.verbose:
            self._print_summary(results)

        return results

    def _benchmark_static_comparison(self, G: nx.Graph) -> Dict[str, Any]:
        """
        Compare DFLouvain with NetworkX Louvain on static graph.
        """
        if self.verbose:
            print("\n--- Static Algorithm Comparison ---")

        results = {}

        # NetworkX Louvain
        start_time = time.time()
        nx_communities = nx.algorithms.community.louvain_communities(G, weight="weight")
        nx_modularity = nx.algorithms.community.modularity(
            G, nx_communities, weight="weight"
        )
        nx_time = time.time() - start_time

        # Convert communities to list of lists for consistency
        nx_communities_list = _safe_convert_nx_communities(nx_communities)

        results["networkx"] = {
            "modularity": float(nx_modularity),
            "runtime": float(nx_time),
            "num_communities": len(nx_communities_list),
            "communities": nx_communities_list,
        }

        # DFLouvain
        start_time = time.time()
        df_louvain = DynamicFrontierLouvain(G, verbose=False)
        df_communities_dict = df_louvain.run_dynamic_frontier_louvain()
        df_modularity = df_louvain.get_modularity()
        df_time = time.time() - start_time

        # Convert to NetworkX format for comparison
        df_communities = defaultdict(set)
        for node, community in df_communities_dict.items():
            df_communities[community].add(node)
        df_communities = list(df_communities.values())

        results["df_louvain"] = {
            "modularity": df_modularity,
            "runtime": df_time,
            "num_communities": len(df_communities),
            "communities": df_communities,
        }

        # GPDFLouvain
        start_time = time.time()
        gp_df_louvain = GPDynamicFrontierLouvain(G, verbose=False)
        gp_df_communities_dict = gp_df_louvain.run_dynamic_frontier_louvain()
        gp_df_modularity = gp_df_louvain.get_modularity()
        gp_df_time = time.time() - start_time

        # Convert to NetworkX format for comparison
        gp_df_communities = defaultdict(set)
        for node, community in gp_df_communities_dict.items():
            gp_df_communities[community].add(node)
        gp_df_communities = list(gp_df_communities.values())

        results["gp_df_louvain"] = {
            "modularity": gp_df_modularity,
            "runtime": gp_df_time,
            "num_communities": len(gp_df_communities),
            "communities": gp_df_communities,
        }

        # Comparison metrics
        results["comparison"] = {
            "gp_df_modularity_diff": abs(nx_modularity - gp_df_modularity),
            "df_modularity_diff": abs(nx_modularity - df_modularity),
            "modularity_diff": abs(nx_modularity - gp_df_modularity),
            "runtime_ratio": gp_df_time / nx_time if nx_time > 0 else float("inf"),
            "community_count_diff": abs(len(nx_communities_list) - len(df_communities)),
        }

        if self.verbose:
            print(
                f"NetworkX: Modularity={nx_modularity:.4f}, Runtime={nx_time:.3f}s, Communities={len(nx_communities_list)}"
            )
            print(
                f"DFLouvain: Modularity={df_modularity:.4f}, Runtime={df_time:.3f}s, Communities={len(df_communities)}"
            )
            print(
                f"GPDFLouvain: Modularity={gp_df_modularity:.4f}, Runtime={gp_df_time:.3f}s, Communities={len(gp_df_communities)}"
            )
            print(
                f"Modularity difference: {results['comparison']['modularity_diff']:.4f}"
            )
            print(
                f"Runtime ratio (DF/NX): {results['comparison']['runtime_ratio']:.2f}"
            )

        return results

    def _benchmark_dynamic_performance(
        self, G: nx.Graph, temporal_changes: List[Dict]
    ) -> Dict[str, Any]:
        """
        Benchmark dynamic performance of DFLouvain vs NetworkX Louvain.
        """
        if self.verbose:
            print("\n--- Dynamic Performance Benchmark ---")

        results = {
            "df_runtimes": [],
            "df_modularities": [],
            "gp_df_runtimes": [],
            "gp_df_modularities": [],
            "nx_runtimes": [],
            "nx_modularities": [],
            "df_affected_nodes": [],
            "gp_df_affected_nodes": [],
            "total_runtime": 0,
            "iterations_per_step": [],
        }

        # Initialize DFLouvain
        df_louvain = DynamicFrontierLouvain(G, verbose=False)
        gp_df_louvain = GPDynamicFrontierLouvain(G, verbose=False)

        # Initial state - both algorithms
        df_start_time = time.time()
        df_louvain.run_dynamic_frontier_louvain()
        df_initial_time = time.time() - df_start_time
        initial_df_modularity = df_louvain.get_modularity()
        results["df_modularities"].append(initial_df_modularity)
        results["df_runtimes"].append(df_initial_time)
        
        # Initial state - GP DFLouvain
        gp_df_start_time = time.time()
        gp_df_louvain.run_dynamic_frontier_louvain()
        gp_df_initial_time = time.time() - gp_df_start_time
        initial_gp_df_modularity = gp_df_louvain.get_modularity()
        results["gp_df_modularities"].append(initial_gp_df_modularity)
        results["gp_df_runtimes"].append(gp_df_initial_time)

        # Initial NetworkX Louvain
        nx_start_time = time.time()
        nx_communities = nx.algorithms.community.louvain_communities(G, weight="weight")
        initial_nx_modularity = nx.algorithms.community.modularity(
            G, nx_communities, weight="weight"
        )
        nx_initial_time = time.time() - nx_start_time
        results["nx_modularities"].append(initial_nx_modularity)
        results["nx_runtimes"].append(nx_initial_time)

        # Keep track of current graph for NetworkX
        current_graph = G.copy()

        total_start_time = time.time()

        # Process temporal changes
        progress_bar = tqdm(
            enumerate(temporal_changes[:100]),
            desc="Processing Temporal Changes",
            total=len(temporal_changes[:100]),
        )
        for i, changes in progress_bar:
            progress_bar.set_description(f"Step {i + 1}/{len(temporal_changes)}")
            deletions = changes.get("deletions", [])
            insertions = changes.get("insertions", [])
            progress_bar.set_description(
                f"Step {i + 1}/{len(temporal_changes)} - Get changes"
            )

            # DFLouvain dynamic update
            df_step_start_time = time.time()
            df_louvain.run_dynamic_frontier_louvain(deletions, insertions)
            df_modularity = df_louvain.get_modularity()
            df_affected_nodes = len(df_louvain.get_affected_nodes())
            df_step_runtime = time.time() - df_step_start_time
            
            # GPDFLouvain dynamic update
            gp_df_step_start_time = time.time()
            gp_df_louvain.run_dynamic_frontier_louvain(deletions, insertions)
            gp_df_modularity = gp_df_louvain.get_modularity()
            gp_df_affected_nodes = len(gp_df_louvain.get_affected_nodes())
            gp_df_step_runtime = time.time() - gp_df_step_start_time

            # Update current graph for NetworkX
            for edge in deletions:
                if current_graph.has_edge(*edge):
                    current_graph.remove_edge(*edge)
            for edge in insertions:
                if len(edge) == 3:  # (u, v, weight)
                    current_graph.add_edge(edge[0], edge[1], weight=edge[2])
                else:  # (u, v)
                    current_graph.add_edge(edge[0], edge[1], weight=1.0)

            # NetworkX Louvain from scratch
            nx_step_start_time = time.time()
            nx_communities = nx.algorithms.community.louvain_communities(
                current_graph, weight="weight"
            )
            nx_modularity = nx.algorithms.community.modularity(
                current_graph, nx_communities, weight="weight"
            )
            nx_step_runtime = time.time() - nx_step_start_time

            # Store results
            results["df_runtimes"].append(df_step_runtime)
            results["gp_df_runtimes"].append(gp_df_step_runtime)
            results["df_modularities"].append(df_modularity)
            results["gp_df_modularities"].append(gp_df_modularity)
            results["nx_runtimes"].append(nx_step_runtime)
            results["nx_modularities"].append(nx_modularity)
            results["df_affected_nodes"].append(df_affected_nodes)
            results["gp_df_affected_nodes"].append(gp_df_affected_nodes)

            progress_bar.set_postfix(
                {
                    "DF_Q": f"{df_modularity:.4f}",
                    "GPDF_Q": f"{gp_df_modularity:.4f}",
                    "NX_Q": f"{nx_modularity:.4f}",
                    "DF Affected": df_affected_nodes,
                    "GPDF Affected": gp_df_affected_nodes,
                    "DF_Runtime": f"{df_step_runtime:.3f}s",
                    "GPDF_Runtime": f"{gp_df_step_runtime:.3f}s",
                    "NX_Runtime": f"{nx_step_runtime:.3f}s",
                }
            )
            # if self.verbose and i < 5:  # Show first few steps
            #     print(
            #         f"Step {i + 1}: DF_Q={df_modularity:.4f} ({df_step_runtime:.3f}s), NX_Q={nx_modularity:.4f} ({nx_step_runtime:.3f}s), Affected={affected_nodes}"
            #     )

        results["total_runtime"] = time.time() - total_start_time
        results["avg_df_runtime"] = np.mean(results["df_runtimes"])
        results["avg_gp_df_runtime"] = np.mean(results["gp_df_runtimes"])
        results["avg_nx_runtime"] = np.mean(results["nx_runtimes"])
        results["df_modularity_stability"] = np.std(results["df_modularities"])
        results["gp_df_modularity_stability"] = np.std(results["gp_df_modularities"])
        results["nx_modularity_stability"] = np.std(results["nx_modularities"])

        if self.verbose:
            print(f"Total dynamic runtime: {results['total_runtime']:.3f}s")
            print(f"Average DF step runtime: {results['avg_df_runtime']:.3f}s")
            print(f"Average GPDF step runtime: {results['avg_gp_df_runtime']:.3f}s")
            print(f"Average NX step runtime: {results['avg_nx_runtime']:.3f}s")
            print(
                f"DF Modularity range: [{min(results['df_modularities']):.4f}, {max(results['df_modularities']):.4f}]"
            )
            print(
                f"GP-DF Modularity range: [{min(results['gp_df_modularities']):.4f}, {max(results['gp_df_modularities']):.4f}]"
            )
            print(
                f"NX Modularity range: [{min(results['nx_modularities']):.4f}, {max(results['nx_modularities']):.4f}]"
            )

        return results

    def _analyze_community_stability(
        self, G: nx.Graph, temporal_changes: List[Dict]
    ) -> Dict[str, Any]:
        """
        Analyze community stability over time.
        """
        if self.verbose:
            print("\n--- Community Stability Analysis ---")

        results = {
            "community_evolution": [],
            "stability_metrics": {},
            "node_community_changes": defaultdict(int),
        }

        df_louvain = DynamicFrontierLouvain(G, verbose=False)

        # Track community assignments over time
        community_history = []
        communities = df_louvain.run_dynamic_frontier_louvain()
        community_history.append(communities.copy())

        progress_bar = tqdm(temporal_changes, desc="Analyzing Community Stability")
        for changes in progress_bar:
            progress_bar.set_postfix_str(
                f"Processing changes: {len(changes.get('deletions', []))} deletions, {len(changes.get('insertions', []))} insertions"
            )
            deletions = changes.get("deletions", [])
            insertions = changes.get("insertions", [])

            communities = df_louvain.run_dynamic_frontier_louvain(deletions, insertions)
            community_history.append(communities.copy())

        # Calculate stability metrics
        total_changes = 0
        total_comparisons = 0

        for i in range(1, len(community_history)):
            prev_communities = community_history[i - 1]
            curr_communities = community_history[i]

            # Count nodes that changed communities
            changes_this_step = 0
            for node in prev_communities:
                if node in curr_communities:
                    if prev_communities[node] != curr_communities[node]:
                        changes_this_step += 1
                        results["node_community_changes"][node] += 1

            total_changes += changes_this_step
            total_comparisons += len(prev_communities)

            results["community_evolution"].append(
                {
                    "step": i,
                    "nodes_changed": changes_this_step,
                    "change_rate": changes_this_step / len(prev_communities)
                    if len(prev_communities) > 0
                    else 0,
                }
            )

        results["stability_metrics"] = {
            "overall_change_rate": total_changes / total_comparisons
            if total_comparisons > 0
            else 0,
            "most_unstable_nodes": sorted(
                results["node_community_changes"].items(),
                key=lambda x: x[1],
                reverse=True,
            )[:10],
            "avg_changes_per_step": total_changes / len(temporal_changes)
            if len(temporal_changes) > 0
            else 0,
        }

        if self.verbose:
            print(
                f"Overall community change rate: {results['stability_metrics']['overall_change_rate']:.3f}"
            )
            print(
                f"Average changes per step: {results['stability_metrics']['avg_changes_per_step']:.1f}"
            )

        return results

    def _analyze_scalability(
        self, G: nx.Graph, temporal_changes: List[Dict]
    ) -> Dict[str, Any]:
        """
        Analyze scalability with respect to graph size and changes.
        """
        if self.verbose:
            print("\n--- Scalability Analysis ---")

        results = {
            "runtime_vs_graph_size": [],
            "runtime_vs_changes": [],
            "memory_usage": {},
        }

        # Test different graph sizes (if graph is large enough)
        original_nodes = list(G.nodes())
        if len(original_nodes) > 100:
            test_sizes = [50, 100, 200, min(500, len(original_nodes))]

            for size in test_sizes:
                if size <= len(original_nodes):
                    # Create subgraph
                    sampled_nodes = np.random.choice(
                        original_nodes, size, replace=False
                    )
                    subgraph = G.subgraph(sampled_nodes).copy()

                    # Time the algorithm
                    start_time = time.time()
                    df_louvain = DynamicFrontierLouvain(subgraph, verbose=False)
                    df_louvain.run_dynamic_frontier_louvain()
                    runtime = time.time() - start_time

                    results["runtime_vs_graph_size"].append(
                        {
                            "nodes": size,
                            "edges": subgraph.number_of_edges(),
                            "runtime": runtime,
                        }
                    )

        # Test runtime vs number of changes
        df_louvain = DynamicFrontierLouvain(G, verbose=False)
        df_louvain.run_dynamic_frontier_louvain()  # Initial

        for i, changes in enumerate(temporal_changes[:5]):  # Test first 5 steps
            deletions = changes.get("deletions", [])
            insertions = changes.get("insertions", [])

            num_changes = len(deletions) + len(insertions)

            start_time = time.time()
            df_louvain.run_dynamic_frontier_louvain(deletions, insertions)
            runtime = time.time() - start_time

            results["runtime_vs_changes"].append(
                {"step": i + 1, "num_changes": num_changes, "runtime": runtime}
            )

        if self.verbose:
            if results["runtime_vs_graph_size"]:
                print("Runtime vs Graph Size:")
                for entry in results["runtime_vs_graph_size"]:
                    print(f"  {entry['nodes']} nodes: {entry['runtime']:.3f}s")

        return results

    def _print_summary(self, results: Dict[str, Any]) -> None:
        """Print a summary of benchmark results."""
        print(f"\n{'=' * 40}")
        print("BENCHMARK SUMMARY")
        print(f"{'=' * 40}")

        # Dataset info
        info = results["dataset_info"]
        print(f"Dataset: {info['name']}")
        print(f"Nodes: {info['nodes']:,}, Edges: {info['edges']:,}")
        print(f"Time steps: {info['time_steps']}")

        # Static comparison
        static = results["static_comparison"]
        print("\nStatic Performance:")
        print(f"  DFLouvain Modularity: {static['df_louvain']['modularity']:.4f}")
        print(f"  GPDFLouvain Modularity: {static['gp_df_louvain']['modularity']:.4f}")
        print(f"  NetworkX Modularity: {static['networkx']['modularity']:.4f}")
        print(f"  Runtime Ratio (DF/NX): {static['comparison']['runtime_ratio']:.2f}")

        # Dynamic performance
        dynamic = results["dynamic_performance"]
        print("\nDynamic Performance:")
        print(f"  Total Runtime: {dynamic['total_runtime']:.3f}s")
        print(f"  Avg DF Step Runtime: {dynamic['avg_df_runtime']:.3f}s")
        print(f"  Avg GPDF Step Runtime: {dynamic['avg_gp_df_runtime']:.3f}s")
        print(f"  Avg NX Step Runtime: {dynamic['avg_nx_runtime']:.3f}s")
        print(f"  DF Modularity Std: {dynamic['df_modularity_stability']:.4f}")
        print(f"  NX Modularity Std: {dynamic['nx_modularity_stability']:.4f}")

        # Community stability
        stability = results["community_stability"]
        print("\nCommunity Stability:")
        print(
            f"  Change Rate: {stability['stability_metrics']['overall_change_rate']:.3f}"
        )
        print(
            f"  Avg Changes/Step: {stability['stability_metrics']['avg_changes_per_step']:.1f}"
        )

    def plot_results(self, dataset_name: str, save_path: Optional[str] = None) -> None:
        """
        Plot comprehensive benchmark results with DF vs NX comparisons.

        Args:
            dataset_name: Name of the dataset to plot
            save_path: Optional path to save the plot
        """
        if dataset_name not in self.results:
            print(f"No results found for dataset: {dataset_name}")
            return

        results = self.results[dataset_name]

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f"DFLouvain vs NetworkX Louvain: {dataset_name}", fontsize=16)

        dynamic = results["dynamic_performance"]

        # 1. Modularity comparison over time (line plot)
        time_steps = range(len(dynamic["df_modularities"]))
        axes[0, 0].plot(
            time_steps,
            dynamic["df_modularities"],
            "blue",
            linewidth=1,
            alpha=0.5,
            label="DF Louvain",
            # marker="o",
        )
        axes[0, 0].plot(
            time_steps,
            dynamic["gp_df_modularities"],
            "teal",
            linewidth=1,
            alpha=0.5,
            label="GP-DF Louvain",
            # marker="o",
        )
        axes[0, 0].plot(
            time_steps,
            dynamic["nx_modularities"],
            "red",
            linewidth=1,
            alpha=0.5,
            label="NetworkX Louvain",
            # marker="s",
        )
        axes[0, 0].set_title("Modularity Over Time")
        axes[0, 0].set_xlabel("Time Step")
        axes[0, 0].set_ylabel("Modularity")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Runtime comparison (bar chart)
        x_pos = np.arange(len(dynamic["df_runtimes"]))

        axes[0, 1].plot(
            range(len(dynamic["df_runtimes"])),
            dynamic["df_runtimes"],
            "blue",
            linewidth=1,
            label="DF Louvain",
            alpha=0.5,
            # marker="o",
        )
        axes[0, 1].plot(
            range(len(dynamic["gp_df_runtimes"])),
            dynamic["gp_df_runtimes"],
            "teal",
            linewidth=1,
            label="GP-DF Louvain",
            alpha=0.5,
            # marker="o",
        )
        axes[0, 1].plot(
            range(len(dynamic["nx_runtimes"])),
            dynamic["nx_runtimes"],
            "red",
            linewidth=1,
            alpha=0.5,
            label="NetworkX Louvain",
            # marker="s",
        )
        axes[0, 1].set_title("Runtime Comparison Per Step")
        axes[0, 1].set_xlabel("Time Step")
        axes[0, 1].set_ylabel("Runtime (seconds)")
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels(
            [f"T{i + 1}" for i in range(len(dynamic["df_runtimes"]))]
        )
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Affected nodes over time
        axes[1, 0].plot(
            range(len(dynamic["df_affected_nodes"])),
            dynamic["df_affected_nodes"],
            "blue",
            linewidth=1,
            # marker="d",
        )
        axes[1, 0].plot(
            range(len(dynamic["gp_df_affected_nodes"])),
            dynamic["gp_df_affected_nodes"],
            "teal",
            linewidth=1,
            # marker="d",
        )
        axes[1, 0].set_title("Affected Nodes Per Step (DF Louvain vs GP-DF Louvain)")
        axes[1, 0].set_xlabel("Time Step")
        axes[1, 0].set_ylabel("Number of Affected Nodes")
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Average runtime comparison (bar chart)
        algorithms = ["DF Louvain", "GP-DF Louvain", "NetworkX Louvain"]
        avg_runtimes = [dynamic["avg_df_runtime"], dynamic["avg_gp_df_runtime"], dynamic["avg_nx_runtime"]]
        colors = ["blue", "teal", "red"]

        bars = axes[1, 1].bar(algorithms, avg_runtimes, color=colors, alpha=0.7)
        axes[1, 1].set_title("Average Runtime Comparison")
        axes[1, 1].set_ylabel("Average Runtime (seconds)")

        # Add value labels on bars
        for bar, runtime in zip(bars, avg_runtimes):
            height = bar.get_height()
            axes[1, 1].text(
                bar.get_x() + bar.get_width() / 2.0,
                height + height * 0.01,
                f"{runtime:.3f}s",
                ha="center",
                va="bottom",
            )

        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to: {save_path}")
        else:
            plt.show()

    def export_results(self, filepath: str) -> None:
        """
        Export benchmark results to CSV for further analysis.

        Args:
            filepath: Path to save the CSV file
        """
        all_data = []

        for dataset_name, results in self.results.items():
            # Dataset info
            info = results["dataset_info"]

            # Static comparison
            static = results["static_comparison"]

            # Dynamic performance summary
            dynamic = results["dynamic_performance"]

            # Stability summary
            stability = results["community_stability"]

            row = {
                "dataset": dataset_name,
                "nodes": info["nodes"],
                "edges": info["edges"],
                "time_steps": info["time_steps"],
                "df_modularity": static["df_louvain"]["modularity"],
                "nx_modularity": static["networkx"]["modularity"],
                "modularity_diff": static["comparison"]["modularity_diff"],
                "runtime_ratio": static["comparison"]["runtime_ratio"],
                "total_dynamic_runtime": dynamic["total_runtime"],
                "avg_df_step_runtime": dynamic["avg_df_runtime"],
                "avg_nx_step_runtime": dynamic["avg_nx_runtime"],
                "df_modularity_stability": dynamic["df_modularity_stability"],
                "nx_modularity_stability": dynamic["nx_modularity_stability"],
                "community_change_rate": stability["stability_metrics"][
                    "overall_change_rate"
                ],
                "avg_changes_per_step": stability["stability_metrics"][
                    "avg_changes_per_step"
                ],
            }

            all_data.append(row)

        df = pd.DataFrame(all_data)
        df.to_csv(filepath, index=False)
        print(f"Results exported to: {filepath}")

    async def benchmark_dataset_async(
        self,
        dataset_name: str,
        dataset_path: str,
        dataset_type: str = "college_msg",
        batch_range: float = 1e-3,
    ) -> Dict[str, Any]:
        """
        Run comprehensive async benchmark on a dataset.

        Args:
            dataset_name: Name identifier for the dataset
            dataset_path: Path to the dataset file
            dataset_type: Type of dataset ('college_msg', 'bitcoin', or 'synthetic')

        Returns:
            Dictionary containing benchmark results
        """
        if self.verbose:
            print(f"\n{'=' * 60}")
            print(f"ASYNC BENCHMARKING: {dataset_name}")
            print(f"{'=' * 60}")

        # Load dataset
        if dataset_type == "college_msg":
            G, temporal_changes = load_college_msg_dataset(dataset_path, batch_range)
        elif dataset_type == "bitcoin":
            G, temporal_changes = load_bitcoin_dataset(dataset_path, batch_range)
        elif dataset_type == "synthetic":
            G, temporal_changes = create_synthetic_dynamic_graph()
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

        if self.verbose:
            print(
                f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
            )
            print(f"Temporal changes: {len(temporal_changes)} time steps")

        # Run async benchmarks
        results = {
            "dataset_info": {
                "name": dataset_name + "_async",
                "nodes": G.number_of_nodes(),
                "edges": G.number_of_edges(),
                "time_steps": len(temporal_changes),
            }
        }

        # 1. Async vs sync comparison
        results[
            "async_vs_sync_comparison"
        ] = await self._benchmark_async_vs_sync_comparison(G)

        # 2. Async dynamic performance
        results[
            "async_dynamic_performance"
        ] = await self._benchmark_async_dynamic_performance(G, temporal_changes)

        # 3. Async scalability analysis
        results["async_scalability"] = await self._analyze_async_scalability(
            G, temporal_changes
        )

        # 4. Parallel processing efficiency
        results["parallel_efficiency"] = await self._analyze_parallel_efficiency(
            G, temporal_changes
        )

        self.results[dataset_name + "_async"] = results

        if self.verbose:
            self._print_async_summary(results)

        return results

    async def _benchmark_async_vs_sync_comparison(self, G: nx.Graph) -> Dict[str, Any]:
        """
        Compare async DFLouvain with sync DFLouvain on static graph.
        """
        if self.verbose:
            print("\n--- Async vs Sync Algorithm Comparison ---")

        results = {}

        # Synchronous DFLouvain
        start_time = time.time()
        sync_louvain = DynamicFrontierLouvain(G, verbose=False)
        sync_communities_dict = sync_louvain.run_dynamic_frontier_louvain()
        sync_modularity = sync_louvain.get_modularity()
        sync_time = time.time() - start_time

        # Convert to list format
        sync_communities = defaultdict(set)
        for node, community in sync_communities_dict.items():
            sync_communities[community].add(node)
        sync_communities = list(sync_communities.values())

        results["sync_df"] = {
            "modularity": sync_modularity,
            "runtime": sync_time,
            "num_communities": len(sync_communities),
            "communities": sync_communities,
        }

        # Asynchronous DFLouvain
        start_time = time.time()
        async_louvain = AsyncDynamicFrontierLouvain(verbose=False)
        async_community_info = await async_louvain.dynamic_frontier_louvain(G)
        async_modularity = await async_louvain.get_modularity_async(
            G, async_community_info.community_assignments
        )
        async_time = time.time() - start_time

        # Convert to list format
        async_communities = defaultdict(set)
        for node, community in async_community_info.community_assignments.items():
            async_communities[community].add(node)
        async_communities = list(async_communities.values())

        results["async_df"] = {
            "modularity": async_modularity,
            "runtime": async_time,
            "num_communities": len(async_communities),
            "communities": async_communities,
        }

        # Comparison metrics
        results["comparison"] = {
            "modularity_diff": abs(sync_modularity - async_modularity),
            "runtime_ratio": async_time / sync_time if sync_time > 0 else float("inf"),
            "community_count_diff": abs(len(sync_communities) - len(async_communities)),
            "speedup": sync_time / async_time if async_time > 0 else float("inf"),
        }

        if self.verbose:
            print(
                f"Sync DF: Modularity={sync_modularity:.4f}, Runtime={sync_time:.3f}s, Communities={len(sync_communities)}"
            )
            print(
                f"Async DF: Modularity={async_modularity:.4f}, Runtime={async_time:.3f}s, Communities={len(async_communities)}"
            )
            print(
                f"Modularity difference: {results['comparison']['modularity_diff']:.4f}"
            )
            print(f"Speedup (Sync/Async): {results['comparison']['speedup']:.2f}x")

        return results

    async def _benchmark_async_dynamic_performance(
        self, G: nx.Graph, temporal_changes: List[Dict]
    ) -> Dict[str, Any]:
        """
        Benchmark async dynamic performance vs sync and NetworkX.
        """
        if self.verbose:
            print("\n--- Async Dynamic Performance Benchmark ---")

        results = {
            "async_df_runtimes": [],
            "async_df_modularities": [],
            "sync_df_runtimes": [],
            "sync_df_modularities": [],
            "nx_runtimes": [],
            "nx_modularities": [],
            "parallel_efficiency": [],
            "total_runtime": 0,
        }

        # Initialize algorithms
        async_louvain = AsyncDynamicFrontierLouvain(verbose=True)
        sync_louvain = DynamicFrontierLouvain(G, verbose=False)

        # Initial state - all algorithms
        async_start_time = time.time()
        async_community_info = await async_louvain.dynamic_frontier_louvain(G)
        initial_async_modularity = await async_louvain.get_modularity_async(
            G, async_community_info.community_assignments
        )
        async_initial_time = time.time() - async_start_time
        results["async_df_modularities"].append(initial_async_modularity)
        results["async_df_runtimes"].append(async_initial_time)

        sync_start_time = time.time()
        sync_louvain.run_dynamic_frontier_louvain()
        initial_sync_modularity = sync_louvain.get_modularity()
        sync_initial_time = time.time() - sync_start_time
        results["sync_df_modularities"].append(initial_sync_modularity)
        results["sync_df_runtimes"].append(sync_initial_time)

        # Initial NetworkX
        nx_start_time = time.time()
        nx_communities = nx.algorithms.community.louvain_communities(G, weight="weight")
        initial_nx_modularity = nx.algorithms.community.modularity(
            G, nx_communities, weight="weight"
        )
        nx_initial_time = time.time() - nx_start_time
        results["nx_modularities"].append(initial_nx_modularity)
        results["nx_runtimes"].append(nx_initial_time)

        # Keep track of current graph for NetworkX
        current_graph = G.copy()
        previous_async_info = async_community_info

        total_start_time = time.time()

        # Process temporal changes
        for i, changes in enumerate(temporal_changes):
            deletions = changes.get("deletions", [])
            insertions = changes.get("insertions", [])

            # Convert insertions to proper format for async
            async_insertions = [
                (u, v, w) if len(edge) == 3 else (u, v, 1.0)
                for edge in insertions
                for u, v, *rest in [edge]
                for w in [rest[0] if rest else 1.0]
            ]

            # Async DFLouvain dynamic update
            async_step_start_time = time.time()
            async_community_info = await async_louvain.dynamic_frontier_louvain(
                current_graph,
                edge_deletions=deletions,
                edge_insertions=async_insertions,
                previous_communities=previous_async_info.community_assignments,
                previous_vertex_degrees=previous_async_info.vertex_degrees,
                previous_community_weights=previous_async_info.community_weights,
            )
            async_modularity = await async_louvain.get_modularity_async(
                current_graph, async_community_info.community_assignments
            )
            async_step_runtime = time.time() - async_step_start_time

            # Sync DFLouvain dynamic update
            sync_step_start_time = time.time()
            sync_louvain.run_dynamic_frontier_louvain(deletions, insertions)
            sync_modularity = sync_louvain.get_modularity()
            sync_step_runtime = time.time() - sync_step_start_time

            # Update current graph for NetworkX
            for edge in deletions:
                if current_graph.has_edge(*edge):
                    current_graph.remove_edge(*edge)
            for edge in insertions:
                if len(edge) == 3:  # (u, v, weight)
                    current_graph.add_edge(edge[0], edge[1], weight=edge[2])
                else:  # (u, v)
                    current_graph.add_edge(edge[0], edge[1], weight=1.0)

            # NetworkX Louvain from scratch
            nx_step_start_time = time.time()
            nx_communities = nx.algorithms.community.louvain_communities(
                current_graph, weight="weight"
            )
            nx_modularity = nx.algorithms.community.modularity(
                current_graph, nx_communities, weight="weight"
            )
            nx_step_runtime = time.time() - nx_step_start_time

            # Calculate parallel efficiency
            parallel_efficiency = (
                sync_step_runtime / async_step_runtime
                if async_step_runtime > 0
                else 1.0
            )

            # Store results
            results["async_df_runtimes"].append(async_step_runtime)
            results["async_df_modularities"].append(async_modularity)
            results["sync_df_runtimes"].append(sync_step_runtime)
            results["sync_df_modularities"].append(sync_modularity)
            results["nx_runtimes"].append(nx_step_runtime)
            results["nx_modularities"].append(nx_modularity)
            results["parallel_efficiency"].append(parallel_efficiency)

            # Update for next iteration
            previous_async_info = async_community_info

            if self.verbose and i < 5:  # Show first few steps
                print(
                    f"Step {i + 1}: Async_Q={async_modularity:.4f} ({async_step_runtime:.3f}s), "
                    f"Sync_Q={sync_modularity:.4f} ({sync_step_runtime:.3f}s), "
                    f"NX_Q={nx_modularity:.4f} ({nx_step_runtime:.3f}s), "
                    f"Efficiency={parallel_efficiency:.2f}x"
                )

        results["total_runtime"] = time.time() - total_start_time
        results["avg_async_runtime"] = np.mean(results["async_df_runtimes"])
        results["avg_sync_runtime"] = np.mean(results["sync_df_runtimes"])
        results["avg_nx_runtime"] = np.mean(results["nx_runtimes"])
        results["avg_parallel_efficiency"] = np.mean(results["parallel_efficiency"])
        results["async_modularity_stability"] = np.std(results["async_df_modularities"])
        results["sync_modularity_stability"] = np.std(results["sync_df_modularities"])

        if self.verbose:
            print(f"Total async dynamic runtime: {results['total_runtime']:.3f}s")
            print(f"Average Async DF step runtime: {results['avg_async_runtime']:.3f}s")
            print(f"Average Sync DF step runtime: {results['avg_sync_runtime']:.3f}s")
            print(f"Average NX step runtime: {results['avg_nx_runtime']:.3f}s")
            print(
                f"Average parallel efficiency: {results['avg_parallel_efficiency']:.2f}x"
            )

        return results

    async def _analyze_async_scalability(
        self, G: nx.Graph, temporal_changes: List[Dict]
    ) -> Dict[str, Any]:
        """
        Analyze async scalability with different graph sizes and parallelization levels.
        """
        if self.verbose:
            print("\n--- Async Scalability Analysis ---")

        results = {
            "runtime_vs_graph_size": [],
            "runtime_vs_parallelization": [],
            "memory_usage": {},
        }

        # Test different graph sizes
        original_nodes = list(G.nodes())
        if len(original_nodes) > 100:
            test_sizes = [50, 100, 200, min(500, len(original_nodes))]

            for size in test_sizes:
                if size <= len(original_nodes):
                    # Create subgraph
                    sampled_nodes = np.random.choice(
                        original_nodes, size, replace=False
                    )
                    subgraph = G.subgraph(sampled_nodes).copy()

                    # Time the async algorithm
                    start_time = time.time()
                    async_louvain = AsyncDynamicFrontierLouvain(verbose=False)
                    await async_louvain.dynamic_frontier_louvain(subgraph)
                    runtime = time.time() - start_time

                    results["runtime_vs_graph_size"].append(
                        {
                            "nodes": size,
                            "edges": subgraph.number_of_edges(),
                            "runtime": runtime,
                        }
                    )

        if self.verbose and results["runtime_vs_graph_size"]:
            print("Async Runtime vs Graph Size:")
            for entry in results["runtime_vs_graph_size"]:
                print(f"  {entry['nodes']} nodes: {entry['runtime']:.3f}s")

        return results

    async def _analyze_parallel_efficiency(
        self, G: nx.Graph, temporal_changes: List[Dict]
    ) -> Dict[str, Any]:
        """
        Analyze parallel processing efficiency of async implementation.
        """
        if self.verbose:
            print("\n--- Parallel Processing Efficiency Analysis ---")

        results = {
            "chunk_size_performance": [],
            "concurrency_levels": [],
            "theoretical_vs_actual_speedup": {},
        }

        # Test with first few changes
        async_louvain = AsyncDynamicFrontierLouvain(verbose=False)

        # Initialize
        await async_louvain.dynamic_frontier_louvain(G)

        for i, changes in enumerate(temporal_changes[:3]):
            deletions = changes.get("deletions", [])
            insertions = changes.get("insertions", [])

            # Convert insertions format
            async_insertions = [
                (u, v, w) if len(edge) == 3 else (u, v, 1.0)
                for edge in insertions
                for u, v, *rest in [edge]
                for w in [rest[0] if rest else 1.0]
            ]

            start_time = time.time()
            await async_louvain.dynamic_frontier_louvain(
                G, edge_deletions=deletions, edge_insertions=async_insertions
            )
            runtime = time.time() - start_time

            results["concurrency_levels"].append(
                {
                    "step": i + 1,
                    "num_changes": len(deletions) + len(insertions),
                    "runtime": runtime,
                    "estimated_parallelism": min(
                        len(G.nodes()) // 100, 10
                    ),  # Rough estimate
                }
            )

        if self.verbose:
            print("Parallel processing efficiency measured across temporal changes")

        return results

    def _print_async_summary(self, results: Dict[str, Any]) -> None:
        """Print a summary of async benchmark results."""
        print(f"\n{'=' * 40}")
        print("ASYNC BENCHMARK SUMMARY")
        print(f"{'=' * 40}")

        # Dataset info
        info = results["dataset_info"]
        print(f"Dataset: {info['name']}")
        print(f"Nodes: {info['nodes']:,}, Edges: {info['edges']:,}")
        print(f"Time steps: {info['time_steps']}")

        # Async vs sync comparison
        async_sync = results["async_vs_sync_comparison"]
        print("\nAsync vs Sync Performance:")
        print(f"  Async Modularity: {async_sync['async_df']['modularity']:.4f}")
        print(f"  Sync Modularity: {async_sync['sync_df']['modularity']:.4f}")
        print(f"  Speedup (Sync/Async): {async_sync['comparison']['speedup']:.2f}x")

        # Dynamic performance
        dynamic = results["async_dynamic_performance"]
        print("\nDynamic Performance:")
        print(f"  Total Runtime: {dynamic['total_runtime']:.3f}s")
        print(f"  Avg Async Step Runtime: {dynamic['avg_async_runtime']:.3f}s")
        print(f"  Avg Sync Step Runtime: {dynamic['avg_sync_runtime']:.3f}s")
        print(f"  Avg Parallel Efficiency: {dynamic['avg_parallel_efficiency']:.2f}x")
        print(f"  Async Modularity Std: {dynamic['async_modularity_stability']:.4f}")
        print(f"  Sync Modularity Std: {dynamic['sync_modularity_stability']:.4f}")

    def plot_async_results(
        self, dataset_name: str, save_path: Optional[str] = None
    ) -> None:
        """
        Plot comprehensive async benchmark results.

        Args:
            dataset_name: Name of the dataset to plot (with '_async' suffix)
            save_path: Optional path to save the plot
        """
        if dataset_name not in self.results:
            print(f"No results found for dataset: {dataset_name}")
            return

        results = self.results[dataset_name]

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(
            f"Async DFLouvain Performance Analysis: {dataset_name}", fontsize=16
        )

        dynamic = results["async_dynamic_performance"]

        # 1. Modularity comparison over time (Async vs Sync vs NX)
        time_steps = range(len(dynamic["async_df_modularities"]))
        axes[0, 0].plot(
            time_steps,
            dynamic["async_df_modularities"],
            "g-",
            linewidth=2,
            label="Async DF",
            marker="o",
        )
        axes[0, 0].plot(
            time_steps,
            dynamic["sync_df_modularities"],
            "b-",
            linewidth=2,
            label="Sync DF",
            marker="s",
        )
        axes[0, 0].plot(
            time_steps,
            dynamic["nx_modularities"],
            "r-",
            linewidth=2,
            label="NetworkX",
            marker="^",
        )
        axes[0, 0].set_title("Modularity Over Time")
        axes[0, 0].set_xlabel("Time Step")
        axes[0, 0].set_ylabel("Modularity")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Runtime comparison (3-way)
        x_pos = np.arange(len(dynamic["async_df_runtimes"]))
        width = 0.25
        axes[0, 1].bar(
            x_pos - width,
            dynamic["async_df_runtimes"],
            width,
            label="Async DF",
            color="green",
            alpha=0.7,
        )
        axes[0, 1].bar(
            x_pos,
            dynamic["sync_df_runtimes"],
            width,
            label="Sync DF",
            color="blue",
            alpha=0.7,
        )
        axes[0, 1].bar(
            x_pos + width,
            dynamic["nx_runtimes"],
            width,
            label="NetworkX",
            color="red",
            alpha=0.7,
        )
        axes[0, 1].set_title("Runtime Comparison Per Step")
        axes[0, 1].set_xlabel("Time Step")
        axes[0, 1].set_ylabel("Runtime (seconds)")
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels(
            [f"T{i + 1}" for i in range(len(dynamic["async_df_runtimes"]))]
        )
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Parallel efficiency over time
        axes[0, 2].plot(
            range(len(dynamic["parallel_efficiency"])),
            dynamic["parallel_efficiency"],
            "purple",
            linewidth=2,
            marker="d",
        )
        axes[0, 2].set_title("Parallel Efficiency (Sync/Async)")
        axes[0, 2].set_xlabel("Time Step")
        axes[0, 2].set_ylabel("Speedup Factor")
        axes[0, 2].axhline(
            y=1.0, color="black", linestyle="--", alpha=0.5, label="No speedup"
        )
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        # 4. Average runtime comparison (bar chart)
        algorithms = ["Async DF", "Sync DF", "NetworkX"]
        avg_runtimes = [
            dynamic["avg_async_runtime"],
            dynamic["avg_sync_runtime"],
            dynamic["avg_nx_runtime"],
        ]
        colors = ["green", "blue", "red"]

        bars = axes[1, 0].bar(algorithms, avg_runtimes, color=colors, alpha=0.7)
        axes[1, 0].set_title("Average Runtime Comparison")
        axes[1, 0].set_ylabel("Average Runtime (seconds)")

        # Add value labels on bars
        for bar, runtime in zip(bars, avg_runtimes):
            height = bar.get_height()
            axes[1, 0].text(
                bar.get_x() + bar.get_width() / 2.0,
                height + height * 0.01,
                f"{runtime:.3f}s",
                ha="center",
                va="bottom",
            )

        axes[1, 0].grid(True, alpha=0.3)

        # 5. Async vs Sync modularity comparison
        async_vs_sync = results["async_vs_sync_comparison"]
        categories = ["Async DF", "Sync DF"]
        modularities = [
            async_vs_sync["async_df"]["modularity"],
            async_vs_sync["sync_df"]["modularity"],
        ]

        bars = axes[1, 1].bar(
            categories, modularities, color=["green", "blue"], alpha=0.7
        )
        axes[1, 1].set_title("Static Modularity Comparison")
        axes[1, 1].set_ylabel("Modularity")

        for bar, mod in zip(bars, modularities):
            height = bar.get_height()
            axes[1, 1].text(
                bar.get_x() + bar.get_width() / 2.0,
                height + height * 0.01,
                f"{mod:.4f}",
                ha="center",
                va="bottom",
            )

        axes[1, 1].grid(True, alpha=0.3)

        # 6. Scalability analysis (if available)
        scalability = results.get("async_scalability", {})
        if scalability.get("runtime_vs_graph_size"):
            sizes = [entry["nodes"] for entry in scalability["runtime_vs_graph_size"]]
            runtimes = [
                entry["runtime"] for entry in scalability["runtime_vs_graph_size"]
            ]

            axes[1, 2].plot(sizes, runtimes, "orange", linewidth=2, marker="o")
            axes[1, 2].set_title("Async Scalability")
            axes[1, 2].set_xlabel("Graph Size (nodes)")
            axes[1, 2].set_ylabel("Runtime (seconds)")
            axes[1, 2].grid(True, alpha=0.3)
        else:
            axes[1, 2].text(
                0.5,
                0.5,
                "Scalability data\nnot available",
                ha="center",
                va="center",
                transform=axes[1, 2].transAxes,
            )
            axes[1, 2].set_title("Async Scalability")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Async plot saved to: {save_path}")
        else:
            plt.show()

    def export_async_results(self, filepath: str) -> None:
        """
        Export async benchmark results to CSV for further analysis.

        Args:
            filepath: Path to save the CSV file
        """
        all_data = []

        for dataset_name, results in self.results.items():
            if not dataset_name.endswith("_async"):
                continue

            # Dataset info
            info = results["dataset_info"]

            # Async vs sync comparison
            async_sync = results["async_vs_sync_comparison"]

            # Dynamic performance summary
            dynamic = results["async_dynamic_performance"]

            row = {
                "dataset": dataset_name,
                "nodes": info["nodes"],
                "edges": info["edges"],
                "time_steps": info["time_steps"],
                "async_modularity": async_sync["async_df"]["modularity"],
                "sync_modularity": async_sync["sync_df"]["modularity"],
                "modularity_diff": async_sync["comparison"]["modularity_diff"],
                "speedup_sync_vs_async": async_sync["comparison"]["speedup"],
                "total_dynamic_runtime": dynamic["total_runtime"],
                "avg_async_step_runtime": dynamic["avg_async_runtime"],
                "avg_sync_step_runtime": dynamic["avg_sync_runtime"],
                "avg_nx_step_runtime": dynamic["avg_nx_runtime"],
                "avg_parallel_efficiency": dynamic["avg_parallel_efficiency"],
                "async_modularity_stability": dynamic["async_modularity_stability"],
                "sync_modularity_stability": dynamic["sync_modularity_stability"],
            }

            all_data.append(row)

        if all_data:
            df = pd.DataFrame(all_data)
            df.to_csv(filepath, index=False)
            print(f"Async results exported to: {filepath}")
        else:
            print("No async results found to export")


def run_comprehensive_benchmark(
    dataset_paths: Dict[str, str],
    batch_range: float = 1e-3,
    initial_fraction: float = 0.6,
) -> DFLouvainBenchmark:
    """
    Run comprehensive benchmarks on multiple datasets.

    Args:
        dataset_paths: Dictionary mapping dataset names to file paths

    Returns:
        DFLouvainBenchmark instance with results
    """
    benchmark = DFLouvainBenchmark(verbose=True)

    for dataset_name, dataset_path in dataset_paths.items():
        # Determine dataset type from name or extension
        if "college" in dataset_name.lower() or dataset_path.endswith(".txt"):
            dataset_type = "college_msg"
        elif "bitcoin" in dataset_name.lower() or dataset_path.endswith(".csv"):
            dataset_type = "bitcoin"
        elif "synthetic" in dataset_name.lower() or dataset_path == "synthetic":
            dataset_type = "synthetic"
        else:
            dataset_type = "college_msg"  # default

        benchmark.benchmark_dataset(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            dataset_type=dataset_type,
            batch_range=batch_range,
            initial_fraction=initial_fraction,
        )

    return benchmark


async def run_comprehensive_async_benchmark(
    dataset_paths: Dict[str, str],
) -> DFLouvainBenchmark:
    """
    Run comprehensive async benchmarks on multiple datasets.

    Args:
        dataset_paths: Dictionary mapping dataset names to file paths

    Returns:
        DFLouvainBenchmark instance with async results
    """
    benchmark = DFLouvainBenchmark(verbose=True)

    for dataset_name, dataset_path in dataset_paths.items():
        # Determine dataset type from name or extension
        if "college" in dataset_name.lower() or dataset_path.endswith(".txt"):
            dataset_type = "college_msg"
        elif "bitcoin" in dataset_name.lower() or dataset_path.endswith(".csv"):
            dataset_type = "bitcoin"
        elif "synthetic" in dataset_name.lower() or dataset_path == "synthetic":
            dataset_type = "synthetic"
        else:
            dataset_type = "college_msg"  # default

        await benchmark.benchmark_dataset_async(
            dataset_name, dataset_path, dataset_type
        )

    return benchmark


async def run_sync_vs_async_comparison(
    dataset_paths: Dict[str, str], batch_range: float = 1e-3
) -> DFLouvainBenchmark:
    """
    Run both sync and async benchmarks for direct comparison.

    Args:
        dataset_paths: Dictionary mapping dataset names to file paths

    Returns:
        DFLouvainBenchmark instance with both sync and async results
    """
    benchmark = DFLouvainBenchmark(verbose=True)

    for dataset_name, dataset_path in dataset_paths.items():
        # Determine dataset type
        if "college" in dataset_name.lower() or dataset_path.endswith(".txt"):
            dataset_type = "college_msg"
        elif "bitcoin" in dataset_name.lower() or dataset_path.endswith(".csv"):
            dataset_type = "bitcoin"
        elif "synthetic" in dataset_name.lower() or dataset_path == "synthetic":
            dataset_type = "synthetic"
        else:
            dataset_type = "college_msg"

        # Run both sync and async benchmarks
        print(f"\n{'=' * 80}")
        print(f"COMPARING SYNC vs ASYNC for {dataset_name}")
        print(f"{'=' * 80}")

        # Sync benchmark
        await benchmark.benchmark_dataset_async(
            dataset_name, dataset_path, dataset_type, batch_range
        )

        benchmark.benchmark_dataset(
            dataset_name, dataset_path, dataset_type, batch_range
        )

        # Async benchmark

        # Print comparison summary
        sync_results = benchmark.results[dataset_name]
        async_results = benchmark.results[dataset_name + "_async"]

        print(f"\n{'=' * 60}")
        print(f"COMPARISON SUMMARY: {dataset_name}")
        print(f"{'=' * 60}")

        sync_static = sync_results["static_comparison"]
        async_static = async_results["async_vs_sync_comparison"]

        print("Static Performance:")
        print(f"  Sync DF Modularity: {sync_static['df_louvain']['modularity']:.4f}")
        print(f"  Async DF Modularity: {async_static['async_df']['modularity']:.4f}")
        print(f"  NetworkX Modularity: {sync_static['networkx']['modularity']:.4f}")

        sync_dynamic = sync_results["dynamic_performance"]
        async_dynamic = async_results["async_dynamic_performance"]

        print("\nDynamic Performance:")
        print(f"  Sync DF Avg Runtime: {sync_dynamic['avg_df_runtime']:.3f}s")
        print(f"  Async DF Avg Runtime: {async_dynamic['avg_async_runtime']:.3f}s")
        print(f"  NetworkX Avg Runtime: {sync_dynamic['avg_nx_runtime']:.3f}s")
        print(
            f"  Async Speedup vs Sync: {sync_dynamic['avg_df_runtime'] / async_dynamic['avg_async_runtime']:.2f}x"
        )

    return benchmark


def plot_sync_vs_async_comparison(
    benchmark: DFLouvainBenchmark, dataset_name: str, save_path: Optional[str] = None
) -> None:
    """
    Create a comprehensive comparison plot between sync and async results.

    Args:
        benchmark: DFLouvainBenchmark instance with both sync and async results
        dataset_name: Base name of the dataset (without '_async' suffix)
        save_path: Optional path to save the plot
    """
    sync_key = dataset_name
    async_key = dataset_name + "_async"

    if sync_key not in benchmark.results or async_key not in benchmark.results:
        print(
            f"Missing results for comparison. Need both '{sync_key}' and '{async_key}'"
        )
        return

    sync_results = benchmark.results[sync_key]
    async_results = benchmark.results[async_key]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"Sync vs Async DFLouvain Comparison: {dataset_name}", fontsize=16)

    # 1. Average runtime comparison
    sync_dynamic = sync_results["dynamic_performance"]
    async_dynamic = async_results["async_dynamic_performance"]

    algorithms = ["Sync DF", "Async DF", "NetworkX"]
    avg_runtimes = [
        sync_dynamic["avg_df_runtime"],
        async_dynamic["avg_async_runtime"],
        sync_dynamic["avg_nx_runtime"],
    ]
    colors = ["blue", "green", "red"]

    bars = axes[0, 0].bar(algorithms, avg_runtimes, color=colors, alpha=0.7)
    axes[0, 0].set_title("Average Runtime Comparison")
    axes[0, 0].set_ylabel("Average Runtime (seconds)")

    for bar, runtime in zip(bars, avg_runtimes):
        height = bar.get_height()
        axes[0, 0].text(
            bar.get_x() + bar.get_width() / 2.0,
            height + height * 0.01,
            f"{runtime:.3f}s",
            ha="center",
            va="bottom",
        )

    axes[0, 0].grid(True, alpha=0.3)

    # 2. Modularity comparison
    sync_static = sync_results["static_comparison"]
    async_static = async_results["async_vs_sync_comparison"]

    algorithms = ["Sync DF", "Async DF", "NetworkX"]
    modularities = [
        sync_static["df_louvain"]["modularity"],
        async_static["async_df"]["modularity"],
        sync_static["networkx"]["modularity"],
    ]

    bars = axes[0, 1].bar(algorithms, modularities, color=colors, alpha=0.7)
    axes[0, 1].set_title("Static Modularity Comparison")
    axes[0, 1].set_ylabel("Modularity")

    for bar, mod in zip(bars, modularities):
        height = bar.get_height()
        axes[0, 1].text(
            bar.get_x() + bar.get_width() / 2.0,
            height + height * 0.01,
            f"{mod:.4f}",
            ha="center",
            va="bottom",
        )

    axes[0, 1].grid(True, alpha=0.3)

    # 3. Runtime over time
    time_steps = range(len(sync_dynamic["df_runtimes"]))
    axes[1, 0].plot(
        time_steps,
        sync_dynamic["df_runtimes"],
        "b-",
        linewidth=2,
        label="Sync DF",
        marker="o",
    )
    axes[1, 0].plot(
        time_steps,
        async_dynamic["async_df_runtimes"],
        "g-",
        linewidth=2,
        label="Async DF",
        marker="s",
    )
    axes[1, 0].plot(
        time_steps,
        sync_dynamic["nx_runtimes"],
        "r-",
        linewidth=2,
        label="NetworkX",
        marker="^",
    )
    axes[1, 0].set_title("Runtime Over Time")
    axes[1, 0].set_xlabel("Time Step")
    axes[1, 0].set_ylabel("Runtime (seconds)")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Speedup analysis
    speedup_over_nx = [
        nx_time / async_time if async_time > 0 else 1.0
        for nx_time, async_time in zip(
            sync_dynamic["nx_runtimes"], async_dynamic["async_df_runtimes"]
        )
    ]
    speedup_over_sync = [
        sync_time / async_time if async_time > 0 else 1.0
        for sync_time, async_time in zip(
            sync_dynamic["df_runtimes"], async_dynamic["async_df_runtimes"]
        )
    ]

    axes[1, 1].plot(
        time_steps,
        speedup_over_nx,
        "purple",
        linewidth=2,
        label="Async vs NetworkX",
        marker="d",
    )
    axes[1, 1].plot(
        time_steps,
        speedup_over_sync,
        "orange",
        linewidth=2,
        label="Async vs Sync DF",
        marker="o",
    )
    axes[1, 1].axhline(
        y=1.0, color="black", linestyle="--", alpha=0.5, label="No speedup"
    )
    axes[1, 1].set_title("Speedup Analysis")
    axes[1, 1].set_xlabel("Time Step")
    axes[1, 1].set_ylabel("Speedup Factor")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Comparison plot saved to: {save_path}")
    else:
        plt.show()
