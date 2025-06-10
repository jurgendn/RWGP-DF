"""
Benchmarking utilities for Dynamic Frontier Louvain algorithm.
"""

import time
import warnings
from collections import defaultdict
from typing import Any, Dict, List, Optional, Text

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from matplotlib import colormaps

from src.data_loader import DatasetBatchManager, DatasetWindowTimeManager
from src.df_louvain import (
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
        methods: Dict[Text, DynamicFrontierLouvain | GPDynamicFrontierLouvain],
        dataset_name: Text,
        G: nx.Graph,
        temporal_changes: Dict[Text, List[List]],
    ) -> Dict[Text, Any]:
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
        results["static_comparison"] = self._benchmark_static_comparison(
            methods=methods, G=G
        )

        # 2. Dynamic performance benchmark
        results["dynamic_performance"] = self._benchmark_dynamic_performance(
            methods=methods, G=G, temporal_changes=temporal_changes
        )

        self.results[dataset_name] = results

        if self.verbose:
            self._print_summary(results)

        return results

    def _benchmark_static_comparison(
        self,
        methods: Dict[Text, DynamicFrontierLouvain | GPDynamicFrontierLouvain],
        G: nx.Graph,
    ) -> Dict[str, Any]:
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

        results["nx"] = {
            "modularity": float(nx_modularity),
            "runtime": float(nx_time),
            "num_communities": len(nx_communities_list),
            "communities": nx_communities_list,
        }

        # Run all methods
        for method_name, method_instance in methods.items():
            start_time = time.time()
            communities_dict = method_instance.run_dynamic_frontier_louvain()
            modularity = method_instance.get_modularity()
            runtime = time.time() - start_time

            # Convert to NetworkX format for comparison
            communities = defaultdict(set)
            for node, community in communities_dict.items():
                communities[community].add(node)
            communities = list(communities.values())

            results[method_name] = {
                "modularity": modularity,
                "runtime": runtime,
                "num_communities": len(communities),
                "communities": communities,
            }

        # Comparison metrics
        # Comparison metrics
        comparison_metrics = {}
        for method_name in methods.keys():
            comparison_metrics[method_name] = {}
        
        # Compare each method with NetworkX
        for method_name, method_results in results.items():
            if method_name != "nx":
                comparison_metrics[method_name]["modularity_diff"] = abs(
                    results["nx"]["modularity"] - method_results["modularity"]
                )
                comparison_metrics[method_name]["runtime_ratio"] = (
                    method_results["runtime"] / results["nx"]["runtime"]
                    if results["nx"]["runtime"] > 0
                    else float("inf")
                )
                comparison_metrics[method_name]["community_count_diff"] = abs(
                    results["nx"]["num_communities"] - method_results["num_communities"]
                )
        
        results["comparison"] = comparison_metrics

        if self.verbose:
            for method_name, method_results in results.items():
                if method_name != "nx" and method_name != "comparison":
                    print(
                        f"{method_name}: Modularity={method_results['modularity']:.4f}, Runtime={method_results['runtime']:.3f}s, Communities={method_results['num_communities']}"
                    )
            
            print(
                f"NetworkX: Modularity={results['nx']['modularity']:.4f}, Runtime={results['nx']['runtime']:.3f}s, Communities={results['networkx']['num_communities']}"
            )
            
            # Print comparison metrics
            for key, value in results['comparison'].items():
                if 'modularity_diff' in key:
                    print(f"{key}: {value:.4f}")
                elif 'runtime_ratio' in key:
                    print(f"{key}: {value:.2f}")

        return results

    def _benchmark_dynamic_performance(
        self,
        methods: Dict[Text, DynamicFrontierLouvain | GPDynamicFrontierLouvain],
        G: nx.Graph,
        temporal_changes: List[Dict],
    ) -> Dict[str, Any]:
        """
        Benchmark dynamic performance of DFLouvain vs NetworkX Louvain.
        """
        if self.verbose:
            print("\n--- Dynamic Performance Benchmark ---")

        # Initialize results dictionary
        results = {"general": {"total_runtime": 0}}
        for method_name, _ in methods.items():
            results[method_name] = {
                "runtimes": [],
                "modularities": [],
                "affected_nodes": [],
                "total_runtime": 0,
                "iterations_per_step": [],
            }
        results["nx"] = {
            "runtimes": [],
            "modularities": [],
            "affected_nodes": [],
            "total_runtime": 0,
            "iterations_per_step": [],
        }
        
        # Initial NetworkX Louvain
        nx_start_time = time.time()
        nx_communities = nx.algorithms.community.louvain_communities(G, weight="weight")
        initial_nx_modularity = nx.algorithms.community.modularity(
            G, nx_communities, weight="weight"
        )
        nx_initial_time = time.time() - nx_start_time
        results['nx']["modularities"].append(initial_nx_modularity)
        results['nx']["runtimes"].append(nx_initial_time)

        for method_name, method_instance in methods.items():
            # Initialize results for each method
            start_time = time.time()
            method_instance.run_dynamic_frontier_louvain()
            initial_time = time.time() - start_time
            initial_modularity = method_instance.get_modularity()
            
            results[method_name]["runtimes"] = [initial_time]
            results[method_name]["modularities"] = [initial_modularity]

        # Keep track of current graph for NetworkX
        current_graph = G.copy()

        total_start_time = time.time()

        # Process temporal changes
        progress_bar = tqdm(
            enumerate(temporal_changes),
            desc="Processing Temporal Changes",
            total=len(temporal_changes),
        )
        for i, changes in progress_bar:
            progress_bar.set_description(f"Step {i + 1}/{len(temporal_changes)}")
            deletions = changes.get("deletions", [])
            insertions = changes.get("insertions", [])
            progress_bar.set_description(
                f"Step {i + 1}/{len(temporal_changes)} - Get changes"
            )
            for method_name, method_instance in methods.items():
                if self.verbose:
                    progress_bar.set_postfix_str(
                        f"Processing {method_name} - Step {i + 1}"
                    )
                step_start_time = time.time()
                method_instance.run_dynamic_frontier_louvain(deletions, insertions)
                step_modularity = method_instance.get_modularity()
                step_affected_nodes = len(method_instance.get_affected_nodes())
                step_runtime = time.time() - step_start_time

                results[method_name]["runtimes"].append(step_runtime)
                results[method_name]["modularities"].append(step_modularity)
                results[method_name]["affected_nodes"].append(step_affected_nodes)
            
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
            results["nx"]["runtimes"].append(nx_step_runtime)
            results["nx"]["modularities"].append(nx_modularity)

            # Get current benchmark to update progress bar
            intermediate_value = {}
            intermediate_value["nx modularity"] = nx_modularity
            intermediate_value["nx runtime"] = nx_step_runtime
            for method_name, method_instance in methods.items():
                if method_name in ["general"]:
                    continue
                intermediate_value[f"{method_instance.__shortname__} modularity"] = results[method_name]["modularities"][-1]
                intermediate_value[f"{method_instance.__shortname__} runtime"] = results[method_name]["runtimes"][-1]
                intermediate_value[f"{method_instance.__shortname__} affected_nodes"] = results[
                    method_name
                ]["affected_nodes"][-1]
                
            progress_bar.set_postfix(ordered_dict=intermediate_value)


        results["general"]["total_runtime"] = time.time() - total_start_time
        # Calculate summary statistics for each method
        for method_name in methods.keys():
            results[method_name]["avg_runtime"] = np.mean(results[method_name]["runtimes"])
            results[method_name]["modularity_stability"] = np.std(results[method_name]["modularities"])
            results[method_name]["total_runtime"] = sum(results[method_name]["runtimes"])
        
        # Calculate summary statistics for NetworkX
        results["nx"]["avg_runtime"] = np.mean(results["nx"]["runtimes"])
        results["nx"]["modularity_stability"] = np.std(results["nx"]["modularities"])
        results["nx"]["total_runtime"] = sum(results["nx"]["runtimes"])

        if self.verbose:
            print(f"Total dynamic runtime: {results['general']['total_runtime']:.3f}s")
            for method_name in methods.keys():
                modularities = results[method_name]['modularities']
                print(
                    f"Average {method_name} step runtime: {results[method_name]['avg_runtime']:.3f}s"
                )
                print(
                    f"{method_name} Modularity range: [{min(modularities):.4f}, {max(modularities):.4f}]"
                )
                print("=" * 40)
            print(f"Average NX step runtime: {results['nx']['avg_runtime']:.3f}s")
            nx_modularities = results['nx']['modularities']
            print(
                f"NX Modularity range: [{min(nx_modularities):.4f}, {max(nx_modularities):.4f}]"
            )
            print("=" * 40)

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
        print(f"  Total Runtime: {dynamic['general']['total_runtime']:.3f}s")
        
        # Print results for each method dynamically
        for method_name in dynamic.keys():
            if method_name != "general" and method_name != "nx":
                print(
                    f"  Avg {method_name} Step Runtime: {dynamic[method_name]['avg_runtime']:.3f}s"
                )
                print(
                    f"  {method_name} Modularity Std: {dynamic[method_name]['modularity_stability']:.4f}"
                )
        
        # Print NetworkX results
        if "nx" in dynamic:
            print(f"  Avg NX Step Runtime: {dynamic['nx']['avg_runtime']:.3f}s")
            print(f"  NX Modularity Std: {dynamic['nx']['modularity_stability']:.4f}")

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
        color_map = colormaps["tab10"]
        if dataset_name not in self.results:
            print(f"No results found for dataset: {dataset_name}")
            return

        results = self.results[dataset_name]

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f"DFLouvain vs NetworkX Louvain: {dataset_name}", fontsize=16)

        dynamic = results["dynamic_performance"]
        info = results["dataset_info"]
        time_steps = range(info["time_steps"] + 1)
        for idx, method_name in enumerate(dynamic.keys()):
            if method_name not in ["general"]:
                modularities = dynamic[method_name]["modularities"]
                # 1. Modularity comparison over time (line plot)
                axes[0, 0].plot(
                    time_steps,
                    modularities,
                    label=method_name,
                    color=color_map(idx % 10),
                    linewidth=1,
                    alpha=0.8,
                )
            axes[0, 0].set_title("Modularity Over Time")
            axes[0, 0].set_xlabel("Time Step")
            axes[0, 0].set_ylabel("Modularity")
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            # 2. Runtime comparison (bar chart)
            if method_name not in ["general"]:
                run_time = dynamic[method_name]["runtimes"]
                axes[0, 1].plot(
                    time_steps,
                    run_time,
                    color=color_map(idx % 10),
                    linewidth=1,
                    label=method_name,
                    alpha=0.8,
                )
            axes[0, 1].set_title("Runtime Comparison Per Step")
            axes[0, 1].set_xlabel("Time Step")
            axes[0, 1].set_ylabel("Runtime (seconds)")
            axes[0, 1].set_xticks(time_steps)
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            # 3. Affected nodes over time
            if method_name not in ["general", "nx"]:
                affected_nodes = dynamic[method_name]["affected_nodes"]
                axes[1, 0].plot(
                    time_steps[:-1],
                    affected_nodes,
                    label=method_name,
                    linewidth=2,
                    alpha=0.75,
                )
            axes[1, 0].set_title(
                "Affected Nodes Per Step (DF Louvain vs GP-DF Louvain)"
            )
            axes[1, 0].set_xlabel("Time Step")
            axes[1, 0].set_xticks(time_steps[:-1])
            axes[1, 0].legend()
            axes[1, 0].set_ylabel("Number of Affected Nodes")
            axes[1, 0].grid(True, alpha=0.3)
        # 4. Average runtime comparison (bar chart)
        algorithms = [
            method_name
            for method_name in dynamic.keys()
            if method_name not in ["general"]
        ]
        avg_runtimes = [
            dynamic[method_name]["avg_runtime"] for method_name in algorithms
        ]
        bars = axes[1, 1].bar(
            algorithms,
            avg_runtimes,
            color=[color_map(i) for i in range(len(algorithms))],
            alpha=1,
        )
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
            print("=" * 40)

            # Dynamic performance summary
            dynamic = results["dynamic_performance"]
            # Stability summary
            stability = results["community_stability"]

            row = {
                "dataset": dataset_name,
                "nodes": info["nodes"],
                "edges": info["edges"],
                "time_steps": info["time_steps"],
                "nx_modularity": static["nx"]["modularity"],
                "total_dynamic_runtime": dynamic["general"]["total_runtime"],
                "avg_nx_step_runtime": dynamic["nx"]["avg_runtime"],
                "avg_changes_per_step": stability["stability_metrics"][
                    "avg_changes_per_step"
                ],
            }
            
            # Add method-specific data using a for loop
            for method_name in dynamic:
                if method_name == "general" or method_name == "nx":
                    continue
                # Static comparison data
                row[f"{method_name} Modularity"] = static[method_name]["modularity"]
                row[f"{method_name} Modularity Diff"] = static["comparison"][method_name]["modularity_diff"]
                
                # Dynamic performance data
                if method_name in dynamic:
                    row[f"avg_{method_name}_step_runtime"] = dynamic[method_name]["avg_runtime"]
                    row[f"{method_name}_modularity_stability"] = dynamic[method_name]["modularity_stability"]
                else:
                    row[f"avg_{method_name}_step_runtime"] = 0
                    row[f"{method_name}_modularity_stability"] = 0

            all_data.append(row)

        df = pd.DataFrame(all_data)
        df.to_csv(filepath, index=False)
        print(f"Results exported to: {filepath}")


def run_comprehensive_benchmark(
    data_manager: DatasetBatchManager | DatasetWindowTimeManager,
    target_datasets: List[Text],
    config: Dict[Text, Any],
) -> DFLouvainBenchmark:
    """
    Run comprehensive benchmarks on multiple datasets.

    Args:
        dataset_paths: Dictionary mapping dataset names to file paths

    Returns:
        DFLouvainBenchmark instance with results
    """
    benchmark = DFLouvainBenchmark(verbose=False)

    for dataset_name, dataset_config in config.items():
        if dataset_name not in target_datasets:
            continue
        G, temporal_changes = data_manager.get_dataset(**dataset_config)
        methods = {
            "Dynamic Frontier Louvain": DynamicFrontierLouvain(graph=G, verbose=False),
            "GP - Dynamic Frontier Louvain": GPDynamicFrontierLouvain(
                graph=G, verbose=False
            ),
        }
        benchmark.benchmark_dataset(
            methods=methods,
            dataset_name=dataset_name,
            G=G,
            temporal_changes=temporal_changes,
        )

    return benchmark
