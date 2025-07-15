from typing import Dict, Optional, Text

import matplotlib.pyplot as plt

from src.components.factory import (
    MethodDynamicResults,
)
from src.constants import Colormaps, Linestyles, Markers


class Plotter:
    def __init__(self):
        pass
    def plot_results(self, dataset_name: str, results: Dict[Text, MethodDynamicResults], save_path: Optional[str] = None) -> None:
        """
        Plot comprehensive benchmark results with DF vs NX comparisons.

        Args:
            dataset_name: Name of the dataset to plot
            save_path: Optional path to save the plot
        """

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f"Communities Detection Benchmarking: {dataset_name}", fontsize=16)

        # Plot modularities from results
        for _, (method_name, method_results) in enumerate(results.items()):
            axes[0, 0].plot(
                method_results.time_steps,
                method_results.modularities,
                label=method_name,
                markersize=1,
                linewidth=1,
            )
            axes[0, 0].set_title("Modularity Over Time")
            axes[0, 0].set_xlabel("Time Step")
            axes[0, 0].set_ylabel("Modularity")
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            axes[0, 1].plot(
                method_results.time_steps,
                method_results.runtimes,
                markersize=1,
                label=method_name,
                alpha=0.8,
            )
            axes[0, 1].set_title("Runtime Comparison Per Step")
            axes[0, 1].set_xlabel("Time Step")
            axes[0, 1].set_ylabel("Runtime (seconds)")
            # axes[0, 1].set_xticks(time_steps)
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            # 3. Affected nodes over time
            axes[1, 0].plot(
                method_results.time_steps,
                method_results.affected_nodes,
                label=method_name,
                markersize=1,
                linewidth=2,
                alpha=0.5,
            )
            axes[1, 0].set_title(
                "Affected Nodes Per Step (DF Louvain vs GP-DF Louvain)"
            )
            axes[1, 0].set_xlabel("Time Step")
            axes[1, 0].set_xticks(range(0, len(method_results.time_steps), 10))
            axes[1, 0].tick_params(axis="x", rotation=-45)
            axes[1, 0].legend()
            axes[1, 0].set_ylabel("Number of Affected Nodes")
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Average runtime comparison (bar chart)
        algorithms = [method_name for method_name in results.keys()]
        avg_runtimes = [
            method_results.avg_runtime for method_results in results.values()
        ]
        bars = axes[1, 1].bar(
            algorithms,
            avg_runtimes,
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
