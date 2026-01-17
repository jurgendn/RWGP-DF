from typing import Dict, Optional, Text

import matplotlib.pyplot as plt
import seaborn as sns
from plotly import graph_objects as go
from plotly.subplots import make_subplots

from src.components.factory import (
    MethodDynamicResults,
)


class Plotter:
    def __init__(self):
        pass

    def plot_results(
        self,
        dataset_name: str,
        results: Dict[Text, MethodDynamicResults],
        save_path: Optional[str] = None,
    ):
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

        return fig


class SeparatePlotter:
    def __init__(self):
        pass

    def plot_results(
        self,
        dataset_name: str,
        results: Dict[Text, MethodDynamicResults],
        save_path: Optional[str] = None,
    ):
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

        return fig

    def plot_modularity(self, dataset_name: str, results: dict):
        fig = go.Figure()
        for method, result in results.items():
            fig.add_trace(
                go.Scatter(
                    x=result.time_steps,
                    y=result.modularities,
                    mode="lines+markers",
                    name=method,
                    line=dict(width=2),
                    marker=dict(size=6),
                )
            )
        fig.update_layout(
            xaxis_title="Time Step",
            yaxis_title="Modularity",
            legend_title="Method",
            template="plotly_white",
            font=dict(size=14),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
            margin=dict(l=40, r=40, t=60, b=40),
        )
        fig.update_traces(marker=dict(line=dict(width=1, color="DarkSlateGrey")))
        return fig

    def plot_modularity_by_batch(
        self, results: Dict[Text, Dict[Text, MethodDynamicResults]]
    ):
        num_batches = len(results)
        fig = make_subplots(
            rows=1,
            cols=num_batches,
            # Removed shared_yaxes=True to allow independent y-axis scaling
        )

        # Define consistent colors and styles for each method
        method_colors = {
            "DF Louvain": "#1f77b4",
            "GP-DF Louvain": "#ff7f0e",
            "Static Louvain": "#2ca02c",
            "ND Louvain": "#d62728",
            "DS Louvain": "#9467bd",
        }

        # Get all unique methods across all batches
        all_methods = set()
        for batch_group in results.values():
            all_methods.update(batch_group.keys())

        # Create consistent styles for all methods
        method_styles = {
            method: dict(
                line=dict(width=2, color=method_colors.get(method, "#000000")),
                marker=dict(size=6, line=dict(width=1, color="DarkSlateGrey")),
            )
            for method in all_methods
        }

        for col_idx, (batch_size, batch_group) in enumerate(results.items()):
            for method, result in batch_group.items():
                style = method_styles.get(method, {})
                fig.add_trace(
                    go.Scatter(
                        x=result.time_steps,
                        y=result.modularities,
                        mode="lines+markers",
                        name=method,
                        line=style["line"],
                        marker=style["marker"],
                        showlegend=(col_idx == 0),  # Only show legend for first subplot
                        legendgroup=method,  # Group traces by method
                    ),
                    row=1,
                    col=col_idx + 1,
                )
            fig.update_xaxes(title_text=batch_size, row=1, col=col_idx + 1)

        fig.update_layout(
            legend_title="Method",
            template="plotly_white",
            font=dict(size=14),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5
            ),
            margin=dict(l=40, r=40, t=80, b=40),
            showlegend=True,
        )

        return fig

    def plot_num_communities_by_batch(
        self, results: Dict[Text, Dict[Text, MethodDynamicResults]]
    ):
        num_batches = len(results)
        fig = make_subplots(
            rows=1,
            cols=num_batches,
            # Removed shared_yaxes=True to allow independent y-axis scaling
        )

        # Define consistent colors and styles for each method
        method_colors = {
            "DF Louvain": "#1f77b4",
            "GP-DF Louvain": "#ff7f0e",
            "Static Louvain": "#2ca02c",
            "ND Louvain": "#d62728",
            "DS Louvain": "#9467bd",
        }

        # Get all unique methods across all batches
        all_methods = set()
        for batch_group in results.values():
            all_methods.update(batch_group.keys())

        # Create consistent styles for all methods
        method_styles = {
            method: dict(
                line=dict(width=2, color=method_colors.get(method, "#000000")),
                marker=dict(size=6, line=dict(width=1, color="DarkSlateGrey")),
            )
            for method in all_methods
        }

        for col_idx, (batch_size, batch_group) in enumerate(results.items()):
            for method, result in batch_group.items():
                style = method_styles.get(method, {})
                fig.add_trace(
                    go.Scatter(
                        x=result.time_steps,
                        y=result.num_communities,
                        mode="lines+markers",
                        name=method,
                        line=style["line"],
                        marker=style["marker"],
                        showlegend=(col_idx == 0),  # Only show legend for first subplot
                        legendgroup=method,  # Group traces by method
                    ),
                    row=1,
                    col=col_idx + 1,
                )
            fig.update_xaxes(title_text=batch_size, row=1, col=col_idx + 1)

        fig.update_layout(
            legend_title="Method",
            template="plotly_white",
            font=dict(size=14),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5
            ),
            margin=dict(l=40, r=40, t=80, b=40),
            showlegend=True,
        )

        return fig

    def plot_avg_modularity_by_batch(
        self, results: Dict[Text, Dict[Text, MethodDynamicResults]]
    ):
        num_batches = len(results)
        fig = make_subplots(rows=1, cols=num_batches)

        # Define consistent colors for each method
        method_colors = {
            "DF Louvain": "#1f77b4",
            "GP-DF Louvain": "#ff7f0e",
            "Static Louvain": "#2ca02c",
            "ND Louvain": "#d62728",
            "DS Louvain": "#9467bd",
        }

        # Get all unique methods across all batches
        all_methods = set()
        for batch_group in results.values():
            all_methods.update(batch_group.keys())

        for col_idx, (batch_size, batch_group) in enumerate(results.items()):
            methods = []
            avg_modularities = []
            colors = []
            for method, result in batch_group.items():
                methods.append(method)
                avg_modularities.append(result.avg_modularities)
                colors.append(method_colors.get(method, "#000000"))
            fig.add_trace(
                go.Bar(
                    x=methods,
                    y=avg_modularities,
                    marker_color=colors,
                    text=[f"{v:.4f}" for v in avg_modularities],
                    textposition="auto",
                    name=str(batch_size),
                ),
                row=1,
                col=col_idx + 1,
            )
            fig.update_xaxes(title_text=batch_size, row=1, col=col_idx + 1)
            fig.update_yaxes(title_text="Avg Modularity", row=1, col=col_idx + 1)

        fig.update_layout(
            legend_title="Batch Size",
            template="plotly_white",
            font=dict(size=14),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5
            ),
            margin=dict(l=40, r=40, t=80, b=40),
            showlegend=False,
        )

        return fig

    def plot_avg_runtime_by_batch(
        self, results: Dict[Text, Dict[Text, MethodDynamicResults]]
    ):
        num_batches = len(results)
        fig = make_subplots(rows=1, cols=num_batches)

        # Define consistent colors for each method
        method_colors = {
            "DF Louvain": "#1f77b4",
            "GP-DF Louvain": "#ff7f0e",
            "Static Louvain": "#2ca02c",
            "ND Louvain": "#d62728",
            "DS Louvain": "#9467bd",
        }

        # Get all unique methods across all batches
        all_methods = set()
        for batch_group in results.values():
            all_methods.update(batch_group.keys())

        for col_idx, (batch_size, batch_group) in enumerate(results.items()):
            methods = []
            avg_runtimes = []
            colors = []
            for method, result in batch_group.items():
                methods.append(method)
                avg_runtimes.append(result.avg_runtime)
                colors.append(method_colors.get(method, "#000000"))
            fig.add_trace(
                go.Bar(
                    x=methods,
                    y=avg_runtimes,
                    marker_color=colors,
                    text=[f"{v:.4f}" for v in avg_runtimes],
                    textposition="auto",
                    name=str(batch_size),
                ),
                row=1,
                col=col_idx + 1,
            )
            fig.update_xaxes(title_text=batch_size, row=1, col=col_idx + 1)
            fig.update_yaxes(title_text="Avg Runtime", row=1, col=col_idx + 1)

        fig.update_layout(
            legend_title="Batch Size",
            template="plotly_white",
            font=dict(size=14),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5
            ),
            margin=dict(l=40, r=40, t=80, b=40),
            showlegend=False,
        )

        return fig

    def plot_batch_avg_modularity_avg_runtime(
        self,
        results: Dict[Text, Dict[Text, MethodDynamicResults]],
        lower_ratio: float = 5 / 7,
        upper_ratio: float = 1,
        font_size: int = 36,
        title: str | None = "Large graph, large community size",
    ):
        def make_subplot_title(results: Dict[Text, Dict[Text, MethodDynamicResults]]):
            subplot_titles = []
            keys = results.keys()
            for key in keys:
                text = f"Batch size: {key}"
                subplot_titles.append(text)
            return subplot_titles

        target_methods = ["GP-DF Louvain", "DF Louvain", "Static Louvain"]
        # Define consistent colors for each method
        method_colors = {
            "GP-DF Louvain": "#ff0202",
            "DF Louvain": "#1f77b4",
            "Static Louvain": "#2ca02c",
            "ND Louvain": "#00adc0",
            "DS Louvain": "#9467bd",
        }
        num_batches = len(results)
        fig = make_subplots(
            rows=2,
            cols=num_batches,
            subplot_titles=make_subplot_title(results=results),
            shared_yaxes=False,
            vertical_spacing=0.05,
            horizontal_spacing=0.1,
        )
        # Increase subplot title font size
        for annotation in fig.layout.annotations:
            annotation.font = dict(size=font_size)
        for col_idx, (batch_size, batch_group) in enumerate(results.items()):
            methods = []
            avg_modularities = []
            avg_runtimes = []
            colors = []
            target_methods = {
                method: batch_group[method]
                for method in target_methods
                if method in batch_group
            }
            for method, result in target_methods.items():
                methods.append(method)
                if method not in target_methods:
                    continue
                avg_modularities.append(result.avg_modularities)
                avg_runtimes.append(result.avg_runtime)
                colors.append(method_colors.get(method, "#000000"))
            # Bar chart for avg modularity (row 1)
            fig.add_trace(
                go.Bar(
                    x=methods,
                    y=avg_modularities,
                    marker_color=colors,
                    text=[f"{v:.4f}" for v in avg_modularities],
                    textposition="auto",
                    textfont=dict(size=font_size),  # Increased title/value font size
                    showlegend=False,
                ),
                row=1,
                col=col_idx + 1,
            )
            # Bar chart for avg runtime (row 2)
            fig.add_trace(
                go.Bar(
                    x=methods,
                    y=avg_runtimes,
                    marker_color=colors,
                    text=[f"{v:.4f}" for v in avg_runtimes],
                    textposition="auto",
                    textfont=dict(size=font_size),  # Increased title/value font size
                    showlegend=False,
                ),
                row=2,
                col=col_idx + 1,
            )
            fig.update_xaxes(
                row=1,
                col=col_idx + 1,
                showticklabels=False,  # Hide x-axis labels on top row
            )
            fig.update_xaxes(
                row=2,
                col=col_idx + 1,
                tickangle=-30,
                tickfont=dict(size=font_size),
            )
            fig.update_yaxes(
                title_text="Avg Modularity",
                row=1,
                col=col_idx + 1,
                showticklabels=True,
                # showticklabels=False,
            )
            fig.update_yaxes(
                title_text="Avg Runtime (s)",
                row=2,
                col=col_idx + 1,
                showticklabels=True,
                # showticklabels=False,
            )
            # Set the y-axis range for the first row in (max(min)/2, max)
            # Dynamically set y-axis range for each subplot in the first row
            max_modularity = max(avg_modularities)
            min_modularity = min(avg_modularities)
            fig.update_yaxes(
                range=[min_modularity * lower_ratio, max_modularity * upper_ratio],
                row=1,
                col=col_idx + 1,
            )

        if title is not None:
            fig.update_layout(
                title=dict(
                    text=title,
                    x=0.5,
                    xanchor="center",
                    y=0.95,
                    yanchor="bottom",
                    font=dict(size=font_size),
                )
            )
        fig.update_layout(
            legend_title="Batch Size",
            template="plotly_white",
            font=dict(size=font_size),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5
            ),
            margin=dict(
                l=10, r=10, t=60, b=60
            ),  # Increased bottom margin for rotated labels
            showlegend=False,
            height=700,
        )
        # Set all font colors to black
        fig.update_layout(font=dict(size=font_size, color="black"))
        # Ensure title font color is black
        if fig.layout.title and fig.layout.title.font:
            fig.layout.title.font.color = "black"
        # Ensure subplot annotation (subplot titles) font colors are black
        for ann in fig.layout.annotations:
            if ann.font:
                ann.font.color = "black"
            else:
                ann.font = dict(size=font_size, color="black")
        # fig.add_shape(
        #     type="line",
        #     x0=0,
        #     x1=1,
        #     y0=0,  # bottom of the figure
        #     y1=0,
        #     xref="paper",
        #     yref="paper",
        #     line=dict(color="black", width=2),
        #     layer="below",
        # )
        return fig

    def plot_batch_avg_modularity_avg_runtime_one_chart(
        self, results: Dict[Text, Dict[Text, MethodDynamicResults]]
    ):
        ieee_font = dict(family="Arial, Helvetica, sans-serif", size=18, color="#222")
        ieee_legend_font = dict(family="Arial, Helvetica, sans-serif", size=18)
        ieee_title_font = dict(
            family="Arial Black, Arial, Helvetica, sans-serif", size=20
        )

        seaborn_colors = sns.color_palette("Set2", 5).as_hex()

        method_colors = {
            "DF Louvain": seaborn_colors[0],
            "GP-DF Louvain": seaborn_colors[1],
            "Static Louvain": seaborn_colors[2],
            "ND Louvain": seaborn_colors[3],
            "DS Louvain": seaborn_colors[4],
        }

        # Different marker symbols for each method
        method_symbols = {
            "DF Louvain": "circle",
            "GP-DF Louvain": "circle",
            "Static Louvain": "circle",
            "ND Louvain": "circle",
            "DS Louvain": "circle",
        }

        num_batches = len(results)
        # subplot_title_texts = list(map(lambda s: s.upper(), results.keys()))
        fig = make_subplots(
            rows=1,
            cols=num_batches,
            # subplot_titles=subplot_title_texts,
            horizontal_spacing=0.08,
            vertical_spacing=0.1,
        )

        for col_idx, (batch_size, batch_group) in enumerate(results.items()):
            for method, result in batch_group.items():
                color = method_colors.get(method, seaborn_colors[0])
                symbol = method_symbols.get(method, "circle")

                # Scatter plot for each method
                fig.add_trace(
                    go.Scatter(
                        x=[result.avg_runtime],
                        y=[result.avg_modularities],
                        mode="markers",
                        marker=dict(
                            color=color,
                            size=18,
                            symbol=symbol,
                            line=dict(color="white", width=3),
                            opacity=0.8,
                        ),
                        text=[method],
                        textposition="top center",
                        textfont=ieee_title_font,
                        name=method,
                        legendgroup=method,
                        showlegend=(col_idx == 0),  # Only show legend for first subplot
                        hovertemplate=(
                            f"<b>{method}</b><br>"
                            "Runtime: %{x:.4f}s<br>"
                            "Modularity: %{y:.4f}<br>"
                            f"Batch: {batch_size}"
                            "<extra></extra>"
                        ),
                    ),
                    row=1,
                    col=col_idx + 1,
                )

            # Update axes for each subplot
            fig.update_xaxes(
                title_text="Avg Runtime (s)",
                row=1,
                col=col_idx + 1,
                showgrid=True,
                gridwidth=1,
                gridcolor="rgba(128,128,128,0.2)",
                tickfont=ieee_font,
            )
            # Update x-ranges and y ranges
            fig.update_xaxes(
                range=[
                    0,
                    max(result.avg_runtime for result in batch_group.values()) * 1.05,
                ],
                row=1,
                col=col_idx + 1,
            )
            fig.update_yaxes(
                range=[
                    min(result.avg_modularities for result in batch_group.values())
                    * 0.8,
                    max(result.avg_modularities for result in batch_group.values())
                    * 1.02,
                ],
                row=1,
                col=col_idx + 1,
            )

            fig.update_yaxes(
                title_text="Avg Modularity",
                row=1,
                col=col_idx + 1,
                showgrid=True,
                gridwidth=1,
                gridcolor="rgba(128,128,128,0.2)",
                tickfont=ieee_font,
            )

        # Available Plotly templates (pio.templates.keys()):
        # plotly, plotly_white, plotly_dark, ggplot2, seaborn, simple_white,
        # presentation, xgridoff, ygridoff, gridon, none
        fig.update_layout(
            template="simple_white",
            font=ieee_font,
            margin=dict(
                l=60, r=160, t=80, b=70
            ),  # extra right margin for outside legend
            height=600,
            width=600,
            plot_bgcolor="rgba(248,249,250,1)",
            paper_bgcolor="white",
            legend=dict(
                orientation="v",  # vertical legend
                yanchor="top",
                y=0.99,  # align with top of plotting area
                xanchor="left",
                x=0.01,  # place inside near the left edge
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="rgba(0,0,0,0.15)",
                borderwidth=1,
                font=ieee_legend_font,
                itemclick="toggleothers",
                itemdoubleclick="toggle",
            ),
            hovermode="closest",
        )
        return fig
