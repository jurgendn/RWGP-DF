import warnings
from collections import defaultdict
from typing import Dict, List, Literal, Text

import networkx as nx
import wandb
from tqdm.auto import tqdm

from src.components.dataset import TemporalChanges
from src.components.factory import (
    MethodDynamicResults,
)
from src.models import (
    DeltaScreeningLouvain,
    DynamicFrontierLouvain,
    GPDynamicFrontierLouvain,
    NaiveDynamicLouvain,
    StaticLouvain,
)

warnings.filterwarnings("ignore")



class BenchmarkMethod:
    def __init__(
        self,
        model_name: Text,
        model_instance: DeltaScreeningLouvain | NaiveDynamicLouvain | DynamicFrontierLouvain | GPDynamicFrontierLouvain | StaticLouvain,
        logger: bool = False,
        verbose: bool = False,
    ):
        """
        Initialize a benchmark method.

        Args:
            name: Name of the method
            method_instance: Instance of the method to benchmark
        """
        self.model_name = model_name
        self.model = model_instance
        self.logger = logger
        self.verbose = verbose

        self.is_fitted = False

    def benchmark(
        self,
        G: nx.Graph,
        temporal_changes: List[TemporalChanges],
    ) -> Dict[Text, MethodDynamicResults]:
        """
        Benchmark dynamic performance of DFLouvain vs NetworkX Louvain.
        The results schema is as follows:

        MethodDynamicResults
            runtimes: List[float],           # Runtime for each step
            modularities: List[float],       # Modularity score for each step
            affected_nodes: List[int],       # Currently unused/empty
            iterations_per_step: List[int],  # Currently unused/empty

            -- Properties --
            total_runtime: float,            # Sum of all runtimes
            avg_runtime: float,              # Mean of runtimes
            modularity_stability: float      # Standard deviation of modularities
            modularity_range: Tuple[Optional[float], Optional[float]]
        """
        if self.verbose:
            print("\n--- Dynamic Performance Benchmark ---")

        # Initialize results dictionary
        results = defaultdict(MethodDynamicResults)
        initial_modularity = self.model.get_modularity()
        for key in results.keys():
            results[key].modularities.append(initial_modularity)

        current_graph = G.copy()
        progress_bar = tqdm(
            enumerate(temporal_changes),
            desc="Processing Temporal Changes",
            total=len(temporal_changes),
        )
        for i, changes in progress_bar:
            progress_bar.set_description(f"Step {i + 1}/{len(temporal_changes)}")
            deletions = changes.deletions
            insertions = changes.insertions
            progress_bar.set_description(
                f"Step {i + 1}/{len(temporal_changes)} - Get changes"
            )
        
            if self.verbose:
                progress_bar.set_postfix_str(
                    f"Processing {self.model_name} - Step {i + 1}"
                )
            res = self.model.run(deletions, insertions)

            for key, value in res.items():
                results[key].runtimes.append(value.runtime)
                results[key].modularities.append(value.modularity)
                results[key].affected_nodes.append(value.affected_nodes)


            for edge in deletions:
                if current_graph.has_edge(*edge):
                    current_graph.remove_edge(*edge)
            for edge in insertions:
                if len(edge) == 3:  # (u, v, weight)
                    current_graph.add_edge(edge[0], edge[1], weight=edge[2])
                else:  # (u, v)
                    current_graph.add_edge(edge[0], edge[1], weight=1.0)
        
        self.is_fitted = True
        self.results = results
        return results


class Runner:
    def __init__(
        self,
        models: Dict[
            Text,
            DeltaScreeningLouvain
            | NaiveDynamicLouvain
            | DynamicFrontierLouvain
            | GPDynamicFrontierLouvain
            | StaticLouvain,
        ],
        sampler_type: Literal["default", "selective_sampler"] = "default",
        logger: bool = False,
        verbose: bool = False,
    ):
        """
        Initialize the benchmark runner.

        Args:
            verbose: Whether to print detailed progress information
        """
        self.verbose = verbose
        self.logger = logger
        self.sampler_type = sampler_type
        self.results = {}
        self.models = models 
        if self.logger:
            wandb.init(
                project="gp-df-louvain",
                reinit=True,
            )

    def __forward_normal_slow(
        self,
        model_name: Text,
        model: DeltaScreeningLouvain
        | NaiveDynamicLouvain
        | DynamicFrontierLouvain
        | GPDynamicFrontierLouvain
        | StaticLouvain,
        G: nx.Graph,
        temporal_changes: List[TemporalChanges],
    ) -> Dict[Text, MethodDynamicResults]:
        """
        Run the benchmark for a single model.

        Args:
            model: Instance of the algorithm to benchmark
            G: Initial graph
            temporal_changes: List of temporal changes to apply

        Returns:
            MethodDynamicResults containing benchmark results
        """
        benchmark_method = BenchmarkMethod(
            model_name=model_name, model_instance=model, verbose=self.verbose
        )
        return benchmark_method.benchmark(G, temporal_changes)

    def forward(
        self,
        G: nx.Graph,
        temporal_changes: List[TemporalChanges],
    ):
        if self.verbose:
            print("\n--- Running Benchmark ---")

        for model_name, model in self.models.items():
            print(f"Running benchmark for {model_name}...")
            result = self.__forward_normal_slow(
                model_name=model_name,
                model=model,
                G=G,
                temporal_changes=temporal_changes,
            )
            self.results = {**self.results, **result}
        return self.results
