"""
Main script to run DFLouvain benchmarks on the provided datasets.
"""
import os
from pathlib import Path
from typing import Any, Dict, List, Text

import networkx as nx
import yaml

from src import GPDynamicFrontierLouvain, NaiveDynamicLouvain, StaticLouvain
from src.benchmarks import Runner
from src.data_loader import DatasetBatchManager, DatasetWindowTimeManager
from src.utils import helpers
from src.utils.plotter import Plotter


def run_comprehensive_benchmark(
    data_manager: DatasetBatchManager | DatasetWindowTimeManager,
    dataset_config: Dict[Text, Any],
):
    full_nodes_config = dataset_config.copy()
    full_nodes_config["load_full_nodes"] = True
    
    G, temporal_changes = data_manager.get_dataset(**dataset_config)
    initial_communities = nx.algorithms.community.louvain_communities(G)

    initial_communities_dict = {}
    for community_id, community in enumerate(initial_communities): # type: ignore
        for node in community:
            initial_communities_dict[node] = community_id
  
    methods = {
        # "Naive Dynamic Louvain": NaiveDynamicLouvain(
        #     graph=G, initial_communities=initial_communities_dict, verbose=False
        # ),
        # "Delta Screening Louvain": DeltaScreeningLouvain(
        #     graph=G, initial_communities=initial_communities_dict, verbose=False
        # ),
        # "Static Louvain": StaticLouvain(
        #     graph=G, initial_communities=initial_communities_dict, verbose=False
        # ),
        "GP - Dynamic Frontier Louvain": GPDynamicFrontierLouvain(
            graph=G,
            initial_communities=initial_communities_dict,
            verbose=False,
            refine_version="v2-full",
        ),
    }
    runner = Runner(
        models=methods,
        logger=False,
        verbose=True,
    )
    results = runner.forward(G, temporal_changes)

    return results

def main():
    with open("./config/default.yaml", "r") as file:
        config = yaml.safe_load(file)
    mode = config["mode"]
    target_datasets = config["target_datasets"]
    if mode not in ["batch", "window_frame"]:
        raise ValueError(f"Invalid mode: {mode}. Expected 'batch' or 'window_frame'.")
    if mode == "batch":
        data_manager = DatasetBatchManager()
        data_config = config["batch_data_manager"]
    elif mode == "window_frame":
        data_manager = DatasetWindowTimeManager()
        data_config = config["window_frame_data_manager"]
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    plotter = Plotter()

    # Export results
    results_dir = Path(os.path.join(os.curdir, "results", f"{mode}_benchmark"))
    results_dir.mkdir(exist_ok=True)

    for dataset_name, dataset_config in data_config.items():
        if dataset_name in target_datasets:
            filename = helpers.generate_plot_filename(
                mode=mode,
                dataset_config=dataset_config,
            )
            results = run_comprehensive_benchmark(
                data_manager=data_manager,
                dataset_config=dataset_config,
            )

            dataset_dir = Path(os.path.join(results_dir, dataset_name))
            dataset_dir.mkdir(exist_ok=True)
            # Create plot for the dataset
            plot_path = os.path.join(
                dataset_dir,
                filename,
            )
            plotter.plot_results(
                dataset_name=dataset_name, results=results, save_path=plot_path
            )

    print(f"\nBenchmark complete! Results saved to {results_dir}")
    print("\nFiles created:")



if __name__ == "__main__":
    main()
