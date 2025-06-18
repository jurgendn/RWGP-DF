"""
Main script to run DFLouvain benchmarks on the provided datasets.
"""

import os
from pathlib import Path

import yaml

from src.benchmarks import run_comprehensive_benchmark
from src.data_loader import DatasetBatchManager, DatasetWindowTimeManager


def main():
    """
    Run comprehensive benchmarks on the College Message, Bitcoin Alpha, and Bitcoin OTC datasets.
    """

    print("Load configuration")
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
    print("Dynamic Frontier Louvain Benchmarking Suite")
    print("=" * 50)


    # Run benchmarks
    benchmark = run_comprehensive_benchmark(
        data_manager=data_manager,
        target_datasets=target_datasets,
        config=data_config,
    )

    # Export results
    results_dir = Path(os.path.join(os.curdir, "results", f"{mode}_benchmark"))
    results_dir.mkdir(exist_ok=True)

    # Export summary CSV
    benchmark.export_results(os.path.join(results_dir, "benchmark_results.csv"))
    # Create plots for each dataset
    for dataset_name, dataset_config in data_config.items():
        if dataset_name in benchmark.results and dataset_name in target_datasets:
            batch_range = dataset_config.get("batch_range")
            initial_fraction = dataset_config.get("initial_fraction")
            # Create plot for the dataset
            plot_path = os.path.join(
                results_dir,
                f"{dataset_name}_batch_range_{batch_range}_initial_fraction_{initial_fraction}_benchmark_plot.png",
            )
            benchmark.plot_results(dataset_name, str(plot_path))

    print(f"\nBenchmark complete! Results saved to {results_dir}")
    print("\nFiles created:")
    print("  - benchmark_results.csv: Summary statistics")
    for dataset_name in data_config.keys():
        if dataset_name in benchmark.results:
            print(f"  - {dataset_name}_benchmark_plots.png: Visualization plots")


if __name__ == "__main__":
    main()
