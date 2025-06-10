"""
Main script to run DFLouvain benchmarks on the provided datasets.
"""

import asyncio
import os
from pathlib import Path

from src.benchmarks import run_comprehensive_benchmark, run_sync_vs_async_comparison

current_dir = "."


def main():
    """
    Run comprehensive benchmarks on the College Message, Bitcoin Alpha, and Bitcoin OTC datasets.
    """
    print("Dynamic Frontier Louvain Benchmarking Suite")
    print("=" * 50)

    batch_range = 1e-4
    initial_fraction = 0.85

    # Dataset paths
    dataset_dir = os.path.join(current_dir, "dataset")
    dataset_paths = {
        "CollegeMsg": os.path.join(dataset_dir, "CollegeMsg.txt"),
        "BitcoinAlpha": os.path.join(dataset_dir, "soc-sign-bitcoinalpha.csv"),
        "BitcoinOTC": os.path.join(dataset_dir, "soc-sign-bitcoinotc.csv"),
        "SX - MathOverFlow": os.path.join(dataset_dir, "sx-mathoverflow.txt"),
    }

    # Verify datasets exist
    for name, path in dataset_paths.items():
        if not os.path.exists(path):
            print(f"WARNING: Dataset {name} not found at {path}")
            continue

    # Run benchmarks
    benchmark = run_comprehensive_benchmark(
        dataset_paths=dataset_paths,
        batch_range=batch_range,
        initial_fraction=initial_fraction,
    )

    # Export results
    results_dir = Path(os.path.join(current_dir, "results", f"{batch_range}_{initial_fraction}"))
    results_dir.mkdir(exist_ok=True)

    # Export summary CSV
    benchmark.export_results(os.path.join(results_dir, "benchmark_results.csv"))
    # Create plots for each dataset
    for dataset_name in dataset_paths.keys():
        if dataset_name in benchmark.results:
            plot_path = os.path.join(results_dir, f"{dataset_name}_benchmark_plots.png")
            benchmark.plot_results(dataset_name, str(plot_path))

    print(f"\nBenchmark complete! Results saved to {results_dir}")
    print("\nFiles created:")
    print("  - benchmark_results.csv: Summary statistics")
    for dataset_name in dataset_paths.keys():
        if dataset_name in benchmark.results:
            print(f"  - {dataset_name}_benchmark_plots.png: Visualization plots")


async def async_main():
    """
    Run comprehensive benchmarks on the College Message, Bitcoin Alpha, and Bitcoin OTC datasets.
    """
    print("Dynamic Frontier Louvain Benchmarking Suite")
    print("=" * 50)

    batch_range = 1e-4

    # Dataset paths
    dataset_dir = os.path.join(current_dir, "dataset")
    dataset_paths = {
        "CollegeMsg": os.path.join(dataset_dir, "CollegeMsg.txt"),
        "BitcoinAlpha": os.path.join(dataset_dir, "soc-sign-bitcoinalpha.csv"),
        "BitcoinOTC": os.path.join(dataset_dir, "soc-sign-bitcoinotc.csv"),
        "SX - MathOverFlow": os.path.join(dataset_dir, "sx-mathoverflow.txt"),
    }

    # Verify datasets exist
    for name, path in dataset_paths.items():
        if not os.path.exists(path):
            print(f"WARNING: Dataset {name} not found at {path}")
            continue

    # Run benchmarks
    benchmark = await run_sync_vs_async_comparison(
        dataset_paths=dataset_paths, batch_range=batch_range
    )

    # Export results
    results_dir = Path(os.path.join(current_dir, "results"))
    results_dir.mkdir(exist_ok=True)

    # Export summary CSV
    benchmark.export_results(os.path.join(results_dir, "benchmark_results.csv"))
    # Create plots for each dataset
    for dataset_name in dataset_paths.keys():
        if dataset_name in benchmark.results:
            plot_path = os.path.join(results_dir, f"{dataset_name}_benchmark_plots.png")
            benchmark.plot_results(dataset_name, str(plot_path))

    print(f"\nBenchmark complete! Results saved to {results_dir}")
    print("\nFiles created:")
    print("  - benchmark_results.csv: Summary statistics")
    for dataset_name in dataset_paths.keys():
        if dataset_name in benchmark.results:
            print(f"  - {dataset_name}_benchmark_plots.png: Visualization plots")


if __name__ == "__main__":
    main()
    # asyncio.run(async_main())
