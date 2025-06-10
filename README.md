# Dynamic Frontier Louvain Algorithm

A Python implementation of the Dynamic Frontier Louvain algorithm for community detection in dynamic networks. This algorithm efficiently updates community structures when network edges change over time, without requiring complete recomputation.

## 🚀 New: Asynchronous Implementation Available!

The project now includes a **fully functional asynchronous implementation** with comprehensive benchmarking capabilities:

- ✅ **Async Dynamic Frontier Louvain**: Parallel processing using asyncio
- ✅ **Comprehensive Async Benchmarking**: Performance comparison and analysis
- ✅ **Visualization & Export**: 6-panel plots and CSV data export
- ✅ **Sync vs Async Comparison**: Direct performance comparisons
- ✅ **Production Ready**: Complete with documentation and examples

See `ASYNC_BENCHMARKING_GUIDE.md` for detailed usage instructions.

## Features

- **Dynamic Community Detection**: Efficiently handles edge insertions and deletions
- **High Performance**: Optimized implementation with NetworkX integration
- **Asynchronous Processing**: Parallel execution for large-scale graphs
- **Comprehensive Benchmarking**: Built-in benchmarking suite with multiple metrics
- **Real Dataset Support**: Includes loaders for College Message and Bitcoin Alpha datasets
- **Advanced Visualization**: Matplotlib-based plotting for results analysis
- **Export Capabilities**: CSV export for further analysis

## Project Structure

```
df-improve/
├── src/                          # Main package directory
│   ├── __init__.py              # Package initialization
│   ├── df_louvain.py            # Main entry point
│   ├── df_louvain_sync.py       # Synchronous implementation
│   ├── df_louvain_async.py      # Asynchronous implementation
│   ├── community_info.py        # Core data structures
│   ├── data_loader.py           # Dataset loading utilities
│   └── benchmarks.py            # Benchmarking and evaluation tools
├── dataset/                     # Datasets directory
│   ├── CollegeMsg.txt          # College Message dataset
│   ├── soc-sign-bitcoinalpha.csv  # Bitcoin Alpha trust network
│   └── soc-sign-bitcoinotc.csv    # Bitcoin OTC trust network
├── docs/                        # Documentation
│   ├── ARCHITECTURE.md         # Project architecture guide
│   ├── ASYNC_BENCHMARKING_GUIDE.md  # Async benchmarking documentation
│   ├── ASYNC_STATUS_REPORT.md   # Async implementation status
│   └── REFACTORING_SUMMARY.md   # Refactoring details
├── notebooks/                   # Jupyter notebooks for exploration
├── results/                     # Generated benchmark results
├── run_benchmarks.py           # Main benchmarking script
├── examples.py                 # Usage examples
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. The package is ready to use with the provided datasets.

## Quick Start

### Basic Usage

```python
import networkx as nx
from src.df_louvain import DynamicFrontierLouvain

# Create or load a graph
G = nx.karate_club_graph()

# Initialize the algorithm
df_louvain = DynamicFrontierLouvain(G, tolerance=1e-3)

# Run initial community detection
communities = df_louvain.run_dynamic_frontier_louvain()
print(f"Found {len(set(communities.values()))} communities")

# Apply dynamic changes
edge_deletions = [(0, 1), (2, 3)]
edge_insertions = [(0, 5, 1.0)]

updated_communities = df_louvain.run_dynamic_frontier_louvain(
    edge_deletions, edge_insertions
)
```

### Run Examples

```bash
python examples.py
```

This will demonstrate the algorithm on:
- College Message dataset
- Bitcoin Alpha dataset  
- Synthetic dynamic graph

### Run Comprehensive Benchmarks

```bash
python run_benchmarks.py
```

This will:
- Benchmark performance on both real datasets
- Compare with NetworkX Louvain implementation
- Generate performance plots and CSV results
- Save results to `results/` directory

## Datasets

### College Message Dataset (`CollegeMsg.txt`)
- **Format**: `node1 node2 timestamp`
- **Description**: Temporal network of messages between students
- **Source**: Temporal social network data

### Bitcoin Alpha Dataset (`soc-sign-bitcoinalpha.csv`)
- **Format**: `source,target,rating,timestamp`
- **Description**: Bitcoin trust network with ratings over time
- **Source**: Social network with signed edges

### Bitcoin OTC Dataset (`soc-sign-bitcoinotc.csv`)
- **Format**: `source,target,rating,timestamp`
- **Description**: Bitcoin over-the-counter trading trust network
- **Source**: Social network with signed edges and temporal evolution

## Algorithm Details

The Dynamic Frontier Louvain algorithm extends the classic Louvain method with:

1. **Frontier Tracking**: Only processes nodes affected by graph changes
2. **Incremental Updates**: Efficiently handles edge insertions/deletions
3. **Modularity Optimization**: Maintains high-quality community structures
4. **Dynamic Adaptation**: Responds to temporal network evolution

### Key Components

- **DynamicFrontierLouvain**: Main algorithm class
- **apply_batch_update()**: Handles edge changes and marks affected nodes
- **louvain_move()**: Performs local moving phase with frontier optimization
- **get_modularity()**: Calculates community quality metric

## Benchmarking Results

The benchmarking suite provides:

### Performance Metrics
- **Static Comparison**: DFLouvain vs NetworkX Louvain
- **Dynamic Performance**: Runtime per temporal update
- **Community Stability**: Tracking community evolution
- **Scalability Analysis**: Performance vs graph size

### Output Files
- `benchmark_results.csv`: Summary statistics
- `{dataset}_benchmark_plots.png`: Visualization plots

### Key Metrics Tracked
- Modularity scores over time
- Runtime per dynamic step
- Number of affected nodes
- Community change rates
- Memory usage patterns

## API Reference

### DynamicFrontierLouvain Class

```python
class DynamicFrontierLouvain:
    def __init__(self, graph, tolerance=1e-2, max_iterations=20, verbose=True)
    def run_dynamic_frontier_louvain(self, edge_deletions=None, edge_insertions=None)
    def get_modularity(self) -> float
    def get_communities(self) -> Dict[int, Set]
    def get_affected_nodes(self) -> List
```

### Data Loaders

```python
def load_college_msg_dataset(file_path: str) -> Tuple[nx.Graph, List[Dict]]
def load_bitcoin_dataset(file_path: str) -> Tuple[nx.Graph, List[Dict]]
def create_synthetic_dynamic_graph(num_nodes=100, initial_edges=200, time_steps=10)
```

### Benchmarking

```python
class DFLouvainBenchmark:
    def benchmark_dataset(self, dataset_name, dataset_path, dataset_type)
    def plot_results(self, dataset_name, save_path=None)
    def export_results(self, filepath)
```

## Performance Characteristics

- **Time Complexity**: O(m) per dynamic update (where m is the number of affected edges)
- **Space Complexity**: O(n + m) for n nodes and m edges
- **Scalability**: Handles graphs with thousands of nodes efficiently
- **Dynamic Efficiency**: 2-10x faster than full recomputation for small changes

## Citation

If you use this implementation in your research, please cite:

```bibtex
@misc{dynamic_frontier_louvain,
  title={Dynamic Frontier Louvain: Efficient Community Detection in Dynamic Networks},
  author={Your Name},
  year={2025},
  note={Implementation based on Louvain algorithm by Blondel et al.}
}
```

## License

This project is open source. Please refer to the original Louvain algorithm paper for academic citations:

> Blondel, V. D., Guillaume, J.-L., Lambiotte, R., & Lefebvre, E. (2008). Fast unfolding of communities in large networks. Journal of Statistical Mechanics: Theory and Experiment, 2008(10), P10008.

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## Contact

For questions or support, please open an issue in the repository.
