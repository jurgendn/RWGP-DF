# RWGP-DF (Dynamic Frontier Louvain + GP refinement)

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](requirements.txt)
[![Platform](https://img.shields.io/badge/platform-cross--platform-lightgrey)](#requirements)
[![Status](https://img.shields.io/badge/status-research%2Fexperimental-orange)](#whats-in-the-repo)

This repository contains a research-oriented implementation of **Dynamic Frontier Louvain** for **dynamic/temporal community detection**, plus a **GP-refined** variant ("GP-DF") and supporting baselines.

The primary workflow is **script-driven benchmarking/experimentation** over temporal edge updates (batch updates or time-windowed updates).

## What’s in the repo

Algorithms (in `src/models/`):

- `DynamicFrontierLouvain` (DF): incremental Louvain with an “affected frontier”.
- `GPDynamicFrontierLouvain` (GP-DF): DF + refinement using GP separators in `src/gp_df/`.
- `DeltaScreeningLouvain`: DF-style update with delta screening.
- `StaticLouvain`: recompute baseline.
- `NaiveDynamicLouvain`: naive dynamic baseline.

Experiment/benchmark entrypoints:

- `run_benchmarks.py`: benchmark sweep driven by `config/default.yaml`.
- `run.py`: synthetic-graph Optuna experiment (uses MLflow/Optuna).
- `run_college_msg_graph.py`, `run_bitcoin_alpha.py`, `run_bitcoin_otc.py`, `run_sx_mathoverflow.py`: dataset-specific Optuna/MLflow experiments.

## Requirements

Python **3.10+** is required (the codebase uses modern type syntax like `A | B`).

Install the base scientific stack:

```bash
pip install -r requirements.txt
```

For benchmarks + plots you will also typically need:

```bash
pip install pydantic tqdm pyyaml seaborn plotly wandb
```

For Optuna/MLflow experiment scripts (`run.py` and the `run_*.py` dataset scripts):

```bash
pip install optuna mlflow python-dotenv
```

## Datasets

This repo expects a local `dataset/` directory (it is gitignored). Place files there and point `config/default.yaml` at them.

Common filenames referenced by configs/scripts:

- `dataset/CollegeMsg.txt`
- `dataset/soc-sign-bitcoinalpha.csv`
- `dataset/soc-sign-bitcoinotc.csv`
- `dataset/sx-mathoverflow.txt`
- `dataset/email-Eu-core-temporal.txt`
- `dataset/sx-askubuntu.txt`
- `dataset/soc-redditHyperlinks-body.tsv`

## Run benchmarks (recommended starting point)

1) Edit `config/default.yaml`:

- Choose `mode`: `batch` or `window_frame`
- Set `target_datasets`
- Ensure each dataset entry has the right `dataset_path`, `source_idx`, `target_idx`, and (for `window_frame`) `timestamp_idx`

2) Run:

```bash
python run_benchmarks.py
```

Outputs are written under `results/` (also gitignored), by default:

- `results/<mode>_benchmark/<dataset_name>/...`

## Programmatic usage (minimal example)

```python
import networkx as nx

from src.data_loader import DatasetBatchManager
from src.models import DynamicFrontierLouvain

data_manager = DatasetBatchManager()

G, temporal_changes = data_manager.get_dataset(
    dataset_path="dataset/CollegeMsg.txt",
    dataset_type="college_msg",
    source_idx=0,
    target_idx=1,
    batch_range=0.005,
    initial_fraction=0.5,
    max_steps=10,
    load_full_nodes=True,
)

initial = nx.algorithms.community.louvain_communities(G, seed=42)
initial_partition = {node: cid for cid, community in enumerate(initial) for node in community}

model = DynamicFrontierLouvain(graph=G, initial_communities=initial_partition)
model.run([], [])  # “step 0”

for change in temporal_changes:
    metrics_by_name = model.run(change.deletions, change.insertions)
    df_metrics = metrics_by_name["DF Louvain"]
    print(df_metrics.modularity, df_metrics.runtime)
```

## Repository layout

```text
.
├── config/
│   ├── default.yaml
│   └── synthesis.yaml
├── consts/                 # Dataset-/experiment-specific constants (MLflow/Optuna, etc.)
├── docs/                   # Design notes, async status, refactor summary
├── src/
│   ├── benchmarks.py       # Runner + benchmark wiring
│   ├── components/         # Result schemas + temporal change objects
│   ├── data_loader/        # Batch + window-frame dataset loaders
│   ├── gp_df/              # GP separator refinement variants
│   ├── models/             # DF / GP-DF / baselines
│   └── utils/              # Plotting + helpers + MLflow logging
├── run_benchmarks.py
├── run.py
├── run_college_msg_graph.py
├── run_bitcoin_alpha.py
├── run_bitcoin_otc.py
├── run_sx_mathoverflow.py
└── requirements.txt
```

## Docs

- `docs/ARCHITECTURE.md` describes the module breakdown.
- `docs/REFACTORING_SUMMARY.md` contains historical refactor notes.

---

Last updated: 2026-01-07

<!--

# Dynamic Frontier Louvain Algorithm for Community Detection

A comprehensive Python implementation of the Dynamic Frontier Louvain algorithm for efficient community detection in dynamic networks. This project provides multiple implementations including synchronous, asynchronous, and specialized variants with extensive benchmarking capabilities.

## 🚀 Features

- **Dynamic Community Detection**: Efficiently handles edge insertions and deletions without full recomputation
- **Multiple Implementations**: Synchronous, asynchronous, and specialized versions (GP separator, delta screening)
- **High Performance**: Optimized implementations with frontier tracking for incremental updates
- **Comprehensive Benchmarking**: Built-in benchmarking suite with performance comparison and visualization
- **Real Dataset Support**: Includes loaders for temporal networks (College Message, Bitcoin Alpha/OTC, StackOverflow)
- **Advanced Visualization**: Matplotlib-based plotting with 6-panel analysis views
- **Export Capabilities**: CSV export for further analysis and research
- **Modular Architecture**: Clean separation of concerns with reusable components

## Features

- **Dynamic Community Detection**: Efficiently handles edge insertions and deletions
- **High Performance**: Optimized implementation with NetworkX integration
- **Asynchronous Processing**: Parallel execution for large-scale graphs
- **Comprehensive Benchmarking**: Built-in benchmarking suite with multiple metrics
- **Real Dataset Support**: Includes loaders for College Message and Bitcoin Alpha datasets
- **Advanced Visualization**: Matplotlib-based plotting for results analysis
- **Export Capabilities**: CSV export for further analysis

## 📁 Project Structure

```
df-improve/
├── src/                              # Main package directory
│   ├── __init__.py                  # Package initialization
│   ├── df_louvain.py               # Main entry point with imports
│   ├── df_louvain_sync.py          # Synchronous implementation
│   ├── df_louvain_async.py         # Asynchronous implementation
│   ├── df_louvain_sync_separate.py # GP separator version
│   ├── community_info.py           # Core data structures and utilities
│   ├── benchmarks.py               # Benchmarking and evaluation tools
│   ├── naive_dynamic.py            # Naive dynamic baseline
│   ├── delta_screening.py          # Delta screening variant
│   ├── gp_separator.py             # GP separator utilities
│   ├── data_loader/                # Dataset loading utilities
│   │   ├── __init__.py
│   │   ├── data_manager.py         # Dataset management
│   │   ├── batch_loader.py         # Batch loading functions
│   │   └── window_loader.py        # Window-based loading
│   ├── refining/                   # Community refinement algorithms
│   │   ├── gp_separator_v1.py      # Version 1 with K-means
│   │   ├── gp_separator_v2.py      # Version 2 with random walk
│   │   ├── gp_separator_v3.py      # Version 3 optimized
│   │   ├── gp_separator_v4.py      # Version 4 ultra-fast
│   │   └── separator.py            # Base separator
│   └── utils/
│       └── helpers.py              # Utility functions
├── dataset/                        # Datasets directory
│   ├── CollegeMsg.txt             # College Message temporal network
│   ├── soc-sign-bitcoinalpha.csv # Bitcoin Alpha trust network
│   ├── soc-sign-bitcoinotc.csv   # Bitcoin OTC trust network
│   ├── sx-askubuntu.txt           # StackOverflow Ask Ubuntu
│   ├── sx-mathoverflow.txt        # StackOverflow Math Overflow
│   └── synthesis/                 # Synthetic datasets
├── docs/                          # Documentation
│   ├── ARCHITECTURE.md            # Project architecture guide
│   ├── ASYNC_BENCHMARKING_GUIDE.md
│   ├── ASYNC_STATUS_REPORT.md
│   └── REFACTORING_SUMMARY.md
├── notebooks/                     # Jupyter notebooks for exploration
├── results/                       # Generated benchmark results
├── config/                        # Configuration files
│   ├── default.yaml
│   └── synthesis.yaml
├── test/                          # Test files
├── run_benchmarks.py             # Main benchmarking script
├── examples.py                   # Usage examples
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```
## 🛠️ Installation

### Prerequisites
- Python 3.8+
- NetworkX 3.0+
- NumPy, Pandas, Matplotlib, SciPy

### Quick Install
```bash
# Clone the repository
git clone <repository-url>
cd df-improve

# Install dependencies
pip install -r requirements.txt

# Verify installation
python examples.py
```

### Dependencies
```bash
pip install networkx>=3.0 numpy>=1.21.0 pandas>=1.3.0 matplotlib>=3.5.0 scipy>=1.7.0
```

## 🚀 Quick Start

### Basic Usage

```python
import networkx as nx
from src.df_louvain import DynamicFrontierLouvain

# Create or load a graph
G = nx.karate_club_graph()

# Initialize the algorithm
df_louvain = DynamicFrontierLouvain(G, tolerance=1e-3, verbose=True)

# Run initial community detection
communities = df_louvain.run_dynamic_frontier_louvain()
print(f"Found {len(set(communities.values()))} communities")

# Apply dynamic changes
edge_deletions = [(0, 1), (2, 3)]
edge_insertions = [(0, 5, 1.0)]

# Update communities incrementally
updated_communities = df_louvain.run_dynamic_frontier_louvain(
    edge_deletions, edge_insertions
)

# Get performance metrics
modularity = df_louvain.get_modularity()
affected_nodes = df_louvain.get_affected_nodes()
print(f"Modularity: {modularity:.4f}, Affected nodes: {len(affected_nodes)}")
```

### Asynchronous Usage

```python
import asyncio
from src.df_louvain import AsyncDynamicFrontierLouvain

async def run_async_example():
    # Initialize async version
    async_df = AsyncDynamicFrontierLouvain(tolerance=1e-3, verbose=True)
    
    # Run async community detection
    community_info = await async_df.dynamic_frontier_louvain(
        graph=G,
        edge_deletions=edge_deletions,
        edge_insertions=edge_insertions
    )
    
    return community_info

# Run async version
result = asyncio.run(run_async_example())
```

### Run Examples

```bash
# Run basic examples
python examples.py

# Run comprehensive benchmarks
python run_benchmarks.py

# Run with specific datasets
python run_benchmarks.py --dataset college_msg --verbose
```

## 📊 Datasets

### Supported Datasets

#### College Message Dataset (`CollegeMsg.txt`)
- **Format**: `node1 node2 timestamp`
- **Description**: Temporal network of messages between university students
- **Size**: ~1,900 nodes, ~20,000 temporal edges
- **Source**: Social network analysis research

#### Bitcoin Alpha/OTC Datasets
- **Format**: `source,target,rating,timestamp`
- **Description**: Bitcoin trust networks with temporal ratings
- **Bitcoin Alpha**: Trust network with user ratings over time
- **Bitcoin OTC**: Over-the-counter trading trust network

#### StackOverflow Networks
- **Format**: `node1 node2 timestamp`
- **Description**: User interaction networks from Stack Exchange sites
- **Ask Ubuntu**: Ubuntu-related questions and answers
- **Math Overflow**: Mathematics Q&A interactions

### Dataset Loading

```python
from src.data_loader import (
    load_college_msg_dataset,
    load_bitcoin_dataset,
    create_synthetic_dynamic_graph
)

# Load real datasets
graph, temporal_data = load_college_msg_dataset("dataset/CollegeMsg.txt")
btc_graph, btc_data = load_bitcoin_dataset("dataset/soc-sign-bitcoinalpha.csv")

# Create synthetic data
syn_graph, syn_changes = create_synthetic_dynamic_graph(
    num_nodes=100, initial_edges=200, time_steps=10
)
```

## 🔬 Algorithm Details

### Core Algorithm Components

1. **Dynamic Frontier Tracking**: Only processes nodes affected by graph changes
2. **Incremental Updates**: Efficiently handles edge insertions/deletions
3. **Modularity Optimization**: Maintains high-quality community structures
4. **Multi-Pass Refinement**: Iterative improvement with convergence detection

### Implementation Variants

#### 1. Synchronous Implementation (`DynamicFrontierLouvain`)
- **Best for**: Small to medium graphs (<1000 nodes)
- **Features**: Memory efficient, fast convergence
- **Use cases**: Real-time applications, interactive analysis

#### 2. Asynchronous Implementation (`AsyncDynamicFrontierLouvain`)
- **Best for**: Large graphs (>1000 nodes)
- **Features**: Parallel processing, concurrent vertex optimization
- **Use cases**: Batch processing, large-scale analysis

#### 3. GP Separator Version (`GPDynamicFrontierLouvain`)
- **Best for**: Community refinement applications
- **Features**: Advanced community splitting with random walk
- **Use cases**: High-quality community detection

#### 4. Specialized Variants
- **Delta Screening**: Modularity-based vertex screening
- **Naive Dynamic**: Baseline without frontier optimization

### Key Performance Features

- **Time Complexity**: O(m) per dynamic update (m = affected edges)
- **Space Complexity**: O(n + m) for n nodes and m edges
- **Dynamic Efficiency**: 2-10x faster than full recomputation
- **Scalability**: Handles graphs with thousands of nodes efficiently

## 📈 Benchmarking

### Built-in Benchmarking Suite

The project includes comprehensive benchmarking capabilities:

```python
from src.benchmarks import DFLouvainBenchmark

# Initialize benchmark
benchmark = DFLouvainBenchmark()

# Run dataset comparison
results = benchmark.benchmark_dataset(
    dataset_name="college_msg",
    dataset_path="dataset/CollegeMsg.txt",
    dataset_type="college_msg"
)

# Generate visualizations
benchmark.plot_results("college_msg", save_path="results/")

# Export results
benchmark.export_results("results/benchmark_results.csv")
```

### Performance Metrics

- **Modularity Evolution**: Community quality over time
- **Runtime Analysis**: Per-step execution time
- **Affected Nodes**: Frontier size tracking
- **Memory Usage**: Resource consumption patterns
- **Convergence Rate**: Algorithm stability metrics

### Visualization Outputs

- **6-Panel Analysis Plots**: Comprehensive performance overview
- **Temporal Evolution**: Community changes over time
- **Comparison Charts**: Algorithm performance comparisons
- **Export Formats**: PNG plots, CSV data

## 🔧 API Reference

### Core Classes

#### `DynamicFrontierLouvain`
```python
class DynamicFrontierLouvain:
    def __init__(self, graph, initial_communities=None, tolerance=1e-2, 
                 max_iterations=20, verbose=False)
    def run_dynamic_frontier_louvain(self, edge_deletions=None, edge_insertions=None)
    def apply_batch_update(self, edge_deletions=None, edge_insertions=None)
    def get_modularity(self) -> float
    def get_communities(self) -> Dict[int, Set]
    def get_affected_nodes(self) -> List[int]
```

#### `AsyncDynamicFrontierLouvain`
```python
class AsyncDynamicFrontierLouvain:
    def __init__(self, tolerance=1e-2, max_iterations=20, max_passes=10, verbose=True)
    async def dynamic_frontier_louvain(self, graph, edge_deletions=None, 
                                     edge_insertions=None, previous_communities=None)
```

#### `CommunityInfo` (Data Structure)
```python
@dataclass
class CommunityInfo:
    vertex_degrees: Dict[int, float]      # Weighted degrees
    community_weights: Dict[int, float]   # Community total weights
    community_assignments: Dict[int, int] # Node-to-community mapping
```

### Utility Functions

#### Data Loading
```python
def load_college_msg_dataset(file_path: str) -> Tuple[nx.Graph, List[Dict]]
def load_bitcoin_dataset(file_path: str) -> Tuple[nx.Graph, List[Dict]]
def create_synthetic_dynamic_graph(num_nodes=100, initial_edges=200, time_steps=10)
```

#### Community Analysis
```python
def calculate_modularity(graph: nx.Graph, communities: Dict[int, int]) -> float
def calculate_weighted_degrees(graph: nx.Graph) -> Dict[int, float]
def get_neighbor_communities(graph: nx.Graph, node: int, communities: Dict[int, int])
```

## ⚙️ Configuration

### Algorithm Parameters

- **tolerance**: Convergence threshold (default: 1e-2)
- **max_iterations**: Maximum local moving iterations (default: 20)
- **max_passes**: Maximum aggregation passes (default: 10)
- **verbose**: Enable detailed logging (default: False)

### Benchmarking Configuration

Edit `config/default.yaml` for custom benchmarking settings:

```yaml
algorithms:
  - "Dynamic Frontier Louvain"
  - "Naive Dynamic Louvain"
  - "Delta Screening Louvain"

datasets:
  college_msg:
    path: "dataset/CollegeMsg.txt"
    type: "temporal"
  
metrics:
  - "modularity"
  - "runtime"
  - "affected_nodes"
```

## 🧪 Testing and Validation

### Running Tests

```bash
# Run basic tests
python test/test_dataloader.py

# Run algorithm validation
python examples.py

# Benchmark validation
python run_benchmarks.py --quick
```

### Validation Metrics

- **Modularity Consistency**: Ensure stable community quality
- **Performance Regression**: Validate optimization improvements
- **Memory Efficiency**: Monitor resource usage
- **Convergence Behavior**: Verify algorithm stability

## 📚 Research and Citations

### Academic Background

This implementation is based on the seminal work:

> Blondel, V. D., Guillaume, J.-L., Lambiotte, R., & Lefebvre, E. (2008). 
> Fast unfolding of communities in large networks. Journal of Statistical 
> Mechanics: Theory and Experiment, 2008(10), P10008.

### Citing This Work

```bibtex
@misc{dynamic_frontier_louvain_2025,
  title={Dynamic Frontier Louvain: Efficient Community Detection in Dynamic Networks},
  author={Implementation based on Blondel et al.},
  year={2025},
  note={Python implementation with async support and comprehensive benchmarking},
  url={<repository-url>}
}
```

### Related Research Areas

- **Community Detection**: Network analysis and social networks
- **Dynamic Networks**: Temporal graph analysis
- **Modularity Optimization**: Community quality metrics
- **Parallel Algorithms**: Asynchronous graph processing

## 🤝 Contributing

### Development Guidelines

1. **Code Style**: Follow PEP 8 conventions
2. **Documentation**: Add docstrings for all public methods
3. **Testing**: Include tests for new features
4. **Benchmarking**: Validate performance improvements

### Contributing Process

```bash
# Fork the repository
git fork <repository-url>

# Create feature branch
git checkout -b feature/new-algorithm

# Make changes and test
python examples.py
python run_benchmarks.py

# Submit pull request
git push origin feature/new-algorithm
```

### Areas for Contribution

- **New Algorithms**: Additional community detection variants
- **Dataset Support**: New temporal network loaders
- **Performance Optimization**: Algorithm efficiency improvements
- **Visualization**: Enhanced plotting and analysis tools

## 📄 License

This project is open source under the MIT License. See `LICENSE` file for details.

## 📞 Support

### Getting Help

- **Issues**: Report bugs and request features via GitHub Issues
- **Documentation**: Check `docs/` directory for detailed guides
- **Examples**: See `examples.py` and `notebooks/` for usage patterns

### Troubleshooting

#### Common Issues

1. **NetworkX Compatibility**: Ensure NetworkX >= 3.0
2. **Memory Issues**: Use async version for large graphs
3. **Convergence Problems**: Adjust tolerance and max_iterations
4. **Performance**: Enable verbose mode for debugging

#### Performance Optimization Tips

- Use synchronous version for graphs < 1000 nodes
- Enable caching for repeated dataset loading
- Adjust chunk size for async processing
- Monitor memory usage with large datasets

---

**Last Updated**: July 2025
**Version**: 2.0
**Python Compatibility**: 3.8+

-->
