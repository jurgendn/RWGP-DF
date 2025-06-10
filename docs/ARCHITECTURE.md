# Dynamic Frontier Louvain - Architecture Guide

## Project Overview

This project implements the Dynamic Frontier Louvain algorithm for community detection in dynamic networks, providing both synchronous and asynchronous implementations with comprehensive benchmarking capabilities.

## Module Architecture

### Core Algorithm Modules

#### `src/community_info.py`
- **Purpose**: Core data structures and utility functions
- **Components**:
  - `CommunityInfo`: Dataclass for storing community state
  - `CommunityUtils`: Static methods for community operations
- **Key Functions**:
  - Weighted degree calculations
  - Community weight calculations
  - Modularity calculations
  - Community initialization

#### `src/df_louvain_sync.py`
- **Purpose**: Synchronous implementation of Dynamic Frontier Louvain
- **Main Class**: `DynamicFrontierLouvain`
- **Features**:
  - Optimized single-threaded implementation
  - Efficient local moving and aggregation phases
  - Dynamic frontier tracking for incremental updates
  - Memory-efficient community management

#### `src/df_louvain_async.py`
- **Purpose**: Asynchronous implementation for parallel processing
- **Main Class**: `AsyncDynamicFrontierLouvain`
- **Features**:
  - Asynchronous implementation using `asyncio`
  - Concurrent processing of vertex chunks
  - Parallel batch update handling
  - Scalable for large graphs

### Support Modules

#### `src/data_loader.py`
- **Purpose**: Dataset loading and preprocessing utilities
- **Functions**:
  - `load_college_msg_dataset()`: Load College Message dataset
  - `load_bitcoin_dataset()`: Load Bitcoin trust networks
  - `create_synthetic_dynamic_graph()`: Generate test data

#### `src/benchmarks.py`
- **Purpose**: Comprehensive benchmarking and evaluation
- **Main Class**: `DFLouvainBenchmark`
- **Features**:
  - Performance comparison (DF vs NetworkX)
  - Dynamic performance analysis
  - Async vs sync comparison
  - Scalability analysis
  - Visualization and export capabilities

#### `src/df_louvain.py`
- **Purpose**: Main entry point and compatibility layer
- **Function**: Re-exports main classes for easy import

## Data Flow

```
Graph Input → Data Loader → DF Louvain Algorithm → Community Output
                ↓
            Benchmarks → Performance Metrics → Visualizations/Reports
```

## Algorithm Workflow

1. **Initialization**: Load graph and initialize community structures
2. **Initial Detection**: Run initial community detection
3. **Dynamic Updates**: Process edge insertions/deletions
4. **Frontier Tracking**: Track affected nodes for efficient updates
5. **Optimization**: Local moving and aggregation phases
6. **Modularity Calculation**: Assess community quality

## Performance Characteristics

### Synchronous Implementation
- **Best for**: Small to medium graphs (<1000 nodes)
- **Memory**: Low overhead
- **Speed**: Fast for incremental updates
- **Use cases**: Real-time applications, resource-constrained environments

### Asynchronous Implementation
- **Best for**: Large graphs (>1000 nodes)
- **Memory**: Higher overhead due to async structures
- **Speed**: Benefits from parallelization on large datasets
- **Use cases**: Batch processing, large-scale analysis

## Integration Points

### Benchmarking Integration
- Automatic dataset type detection
- Configurable performance metrics
- Export capabilities (CSV, PNG)
- Comparison visualizations

### External Dependencies
- **NetworkX**: Graph structures and reference implementations
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation and export
- **Matplotlib**: Visualization
- **asyncio**: Asynchronous programming

## Extensibility

The modular architecture supports:
- **New algorithms**: Add to core modules
- **Additional datasets**: Extend data loaders
- **Custom metrics**: Enhance benchmarking
- **Alternative implementations**: Parallel module structure

## Error Handling

- Robust community detection fallbacks
- NetworkX compatibility handling
- Async concurrency error management
- Graph connectivity validation
