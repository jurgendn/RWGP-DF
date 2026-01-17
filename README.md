# RWGP-DF-Louvain: Dynamic Frontier Louvain + Random Walk Refinement

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](requirements.txt)
[![Status](https://img.shields.io/badge/status-research%2Fexperimental-orange)](#status)
[![PDF](https://img.shields.io/badge/PDF-view-orange)](https://drive.google.com/file/d/1NAmaJF7buBkKZmAcY181DCdm3XpCNvRN/view?usp=sharing)

Research-oriented implementation of **DF-Louvain (Dynamic Frontier Louvain)** for **dynamic community detection**, augmented with a **random walk graph partition (RWGP) refinement** step that can **split communities** after deletions when modularity improves.

This repo is script-driven: run benchmarks over temporal edge updates (batch updates or time-windowed updates), compare DF vs refined DF, and export plots.

## Why RWGP-DF?

DF-Louvain is efficient for evolving graphs because it updates only an “affected frontier” after edge insertions/deletions. A key limitation (highlighted in the paper) is that pure DF-style updates naturally favor **merges / local adjustments**, and may fail to **split** a community when internal connectivity weakens after deletions.

**RWGP-DF-Louvain** addresses this by adding a lightweight refinement step that proposes **binary splits** via a short random walk inside candidate communities and **accepts a split only if modularity increases**.

## Algorithm (high level)

For each update step (edge insertions + deletions):

1. **Frontier update (DF-Louvain):** apply the batch update and run local Louvain moves restricted to affected nodes.
2. **Build refinement set:** identify communities impacted by intra-community deletions (implementation: communities touched by deleted edges inside the same community / affected frontier).
3. **Random-walk refinement (bisection):** for each candidate community, compute a short random walk distribution and split vertices based on deviation from the stationary distribution; accept the split only if it improves modularity.

Complexity matches the paper’s intent: the refinement cost is proportional to the total edges in refined communities, multiplied by a small walk length $t$.

## Mathematical model (RW refinement)

For a candidate community subgraph with adjacency matrix $A$ and degree diagonal $D$, the random walk uses the row-stochastic transition matrix:

$$P = D^{-1}A$$

Let $d_i$ be the (weighted) degree of node $i$ inside the subgraph. The stationary distribution is:

$$\phi_i = \frac{d_i}{\sum_j d_j}$$

Starting from a source node $s$, define the $t$-step walk distribution over vertices:

$$p^{(t)} = e_s^\top P^t$$

The initial bisection is obtained by comparing $p^{(t)}$ to $\phi$ (the paper describes this as a deviation-from-stationarity split; the repo variants implement it as a degree-proportional threshold):

$$V_1 = \{ i : p^{(t)}_i \ge \phi_i \},\quad V_2 = \{ i : p^{(t)}_i < \phi_i \}$$

### Split acceptance (modularity gain)

The refinement only accepts a split if it improves modularity. In the fast RW splitter (v5), the modularity change for splitting a community into $S_1, S_2$ is tested via:

$$\Delta Q_{\text{split}} = -\frac{e(S_1,S_2)}{2m} + \gamma\,\frac{\mathrm{vol}(S_1)\,\mathrm{vol}(S_2)}{(2m)^2}$$

where $m$ is total (weighted) edge weight in the full graph, $e(S_1,S_2)$ is the cut weight between the two sides, and $\mathrm{vol}(S)=\sum_{i\in S} d_i$. The split is accepted when $\Delta Q_{\text{split}} > \varepsilon$ (and both sides satisfy minimum size constraints).

## How this maps to the codebase

Names in the repo reflect iteration history:

- **DF-Louvain** implementation: `DynamicFrontierLouvain` in `src/models/df_louvain.py`.
- **RWGP-DF-Louvain** implementation: `GPDynamicFrontierLouvain` in `src/models/gp_df_louvain.py`.
  - Despite the “GP” name, the refinement implementations in `src/gp_df/` include **random-walk-based splitters**.
  - The refinement implementation is selected by `refine_version`.

Refinement variants (see `src/gp_df/__init__.py`):

- `refine_version="v2-full"`: dense-matrix random walk split + modularity check.
- `refine_version="v5"`: sparse random walk proposer + fast modularity gain test (recommended default in the Optuna scripts).

Baselines in `src/models/`:

- `StaticLouvain`: recompute baseline
- `NaiveDynamicLouvain`: naive dynamic baseline
- `DeltaScreeningLouvain`: DF-style update with delta screening

## Requirements

- Python **3.10+**

Install core dependencies:

```bash
pip install -r requirements.txt
```

For running the benchmark scripts and plots:

```bash
pip install pyyaml tqdm seaborn plotly wandb
```

For the Optuna/MLflow experiment scripts (`run.py`, `run_bitcoin_*.py`, etc.):

```

## Datasets

Create a local `dataset/` directory (gitignored) and place dataset files there. The benchmark config expects paths like:
- `dataset/soc-sign-bitcoinalpha.csv`
- `dataset/soc-sign-bitcoinotc.csv`
- `dataset/sx-mathoverflow.txt`

The loaders live in `src/data_loader/` and support both **batch updates** and **window-frame** updates.


Benchmarks are driven by `config/default.yaml`.

1) Edit `config/default.yaml`:

- Set `mode`: `batch` or `window_frame`
- Choose `target_datasets`
- Verify dataset file paths + column indices (`source_idx`, `target_idx`, and for window mode also `timestamp_idx`)

2) Run:

```bash
python run_benchmarks.py
```

Outputs:

- Plots are written under `results/<mode>_benchmark/<dataset_name>/...` (the `results/` directory is gitignored).

## Run paper-style experiments (Optuna/MLflow)

The dataset scripts (`run_bitcoin_alpha.py`, `run_bitcoin_otc.py`, `run_college_msg_graph.py`, `run_sx_mathoverflow.py`) run Optuna sweeps and log to MLflow.

Notes:

- The scripts call `load_dotenv(".env")`. If you want a custom MLflow backend, create `.env` and set `MLFLOW_TRACKING_URI` (or adjust the constants in `consts/`).
- Most scripts instantiate `GPDynamicFrontierLouvain(..., refine_version="v5")`.

## Programmatic usage (minimal example)

```python
import networkx as nx

data_manager = DatasetBatchManager()

G, temporal_changes = data_manager.get_dataset(
    dataset_path="dataset/CollegeMsg.txt",
    dataset_type="college_msg",
    source_idx=0,
    target_idx=1,
    batch_range=0.005,
    max_steps=10,
    load_full_nodes=True,
)

initial = nx.algorithms.community.louvain_communities(G, seed=42)
initial_partition = {node: cid for cid, comm in enumerate(initial) for node in comm}

df = DynamicFrontierLouvain(graph=G, initial_communities=initial_partition, verbose=False)
rwgp_df = GPDynamicFrontierLouvain(
    graph=G,
    initial_communities=initial_partition,
    refine_version="v5",  # RW-based refinement
    verbose=False,
)
    df_metrics = df.run(change.deletions, change.insertions)["DF Louvain"]
    rwgp_metrics = rwgp_df.run(change.deletions, change.insertions)["GP - Dynamic Frontier Louvain"]
    print(df_metrics.modularity, rwgp_metrics.modularity)
```

## Repository layout

```text
.
├── config/                 # Benchmark + synthesis configs
├── consts/                 # Dataset-/experiment-specific constants
├── docs/                   # Architecture notes and refactor history
├── src/
│   ├── benchmarks.py       # Benchmark runner
│   ├── components/         # Result schemas + temporal change objects
│   ├── data_loader/        # Batch + window-frame dataset loaders
│   ├── gp_df/              # RWGP refinement implementations (v1..v5)
│   ├── models/             # DF + RWGP-DF + baselines
│   └── utils/              # Plotting + helpers + MLflow logging
├── run_benchmarks.py       # YAML-driven benchmark entrypoint
├── run.py                  # Synthetic Optuna/MLflow experiment
├── run_*.py                # Dataset-specific Optuna/MLflow experiments
└── requirements.txt
```

## Status

This is research / experimental code. Expect rapid iteration (especially in refinement variants) and favor `refine_version="v5"` for the most paper-aligned RW split criterion.

## Docs

- `docs/ARCHITECTURE.md`: module-level overview
- `docs/REFACTORING_SUMMARY.md`: historical refactor notes
