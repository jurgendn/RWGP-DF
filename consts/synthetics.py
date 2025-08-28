from dataclasses import dataclass
from typing import List


@dataclass
class GlobalValues:
    MLFLOW_TRACKING_URI = "http://100.125.199.30:5000"
    EXPERIMENT_NAME = "[RIVF2025] Synthetic Dataset"
    OPTUNA_DB = "postgresql://root:toor@100.125.199.30:5432/optuna"


    SEED = 42
    GENERATOR_TYPE = "gaussian" # Options: 'lfr', 'gaussian'
    SOURCE_IDX = 0
    TARGET_IDX = 1
    INITIAL_FRACTION = 0.5
    DELETE_INSERT_RATIO = 0.7
    SAMPLER_TYPE = "selective"  # Options: 'full', 'selective'
    # SAMPLER_TYPE = "full"  # Options: 'full', 'selective'
    NUM_COMMUNITIES_RANGE = (1, 4)  # Range of communities to consider for the algorithm
    NUM_TRIALS = 300


@dataclass
class BoundedValue:
    name: str
    lower: float | int
    upper: float | int

@dataclass
class GraphParams:
    graph_size: str
    community_size: str
    n: BoundedValue
    s: BoundedValue
    p_in: BoundedValue
    p_out: BoundedValue
    batch_range: List[BoundedValue]

@dataclass
class GraphSizeConfig:
    small_small = GraphParams(
        graph_size="x_small",
        community_size="small",
        n=BoundedValue("n", 2500, 2500),
        s=BoundedValue("s", 200, 300),
        p_in=BoundedValue("p_in", 0.01, 0.03),
        p_out=BoundedValue("p_out", 0.005, 0.0075),
        batch_range=[
            BoundedValue("small", 0.0005, 0.005),
            BoundedValue("medium", 0.005, 0.05),
            BoundedValue("large", 0.05, 0.1),
        ],
    )
    small_medium = GraphParams(
        graph_size="x_small",
        community_size="medium",
        n=BoundedValue("n", 2500, 2500),
        s=BoundedValue("s", 480, 510),
        p_in=BoundedValue("p_in", 0.0125, 0.0175),
        p_out=BoundedValue("p_out", 0.0075, 0.01),
        batch_range=[
            BoundedValue("small", 0.0005, 0.005),
            BoundedValue("medium", 0.005, 0.05),
            BoundedValue("large", 0.05, 0.06),
        ],
    )
    small_large = GraphParams(
        graph_size="x_small",
        community_size="large",
        n=BoundedValue("n", 2500, 2500),
        s=BoundedValue("s", 850, 1000),
        p_in=BoundedValue("p_in", 0.005, 0.01),
        p_out=BoundedValue("p_out", 0.00075, 0.001),
        batch_range=[
            BoundedValue("small", 0.0001, 0.005),
            BoundedValue("medium", 0.005, 0.05),
            BoundedValue("large", 0.05, 0.1),
        ],
    )
    large_small = GraphParams(
        graph_size="x_large",
        community_size="small",
        n=BoundedValue("n", 7500, 7500),
        s=BoundedValue("s", 100, 150),
        p_in=BoundedValue("p_in", 0.022, 0.025),
        p_out=BoundedValue("p_out", 0.0011, 0.0013),
        batch_range=[
            BoundedValue("small", 0.0001, 0.001),
            BoundedValue("medium", 0.001, 0.01),
            BoundedValue("large", 0.01, 0.1),
        ],
    )
    large_medium = GraphParams(
        graph_size="x_large",
        community_size="medium",
        n=BoundedValue("n", 7500, 7500),
        s=BoundedValue("s", 300, 500),
        p_in=BoundedValue("p_in", 0.012, 0.014),
        p_out=BoundedValue("p_out", 0.00075, 0.001),
        batch_range=[
            BoundedValue("small", 0.0001, 0.001),
            BoundedValue("medium", 0.001, 0.01),
            BoundedValue("large", 0.01, 0.1),
        ],
    )
    large_large = GraphParams(
        graph_size="x_large",
        community_size="large",
        n=BoundedValue("n", 7500, 7500),
        s=BoundedValue("s", 800, 1000),
        p_in=BoundedValue("p_in", 0.02, 0.024),
        p_out=BoundedValue("p_out", 0.0011, 0.0013),
        batch_range=[
            BoundedValue("small", 0.0001, 0.001),
            BoundedValue("medium", 0.001, 0.01),
            BoundedValue("large", 0.01, 0.1),
        ],
    )

    def get_list(self):
        return [
            # self.small_small,
            # self.small_medium,
            # self.small_large,
            self.large_small,
            # self.large_medium,
            self.large_large,
        ]