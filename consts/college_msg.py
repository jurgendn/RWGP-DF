from dataclasses import dataclass
from typing import List


@dataclass
class GlobalValues:
    MLFLOW_TRACKING_URI = "http://100.125.199.30:5000"
    EXPERIMENT_NAME = "[RIVF2025][college-msg] Realworld Dataset"
    OPTUNA_DB = "postgresql://root:toor@100.125.199.30:5432/optuna"
    SEED = 42
    SOURCE_IDX = 0
    TARGET_IDX = 1
    INITIAL_FRACTION = 0.4
    DELETE_INSERT_RATIO = 0.8
    SAMPLER_TYPE = "selective"  # Options: 'full', 'selective'
    # SAMPLER_TYPE = "full"  # Options: 'full', 'selective'
    NUM_COMMUNITIES_RANGE = (1, 3)  # Range of communities to consider for the algorithm
    NUM_TRIALS = 300


@dataclass
class BoundedValue:
    name: str
    lower: float | int
    upper: float | int


@dataclass
class GraphParams:
    initial_fraction: BoundedValue
    delete_insertion_ratio: BoundedValue
    batch_range: List[BoundedValue]


@dataclass
class BatchSizeConfig:
    small_small = GraphParams(
        initial_fraction=BoundedValue("initial_fraction", 0.5, 0.6),
        delete_insertion_ratio=BoundedValue("delete_insertion_ratio", 0.6, 0.8),
        batch_range=[
            BoundedValue("small", 0.00001, 0.0001),
            BoundedValue("medium", 0.0001, 0.001),
            BoundedValue("large", 0.001, 0.005),
        ],
    )

    def get_list(self):
        return [
            self.small_small,
        ]
