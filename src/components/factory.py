from typing import List

import numpy as np
from pydantic import BaseModel


class IntermediateResults(BaseModel):
    runtime: float = 0.0
    modularity: float = 0.0
    affected_nodes: int = 0


class MethodDynamicResults(BaseModel):
    runtimes: List[float] = []
    modularities: List[float] = []
    affected_nodes: List[int] = []
    iterations_per_step: List[int] = []

    @property
    def avg_runtime(self):
        if not self.runtimes:
            return 0.0
        return np.mean(self.runtimes)
    
    @property
    def total_runtime(self) -> float:
        return sum(self.runtimes)
    
    @property
    def modularity_stability(self) -> float:
        if len(self.modularities) < 2:
            return 0.0
        return max(self.modularities) - min(self.modularities)
    
    @property
    def modularity_range(self):
        if not self.modularities:
            return (None, None)
        return (min(self.modularities), max(self.modularities))

    @property
    def time_steps(self) -> List[int]:
        return list(range(len(self.runtimes)))