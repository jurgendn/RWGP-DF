"""
Dynamic Frontier Louvain package for community detection in dynamic networks.
"""

from .df_louvain import DynamicFrontierLouvain
from .benchmarks import DFLouvainBenchmark

__all__ = ["DynamicFrontierLouvain", "DFLouvainBenchmark"]
