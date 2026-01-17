from dataclasses import dataclass


@dataclass(frozen=True)
class Colormaps:
    dynamic_frontier_louvain: str = "blue"
    naive_dynamic_louvain: str = "orange"
    delta_screening_louvain: str = "purple"
    gp_dynamic_frontier_louvain: str = "red"
    nx: str = "green"


@dataclass(frozen=True)
class Markers:
    dynamic_frontier_louvain: str = "x"
    naive_dynamic_louvain: str = "o"
    delta_screening_louvain: str = "s"
    gp_dynamic_frontier_louvain: str = "^"
    nx: str = "D"


@dataclass(frozen=True)
class Linestyles:
    dynamic_frontier_louvain: str = "--"
    naive_dynamic_louvain: str = "-"
    delta_screening_louvain: str = "--"
    gp_dynamic_frontier_louvain: str = "-."
    nx: str = ":"
