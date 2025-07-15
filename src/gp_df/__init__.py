from .gp_separator_v1 import separate_communities_v1
from .gp_separator_v2 import separate_communities_v2
from .gp_separator_v2_full import separate_communities_v2_full
from .gp_separator_v3 import (
    separate_communities_v3,
    separate_communities_v3_ultra_fast,
)
from .gp_separator_v4 import separate_communities_v4

separate_communities = {
    "v1": separate_communities_v1,
    "v2": separate_communities_v2,
    "v2-full": separate_communities_v2_full,
    "v3": separate_communities_v3,
    "v3_fast": separate_communities_v3_ultra_fast,
    "v4": separate_communities_v4,
}
