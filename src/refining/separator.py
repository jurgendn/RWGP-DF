from src.refining.gp_separator_v1 import separate_communities_v1
from src.refining.gp_separator_v2 import separate_communities_v2
from src.refining.gp_separator_v3 import separate_communities_v3
from src.refining.gp_separator_v4 import separate_communities_v4

separate_communities = {
    "v1": separate_communities_v1,
    "v2": separate_communities_v2,
    "v3": separate_communities_v3,
    "v4": separate_communities_v4,
}
