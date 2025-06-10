"""
Dynamic Frontier Louvain algorithm for community detection in dynamic networks.

This module has been refactored for better maintainability. The original implementation
has been split into smaller, focused modules for improved code organization.

This file serves as the main entry point and imports from the refactored modules.
"""

# Import all components from the refactored modules
from .community_info import CommunityInfo, CommunityUtils
from .df_louvain_async import AsyncDynamicFrontierLouvain
from .df_louvain_sync import DynamicFrontierLouvain
from .df_louvain_sync_separate import GPDynamicFrontierLouvain

# Re-export for backward compatibility
__all__ = [
    "CommunityInfo",
    "GPDynamicFrontierLouvain",
    "CommunityUtils",
    "DynamicFrontierLouvain",
    "AsyncDynamicFrontierLouvain",
]
