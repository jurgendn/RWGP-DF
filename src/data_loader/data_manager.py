"""
Dataset management module for DFLouvain benchmarks.

This module provides specialized dataset managers for different data loading approaches.
"""

import os
import pickle
import time
from typing import Dict, List, Optional, Text, Tuple

import networkx as nx

from .batch_loader import (
    create_synthetic_dynamic_graph,
    load_bitcoin_dataset,
    load_college_msg_dataset,
    load_sx_mathoverflow_dataset,
)
from .window_loader import (
    load_bitcoin_sliding_window,
    load_college_msg_sliding_window,
    load_sx_mathoverflow_sliding_window,
)


class BaseDatasetManager:
    """Base class for dataset managers with common functionality."""
    
    def __init__(self, cache_dir: str = None, use_cache: bool = True, verbose: bool = True):
        """
        Initialize the dataset manager.
        
        Args:
            cache_dir: Directory to store cached datasets
            use_cache: Whether to use cached datasets when available
            verbose: Whether to print detailed information
        """
        self.cache_dir = cache_dir
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        self.use_cache = use_cache
        self.verbose = verbose
    
    def _get_cache_filename(self, cache_key: str) -> str:
        """Generate a unique cache filename."""
        if not self.cache_dir:
            return None
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")
    
    def _get_from_cache(self, cache_key: str) -> Optional[Tuple[nx.Graph, List[Dict]]]:
        """Try to load dataset from cache."""
        if not self.use_cache or not self.cache_dir:
            return None
            
        cache_file = self._get_cache_filename(cache_key)
        if not cache_file or not os.path.exists(cache_file):
            return None
            
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            if self.verbose:
                print(f"Cache loading failed: {e}")
            return None
    
    def _save_to_cache(self, cache_key: str, data: Tuple[nx.Graph, List[Dict]]) -> None:
        """Save dataset to cache."""
        if not self.use_cache or not self.cache_dir:
            return
            
        cache_file = self._get_cache_filename(cache_key)
        if not cache_file:
            return
            
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            if self.verbose:
                print(f"Dataset cached to {cache_file}")
        except Exception as e:
            if self.verbose:
                print(f"Failed to cache dataset: {e}")
    
    def clear_cache(self, prefix: Optional[str] = None) -> None:
        """
        Clear cached datasets.
        
        Args:
            prefix: Optional prefix to clear (or all if None)
        """
        if not self.cache_dir or not os.path.exists(self.cache_dir):
            return
            
        count = 0
        for filename in os.listdir(self.cache_dir):
            if prefix is None or filename.startswith(prefix):
                try:
                    os.remove(os.path.join(self.cache_dir, filename))
                    count += 1
                except Exception as e:
                    if self.verbose:
                        print(f"Failed to remove {filename}: {e}")
        
        if self.verbose:
            print(f"Cleared {count} cached datasets")
    
    def list_available_datasets(self, data_dir: str) -> Dict[str, Dict[str, str]]:
        """
        List available datasets in the specified directory.
        
        Args:
            data_dir: Directory to scan for datasets
            
        Returns:
            Dictionary of dataset types and their paths
        """

        datasets = {
            "college_msg": {},
            "bitcoin": {},
            "sx-mathoverflow": {},
            "synthetic": {"synthetic": "Generated on demand"}
        }
        
        if not os.path.exists(data_dir):
            if self.verbose:
                print(f"Data directory not found: {data_dir}")
            return datasets
        
        # Scan for dataset files
        for filename in os.listdir(data_dir):
            filepath = os.path.join(data_dir, filename)
            if os.path.isfile(filepath):
                lower_name = filename.lower()
                if "college" in lower_name or "msg" in lower_name:
                    datasets["college_msg"][filename] = filepath
                elif "bitcoin" in lower_name:
                    datasets["bitcoin"][filename] = filepath
                elif "mathoverflow" in lower_name or "sx-" in lower_name:
                    datasets["sx-mathoverflow"][filename] = filepath
        
        return datasets


class DatasetBatchManager(BaseDatasetManager):
    """
    Manager for batch-based dataset loading and caching.
    
    This class handles loading datasets with traditional batch-range 
    based loading, where data is split into initial fraction and batches.
    """
    def __init__(self, cache_dir = None, use_cache = True, verbose = True):
        super().__init__(cache_dir, use_cache, verbose)
        self.dataset_map = {
            "synthetic": create_synthetic_dynamic_graph,
            "college_msg": load_college_msg_dataset,
            "bitcoin_alpha": load_bitcoin_dataset,
            "bitcoin_otc": load_bitcoin_dataset,
            "sx-mathoverflow": load_sx_mathoverflow_dataset,
        }
    def get_dataset(
        self,
        dataset_path: str,
        dataset_type: str,
        batch_range: float = 1e-3,
        initial_fraction: float = 0.3,
        max_steps: int = 100,
        force_reload: bool = False,
    ) -> Tuple[nx.Graph, List[Dict[Text, List]]]:
        """
        Get a dataset with batch-based loading parameters.
        
        Args:
            dataset_path: Path to the dataset file
            dataset_type: Type of dataset ('college_msg', 'bitcoin_otc', 'bitcoin_alpha', 'sx-mathoverflow', or 'synthetic')
            batch_range: Batch range for data loading
            initial_fraction: Initial fraction of data for the base graph
            force_reload: Whether to force reload even if cached version exists
            
        Returns:
            Tuple of (initial_graph, temporal_changes)
        """
        # Generate cache key
        cache_key = f"batch_{os.path.basename(dataset_path)}_{dataset_type}_b{batch_range}_i{initial_fraction}"
        if dataset_type == "synthetic":
            cache_key = "synthetic_batch_data"

        # Check cache first if enabled
        if not force_reload:
            cached_dataset = self._get_from_cache(cache_key)
            if cached_dataset:
                if self.verbose:
                    print(f"Loaded batch dataset from cache: {dataset_type}")
                return cached_dataset
        
        # Load dataset based on type and parameters
        start_time = time.time()
        if dataset_type not in self.dataset_map:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        # Load the dataset using the appropriate function
        load_function = self.dataset_map[dataset_type]
        # Call the loading function with the appropriate parameters
        G, temporal_changes = load_function(
            file_path=dataset_path,
            batch_range=batch_range,
            initial_fraction=initial_fraction,
            max_steps=max_steps,
        )
        
        load_time = time.time() - start_time
        
        # Report on loaded dataset
        if self.verbose:
            print(f"Loaded {dataset_type} batch dataset in {load_time:.2f}s")
            print(f"  - Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
            print(f"  - Temporal changes: {len(temporal_changes)} steps")
        
        # Cache dataset
        self._save_to_cache(cache_key, (G, temporal_changes))
        
        return G, temporal_changes


class DatasetWindowTimeManager(BaseDatasetManager):
    """
    Manager for time window-based dataset loading and caching.
    
    This class handles loading datasets with sliding window approach,
    where data is processed in time windows that slide over the timeline.
    """
    def __init__(self, cache_dir = None, use_cache = True, verbose = True):
        super().__init__(cache_dir, use_cache, verbose)
        self.dataset_map = {
            "synthetic": create_synthetic_dynamic_graph,
            "college_msg": load_college_msg_sliding_window,
            "bitcoin_alpha": load_bitcoin_sliding_window,
            "bitcoin_otc": load_bitcoin_sliding_window,
            "sx-mathoverflow": load_sx_mathoverflow_sliding_window,
        }
    def get_dataset(
        self,
        dataset_path: str,
        dataset_type: str,
        window_size: int = 5,
        step_size: int = 1,
        initial_fraction: float = 0.3,
        max_steps: int | None = None,
        force_reload: bool = False,
        load_full_nodes: bool = True,
    ) -> Tuple[nx.Graph, List[Dict]]:
        """
        Get a dataset with sliding window-based loading parameters.
        
        Args:
            dataset_path: Path to the dataset file
            dataset_type: Type of dataset ('college_msg', 'bitcoin_otc', 'bitcoin_alpha', 'sx-mathoverflow', or 'synthetic')
            window_size: Number of time units in each window
            step_size: Number of time units to slide the window
            force_reload: Whether to force reload even if cached version exists
            
        Returns:
            Tuple of (initial_graph, temporal_changes)
        """
        # Generate cache key
        cache_key = f"window_{os.path.basename(dataset_path)}_{dataset_type}_w{window_size}_s{step_size}"
        if dataset_type == "synthetic":
            cache_key = "synthetic_window_data"
        
        # Check cache first if enabled
        if not force_reload:
            cached_dataset = self._get_from_cache(cache_key)
            if cached_dataset:
                if self.verbose:
                    print(f"Loaded window dataset from cache: {dataset_type}")
                return cached_dataset
        
        # Load dataset based on type and parameters
        start_time = time.time()
        
        if dataset_type not in self.dataset_map:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        # Load the dataset using the appropriate function
        load_function = self.dataset_map[dataset_type]
        # Call the loading function with the appropriate parameters
        G, temporal_changes = load_function(
            file_path=dataset_path,
            window_size=window_size,
            step_size=step_size,
            initial_fraction=initial_fraction,
            max_steps=max_steps,
            load_full_nodes=load_full_nodes,
        )
        
        load_time = time.time() - start_time
        
        # Report on loaded dataset
        if self.verbose:
            print(f"Loaded {dataset_type} window dataset in {load_time:.2f}s")
            print(f"  - Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
            print(f"  - Temporal changes: {len(temporal_changes)} steps")
        
        # Cache dataset
        self._save_to_cache(cache_key, (G, temporal_changes))
        
        return G, temporal_changes
