from typing import Any, Dict


def generate_plot_filename(mode: str, dataset_config: Dict[str, Any]) -> str:
    """
    Generate plot filename based on mode and dataset configuration.

    Args:
        mode: The benchmark mode ('batch' or 'window_frame').
        dataset_config: Dictionary containing dataset configuration parameters.

    Returns:
        Generated filename for the plot.

    Raises:
        ValueError: If mode is not supported.
        KeyError: If required configuration keys are missing.
    """
    try:
        initial_fraction = dataset_config['initial_fraction']
        load_full_nodes = dataset_config['load_full_nodes']
    except KeyError as e:
        raise KeyError(f"Missing required configuration key: {e}")

    load_type = 'load_full_nodes' if load_full_nodes else 'load_partial_nodes'
    common_suffix = f"initial_fraction_{initial_fraction}_{load_type}_benchmark_plot.png"

    if mode == "batch":
        try:
            batch_range = dataset_config['batch_range']
            return f"batch_range_{batch_range}_{common_suffix}"
        except KeyError:
            raise KeyError("Missing 'batch_range' key for batch mode")
    elif mode == "window_frame":
        try:
            step_size = dataset_config['step_size']
            window_size = dataset_config['window_size']
            return f"window_size_{window_size}_step_size_{step_size}_{common_suffix}"
        except KeyError:
            raise KeyError("Missing 'step_size' key for window_frame mode")
    else:
        raise ValueError(
            f"Unsupported mode: '{mode}'. Expected 'batch' or 'window_frame'."
        )