import mlflow
from src.components.factory import MethodDynamicResults


def log_dynamic_results(method_name: str, method_results: MethodDynamicResults):
    """
    Log dynamic results to MLflow.

    Args:
        method_name (str): Name of the method.
        method_results (MethodDynamicResults): Results of the method.
    """

    num_samples = len(method_results.modularities)
    for step in range(1, num_samples + 1):
        mlflow.log_metric(
            f"modularity/{method_name}",
            method_results.modularities[step - 1],
            step=step,
        )
        mlflow.log_metric(
            f"runtime/{method_name}",
            method_results.runtimes[step - 1],
            step=step,
        )
        mlflow.log_metric(
            f"affected_nodes/{method_name}",
            method_results.affected_nodes[step - 1],
            step=step,
        )
        mlflow.log_metric(
            f"num_communities/{method_name}",
            method_results.num_communities[step - 1],
            step=step,
        )
