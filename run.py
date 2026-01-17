import os
import pickle
import random
from typing import Dict

import mlflow
import networkx as nx
import numpy as np
import optuna
from dotenv import load_dotenv
from tqdm.auto import tqdm

from consts.synthetics import BoundedValue, GlobalValues, GraphSizeConfig
from src.components.factory import IntermediateResults, MethodDynamicResults
from src.data_loader import DatasetBatchManager
from src.models import (
    DeltaScreeningLouvain,
    DynamicFrontierLouvain,
    GPDynamicFrontierLouvain,
    NaiveDynamicLouvain,
    StaticLouvain,
)
from src.models.community_info import CommunityUtils
from src.utils import helpers, mlflow_logger
from src.utils.plotter import Plotter

os.environ["PYTHONHASHSEED"] = f"{GlobalValues.SEED}"
np.random.seed(GlobalValues.SEED)
random.seed(GlobalValues.SEED)

mlflow.set_tracking_uri(GlobalValues.MLFLOW_TRACKING_URI)
load_dotenv(".env")

data_manager = DatasetBatchManager()


experiment = mlflow.get_experiment_by_name(GlobalValues.EXPERIMENT_NAME)
if experiment is None:
    mlflow.create_experiment(GlobalValues.EXPERIMENT_NAME)
experiment_id = mlflow.get_experiment_by_name(
    GlobalValues.EXPERIMENT_NAME
).experiment_id
mlflow.set_experiment(experiment_name=GlobalValues.EXPERIMENT_NAME)


def make_dataset(
    batch_range: float,
    initial_fraction: float,
    delete_insert_ratio: float,
    type: str = "lfr",
    **kwargs,
):
    dataset_config = {
        "dataset_path": "dataset/synthetic_graph.txt",
        "dataset_type": "synthetic_graph",
        "source_idx": GlobalValues.SOURCE_IDX,
        "target_idx": GlobalValues.TARGET_IDX,
        "max_steps": 15,
        "initial_fraction": initial_fraction,
        "delete_insert_ratio": delete_insert_ratio,
        "batch_range": batch_range,
    }
    if type == "lfr":
        graph = helpers.make_lfr_dataset(**kwargs)
    elif type == "gaussian":
        graph = helpers.make_gaussian_dataset(**kwargs)
    graph_info = {
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
    }
    graph = nx.convert_node_labels_to_integers(graph, first_label=0, ordering="sorted")
    nx.write_edgelist(
        G=graph, path="dataset/synthetic_graph.txt", delimiter=" ", data=False
    )
    G, temporal_changes = data_manager.get_dataset(**dataset_config)
    initial_communities = nx.algorithms.community.louvain_communities(
        G, resolution=1, seed=42
    )

    initial_communities_dict = {}
    for community_id, community in enumerate(initial_communities):  # type: ignore
        for node in community:
            initial_communities_dict[node] = community_id
    return G, temporal_changes, initial_communities_dict, graph_info


def optimize_delta_q(
    batch_range_config: BoundedValue,
    n_config: BoundedValue,
    s_config: BoundedValue,
    p_in_config: BoundedValue,
    p_out_config: BoundedValue,
    tags: Dict | None = None,
):
    plotter = Plotter()

    # Use Optuna to optimize the delta_q threshold
    def objective(trial: optuna.Trial) -> float:
        count = 0
        if GlobalValues.GENERATOR_TYPE == "gaussian":
            params = {
                "n": trial.suggest_int("n", int(n_config.lower), int(n_config.upper)),
                "s": trial.suggest_int("s", int(s_config.lower), int(s_config.upper)),
                "v": trial.suggest_float("v", 2.5, 2.5),
                "p_in": trial.suggest_float(
                    "p_in", p_in_config.lower, p_in_config.upper
                ),
                "p_out": trial.suggest_float(
                    "p_out", p_out_config.lower, p_out_config.upper
                ),
            }
        elif GlobalValues.GENERATOR_TYPE == "lfr":
            params = {
                "n": trial.suggest_int("n", 400, 500),
                "tau1": trial.suggest_float("tau1", 2.0, 3.0),
                "tau2": trial.suggest_float("tau2", 1.0, 2.0),
                "mu": trial.suggest_float("mu", 0.4, 0.7),
                "min_community": trial.suggest_int("min_community", 50, 100),
                "max_community": trial.suggest_int("max_community", 200, 300),
            }
        batch_range = trial.suggest_float(
            "batch_range", batch_range_config.lower, batch_range_config.upper
        )

        G, temporal_changes, initial_communities_dict, graph_info = make_dataset(
            batch_range=batch_range,
            initial_fraction=GlobalValues.INITIAL_FRACTION,
            delete_insert_ratio=GlobalValues.DELETE_INSERT_RATIO,
            type=GlobalValues.GENERATOR_TYPE,
            **params,
        )

        intial_results = IntermediateResults(
            modularity=CommunityUtils.calculate_modularity(G, initial_communities_dict),
            runtime=0.0,
            affected_nodes=0,
            num_communities=len(set(initial_communities_dict.values())),
        )
        gp_df = GPDynamicFrontierLouvain(
            graph=G,
            initial_communities=initial_communities_dict,
            sampler_type=GlobalValues.SAMPLER_TYPE,
            num_communities_range=GlobalValues.NUM_COMMUNITIES_RANGE,
            # refine_version="v2-full",
            refine_version="v5",
            verbose=False,
        )
        df = DynamicFrontierLouvain(
            graph=G,
            initial_communities=initial_communities_dict,
            sampler_type=GlobalValues.SAMPLER_TYPE,
            num_communities_range=GlobalValues.NUM_COMMUNITIES_RANGE,
            verbose=False,
        )
        static_louvain = StaticLouvain(
            graph=G,
            initial_communities=initial_communities_dict,
            sampler_type=GlobalValues.SAMPLER_TYPE,
            num_communities_range=GlobalValues.NUM_COMMUNITIES_RANGE,
            verbose=False,
        )
        naive_df = NaiveDynamicLouvain(
            graph=G,
            initial_communities=initial_communities_dict,
            sampler_type=GlobalValues.SAMPLER_TYPE,
            num_communities_range=GlobalValues.NUM_COMMUNITIES_RANGE,
            verbose=False,
        )
        delta_screening = DeltaScreeningLouvain(
            graph=G,
            initial_communities=initial_communities_dict,
            sampler_type=GlobalValues.SAMPLER_TYPE,
            num_communities_range=GlobalValues.NUM_COMMUNITIES_RANGE,
            verbose=False,
        )

        df_res = MethodDynamicResults()
        gp_df_res = MethodDynamicResults()
        static_louvain_res = MethodDynamicResults()
        naive_df_res = MethodDynamicResults()
        delta_screening_res = MethodDynamicResults()

        df_res.update_intermediate_results(intial_results)
        gp_df_res.update_intermediate_results(intial_results)
        static_louvain_res.update_intermediate_results(intial_results)
        naive_df_res.update_intermediate_results(intial_results)
        delta_screening_res.update_intermediate_results(intial_results)

        best_q = 0
        temporal_progress = tqdm(
            temporal_changes, total=len(temporal_changes), leave=False
        )
        for idx, change in enumerate(temporal_progress):
            gp_df_intermediate_res = gp_df.run(change.deletions, change.insertions)
            df_intermediate_res = df.run(change.deletions, change.insertions)
            static_intermediate_res = static_louvain.run(
                change.deletions, change.insertions
            )
            naive_df_intermediate_res = naive_df.run(
                change.deletions, change.insertions
            )
            delta_screening_intermediate_res = delta_screening.run(
                change.deletions, change.insertions
            )

            step_gp_df_res = gp_df_intermediate_res["GP - Dynamic Frontier Louvain"]
            step_df_res = df_intermediate_res["DF Louvain"]
            step_static_louvain_res = static_intermediate_res["Static Louvain"]
            step_naive_df_res = naive_df_intermediate_res["Naive Dynamic Louvain"]
            step_delta_screening_res = delta_screening_intermediate_res[
                "Delta Screening"
            ]

            df_res.update_intermediate_results(intermediate_results=step_df_res)
            gp_df_res.update_intermediate_results(intermediate_results=step_gp_df_res)
            static_louvain_res.update_intermediate_results(
                intermediate_results=step_static_louvain_res
            )
            naive_df_res.update_intermediate_results(
                intermediate_results=step_naive_df_res
            )
            delta_screening_res.update_intermediate_results(
                intermediate_results=step_delta_screening_res
            )

            delta_q = step_gp_df_res.modularity - step_df_res.modularity
            if delta_q > best_q:
                count += 1
                best_q = delta_q
            if delta_q < 0:
                raise optuna.TrialPruned(
                    f"Delta Q is negative at step {idx}, stopping trial."
                )

            temporal_progress.set_postfix(
                {
                    "Delta Q": step_gp_df_res.modularity - step_df_res.modularity,
                }
            )
        target = count + best_q

        def should_log_to_mlflow(
            target_val: float,
            gp_df_results: MethodDynamicResults,
            static_results: MethodDynamicResults,
        ):
            return (
                target_val > 0
                and gp_df_results.avg_runtime < static_results.avg_runtime
            )

        if should_log_to_mlflow(target, gp_df_res, static_louvain_res):
            with mlflow.start_run():
                fig = plotter.plot_results(
                    dataset_name="synthetic_graph",
                    results={
                        "DF Louvain": df_res,
                        "RWGP - DF Louvain": gp_df_res,
                        "Static Louvain": static_louvain_res,
                        "ND  Louvain": naive_df_res,
                        "DS Louvain": delta_screening_res,
                    },
                )
                mlflow.log_params(params)
                mlflow.log_figure(fig, "results.png")
                mlflow_logger.log_dynamic_results("DF Louvain", df_res)
                mlflow_logger.log_dynamic_results("GP-DF Louvain", gp_df_res)
                mlflow_logger.log_dynamic_results("Static Louvain", static_louvain_res)
                mlflow_logger.log_dynamic_results("ND Louvain", naive_df_res)
                mlflow_logger.log_dynamic_results("DS Louvain", delta_screening_res)

                mlflow.log_dict(
                    dictionary={
                        "DF Louvain": df_res.model_dump(),
                        "GP-DF Louvain": gp_df_res.model_dump(),
                        "Static Louvain": static_louvain_res.model_dump(),
                        "ND Louvain": naive_df_res.model_dump(),
                        "DS Louvain": delta_screening_res.model_dump(),
                    },
                    artifact_file="dynamic_results.json",
                )
                mlflow.log_dict(dictionary=graph_info, artifact_file="graph_info.json")
                mlflow.log_artifact(
                    "dataset/synthetic_graph.txt", "synthetic_graph.txt"
                )
                with open("graph-snapshot.pkl", "wb") as f:
                    payload = {
                        "graph": G,
                        "temporal_changes": temporal_changes,
                        "initial_communities": initial_communities_dict,
                    }
                    pickle.dump(payload, f)
                mlflow.log_artifact("graph-snapshot.pkl")
                mlflow.log_metric("target", target)
                mlflow.log_param("batch_range_type", batch_range_config.name)
                if tags is not None:
                    mlflow.set_tags(tags)
        return best_q

    study = optuna.create_study(
        direction="maximize",
        storage=GlobalValues.OPTUNA_DB,
        sampler=optuna.samplers.NSGAIISampler(
            population_size=2 * GlobalValues.NUM_TRIALS
        ),
        pruner=optuna.pruners.SuccessiveHalvingPruner(),
    )
    study.optimize(
        objective,
        n_trials=GlobalValues.NUM_TRIALS,
        show_progress_bar=True,
    )
    return study


def runner():
    for graph_size in GraphSizeConfig().get_list():
        print(f"Running for graph size: {graph_size.graph_size}")
        for batch_range in graph_size.batch_range:
            print(
                f"Optimizing for graph size {graph_size.graph_size}, community size {graph_size.community_size} batch range: {batch_range.name}"
            )
            optimize_delta_q(
                batch_range_config=batch_range,
                n_config=graph_size.n,
                s_config=graph_size.s,
                p_in_config=graph_size.p_in,
                p_out_config=graph_size.p_out,
                tags={
                    "graph_size": graph_size.graph_size,
                    "community_size": graph_size.community_size,
                    "batch_range": batch_range.name,
                },
            )


if __name__ == "__main__":
    runner()
