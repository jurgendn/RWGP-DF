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

from consts.sx_mathoverflow import BatchSizeConfig, BoundedValue, GlobalValues
from src.components.factory import IntermediateResults, MethodDynamicResults
from src.data_loader import DatasetBatchManager
from src.data_loader.batch_loader import load_txt_dataset
from src.models import (
    DeltaScreeningLouvain,
    DynamicFrontierLouvain,
    GPDynamicFrontierLouvain,
    NaiveDynamicLouvain,
    StaticLouvain,
)
from src.models.community_info import CommunityUtils
from src.utils import mlflow_logger
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
    batch_range: float, initial_fraction: float, delete_insert_ratio: float, **kwargs
):
    G, temporal_changes = load_txt_dataset(
        file_path="dataset/sx-mathoverflow.txt",
        source_idx=1,
        target_idx=2,
        batch_range=batch_range,
        initial_fraction=initial_fraction,
        delete_insert_ratio=delete_insert_ratio,
        load_full_nodes=False,
        max_steps=15,
    )
    initial_communities = nx.algorithms.community.louvain_communities(
        G, resolution=1, seed=42
    )
    graph_info = {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
    }
    initial_communities_dict = {}
    for community_id, community in enumerate(initial_communities):  # type: ignore
        for node in community:
            initial_communities_dict[node] = community_id
    return G, temporal_changes, initial_communities_dict, graph_info


def optimize_delta_q(
    batch_range_config: BoundedValue,
    initial_fraction_config: BoundedValue,
    delete_insertion_ratio_config: BoundedValue,
    tags: Dict | None = None,
):
    plotter = Plotter()

    # Use Optuna to optimize the delta_q threshold
    def objective(trial: optuna.Trial) -> float:
        count = 0
        batch_range = trial.suggest_float(
            "batch_range", batch_range_config.lower, batch_range_config.upper
        )
        initial_fraction = trial.suggest_float(
            "initial_fraction",
            low=initial_fraction_config.lower,
            high=initial_fraction_config.upper,
        )
        delete_insert_ratio = trial.suggest_float(
            "delete_insert_ratio",
            low=delete_insertion_ratio_config.lower,
            high=delete_insertion_ratio_config.upper,
        )
        params = {
            "batch_range": batch_range,
            "initial_fraction": initial_fraction,
            "delete_insert_ratio": delete_insert_ratio,
        }
        G, temporal_changes, initial_communities_dict, graph_info = make_dataset(
            batch_range=batch_range,
            initial_fraction=initial_fraction,
            delete_insert_ratio=delete_insert_ratio,
            delete_insertion_ratio=delete_insert_ratio,
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
                mlflow.log_artifact("dataset/sx-mathoverflow.txt", "dataset.txt")
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
    for params in BatchSizeConfig().get_list():
        print("Running for SX Mathoverflow graph")
        for batch_range in params.batch_range:
            if batch_range.name not in ["small"]:
                continue
            optimize_delta_q(
                batch_range_config=batch_range,
                initial_fraction_config=params.initial_fraction,
                delete_insertion_ratio_config=params.delete_insertion_ratio,
                tags={
                    "dataset": "sx-mathoverflow",
                    "batch_range": batch_range.name,
                },
            )


if __name__ == "__main__":
    runner()
