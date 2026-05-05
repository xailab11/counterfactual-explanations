"""
experiments/run_experiments.py

Main entry point for running all experiments reported in the paper.

This script executes:
- consistency analysis,
- additivity evaluation,
- minimality evaluation,

for a given set of datasets and language models.
"""
from pathlib import Path
from datetime import datetime
import argparse
import logging

from .experiments_utils import (
    set_seed,
    show_and_save,
    make_experiment_dir,
    make_run_folder,
    setup_logger,
    summarize_rank_stability,
    summarize_rank_agreement
)
from .query_generation import load_dataset
from counterfactual.cache_utils import cache_manager
from evaluation.consistency import (
    compute_pi_descriptive_measures,
    analyze_convergence,
    plot_node_rank_stability_aggregated,
    plot_node_rank_stability,
    plot_kendalls_w_aggregated,
    plot_kendalls_w_boxplot
)
from evaluation.additivity import (
    evaluate_additivity,
    plot_mae_by_graph_size,
    plot_mae_by_subset_size,
    plot_mae_distributions
)
from evaluation.minimality import (
    evaluate_minimality,
    plot_minimality_scatter,
    plot_ratio_distribution,
    merge_minimality_dfs
)

# Config
datasets = ["data_mcu", "data_wiki", "data_football", "data_parks", "data_hetionet"]
models = ["gemma4:latest", "phi3:medium", "gpt-oss:120b", "llama3:70b", "qwen2.5:72b"]

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
DATA_DIR = BASE_DIR / "data"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# Consistency experiments (stability of Pi and rankings)
# ---------------------------------------------------------
def run_consistency(graphs, questions, data, model, run_folder):
    with cache_manager(data, model):
        path_consistency_out = run_folder / "consistency"
        exp_consistency_dir = make_experiment_dir(path_consistency_out, data, model)

        # Folders
        pi_distribution_dir = exp_consistency_dir / "pi_distribution"
        pi_distribution_dir.mkdir(exist_ok=True)
        (pi_distribution_dir / "measures").mkdir(exist_ok=True)
        (pi_distribution_dir / "distribution_details").mkdir(exist_ok=True)
        pi_convergence_dir = exp_consistency_dir / "pi_convergence"
        pi_convergence_dir.mkdir(exist_ok=True)

        # Compute measures
        df_pi_measures = compute_pi_descriptive_measures(graphs, questions, model, clear_cache=False)

        for _, row in df_pi_measures.iterrows():
            row["pi"].to_csv((pi_distribution_dir / "distribution_details" / f"{row['graph_id']}_pi.csv"))

        show_and_save(df_pi_measures.drop(columns=["pi"]), exp_consistency_dir, "df_pi_measures", csv=True, pkl=True)

    with cache_manager(data, model):
        analyze_convergence(graphs, questions, model, pi_convergence_dir, reload_existing=True, n_runs=10, display_obj=False)

        df_ranks_all = summarize_rank_stability(graphs, pi_convergence_dir= pi_convergence_dir)
        show_and_save(plot_node_rank_stability(df_ranks_all), pi_convergence_dir, "node_rank_stability")
        show_and_save(plot_node_rank_stability_aggregated(df_ranks_all), pi_convergence_dir, "node_rank_stability_aggregated")

        df_global_rank_agreement = summarize_rank_agreement(graphs, pi_convergence_dir= pi_convergence_dir)
        show_and_save(plot_kendalls_w_aggregated(df_global_rank_agreement), pi_convergence_dir, "kendalls_w_aggregated")
        show_and_save(plot_kendalls_w_boxplot(df_global_rank_agreement), pi_convergence_dir, "kendalls_w_boxplot")

# -------------------------------------------------------------
# Additivity experiments (validating the additivity assumption)
# -------------------------------------------------------------
def run_additivity(graphs, questions, data, model, run_folder):
    with cache_manager(data, model):
        exp_additivity_dir = make_experiment_dir(run_folder / "additivity", data, model)
        (exp_additivity_dir / "details").mkdir(exist_ok=True)
        (exp_additivity_dir / "subsets").mkdir(exist_ok=True)

        df_add = evaluate_additivity(graphs, questions, model, max_subset_size=None)
        show_and_save(df_add, exp_additivity_dir, "df_add", csv=True, pkl=True)
        for _, row in df_add.iterrows():
            graph_id = row["graph_id"]
            df_details = row["details"]
            show_and_save(df_details, exp_additivity_dir / "details", f"{graph_id}", csv=True)

        # Save summary
        show_and_save(df_add.drop(columns=["question", "details"]).round(4), exp_additivity_dir, "additivity_summary",
                      csv=True)
        show_and_save(df_add[["graph_id", "n_nodes", "MAE", "MAE_gt1"]].round(4), exp_additivity_dir, "table_mae",
                      csv=True, tex=True)

        show_and_save(plot_mae_distributions(df_add, mae_column="MAE_gt1"), exp_additivity_dir, "additivity_distribution")
        show_and_save(plot_mae_by_graph_size(df_add, mae_column="MAE_gt1"), exp_additivity_dir, "mae_vs_graphsize")
        show_and_save(plot_mae_by_subset_size(df_add, use_subset_gt1=True), exp_additivity_dir, "mae_vs_subsetsize")

# -----------------------------------------------------------
# Minimality experiments (greedy vs. optimal counterfactuals)
# -----------------------------------------------------------
def run_minimality(graphs, questions, data, model, run_folder):
    with cache_manager(data, model):
        exp_minimality_dir = make_experiment_dir(run_folder / "minimality", data, model)

        for tau in [0.1]:
            df_min = evaluate_minimality(graphs, questions, tau=tau, model=model)
            show_and_save(df_min, exp_minimality_dir, f"df_min_tau{int(tau*100)}", csv=True, pkl=True)
            show_and_save(plot_minimality_scatter(df_min, color_by_size=True), exp_minimality_dir, f"minimality_scatter_tau{int(tau*100)}")
            show_and_save(plot_ratio_distribution(df_min, stacked=True), exp_minimality_dir, f"ratio_distribution_tau{int(tau*100)}")

        merge_minimality_dfs(exp_minimality_dir, "df_min")


def run_experiment(data, model, results_dir_override=None):
    data_path = DATA_DIR / f"{data}.json"
    graphs, questions = load_dataset(data_path)
    if results_dir_override is not None:
        run_folder = Path(results_dir_override).resolve()
        run_folder.mkdir(parents=True, exist_ok=True)
    else:
        run_folder = make_run_folder(data, model, RESULTS_DIR)
    logger = setup_logger(name="experiment", log_dir=run_folder / "logs", level=logging.INFO)

    print(f"Start Experiment: Dataset={data}, Model={model}, Folder={run_folder}")
    logger.info("Start Experiment | dataset=%s | model=%s", data, model)

    run_consistency(graphs, questions, data, model, run_folder)
    run_additivity(graphs, questions, data, model, run_folder)
    run_minimality(graphs, questions, data, model, run_folder)

    logger.info("Experiment finished | dataset=%s | model=%s", data, model)

def main():
    set_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        action="append",
        help="Dataset name (can be used multiple times, e.g., --dataset data1 --dataset data2)"
    )
    parser.add_argument(
        "--model",
        type=str,
        action="append",
        help="Model name (can be used multiple times)"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Optional: existing results directory to use"
    )
    args = parser.parse_args()

    selected_datasets = args.dataset if args.dataset else datasets
    selected_models = args.model if args.model else models

    for data in selected_datasets:
        for model in selected_models:
            run_experiment(data, model, args.results_dir)

if __name__ == "__main__":
    main()