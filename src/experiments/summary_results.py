"""
experiments/summary_results.py

Aggregate and summarize experimental results across datasets
and language models.

Produces:
- Per-model outputs
- Global summary tables
- Combined compact table
- Permutation test results
"""
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import seaborn as sns
import numpy as np
import pickle

from experiments_utils import (
    show_and_save,
    latest_run_for_dataset,
    plot_rank_stability,
    plot_kendall_w,
    plot_mae_boxplot,
    plot_cardinality_ratio,
    prepare_graph_arrays,
    dominance_fraction_from_graphs,
    format_p_value
)

sns.set_theme(style="whitegrid")

# -----------------------------------------
# CLI & CONFIG
# -----------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Summarize experiment results.")

    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--tex-dir", type=Path)

    parser.add_argument("--datasets", nargs="+", default=[
        "data_mcu", "data_wiki", "data_football",
        "data_parks", "data_hetionet",
    ])

    parser.add_argument("--models", nargs="+", default=[
        "llama3_70b",
        "qwen2.5_72b",
        "gpt-oss_120b",
        "gemma4_latest",
        "phi3:medium"
    ])

    parser.add_argument("--log-level", default="INFO")

    return parser.parse_args()

def setup_logging(level="INFO"):
    logging.basicConfig(
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        level=getattr(logging, level),
    )

# -----------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------
def ensure_dirs(*paths):
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def to_categorical(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype("category")
    return df

def mean_pm_std(series: pd.Series, decimals: int = 2) -> str:
    if series.empty:
        return "--"
    m = series.mean()
    s = series.std()
    return f"{m:.2f} ± {s:.2f}"

def model_fs_name(model: str) -> str:
    return model.replace(":", "_").replace("-", "_").replace(".", "_")


def strip_data_prefix(df):
    df = df.copy()
    df["dataset"] = df["dataset"].str.replace("^data_", "", regex=True)
    return df

def normalize_model_dataset_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "model" in df.columns:
        df["model"] = (
            df["model"]
            .astype(str)
            .str.replace("_", ":", regex=False)
        )

    if "dataset" in df.columns:
        df["dataset"] = (
            df["dataset"]
            .astype(str)
            .str.replace("^data_", "", regex=True)
        )

    return df

# -----------------------------------------
# DATA LOADING
# -----------------------------------------
def load_model_data(model, datasets, results_dir):
    collectors = {"min": [], "add": [], "rank": [], "kendall": []}

    for ds in datasets:
        run = latest_run_for_dataset(model, ds, results_dir)

        logging.info(f"[RUN MATCH] {model} / {ds}: {run}")

        if run is None:
            continue

        base = run / "consistency" / ds / model_fs_name(model)

        files = {
            "min": run / "minimality" / ds / model_fs_name(model) / "df_min_tau10.pkl",
            "add": run / "additivity" / ds / model_fs_name(model) / "df_add.pkl",
            "rank": base / "pi_convergence" / "rank_stability_all_nodes.pkl",
            "kendall": base / "pi_convergence" / "rank_agreement_all_graphs.pkl",
        }

        for k, path in files.items():
            if path.exists():
                df = pd.read_pickle(path)
                df["dataset"] = ds
                df["model"] = model
                collectors[k].append(df)

    return {
        k: pd.concat(v, ignore_index=True) if v else pd.DataFrame()
        for k, v in collectors.items()
    }

# -----------------------------------------
# PER-MODEL PROCESSING
# -----------------------------------------
def process_model(model, model_dir, tex_dir, dfs):

    for df in dfs.values():
        if not df.empty:
            to_categorical(df, ["dataset", "model"])

    # Save raw data
    show_and_save(dfs["min"], model_dir, f"df_min_all_{model}", csv=True, pkl=True)
    show_and_save(dfs["add"], model_dir, f"df_add_all_{model}", csv=True,pkl=True)
    show_and_save(dfs["rank"], model_dir, f"df_rank_all_{model}", csv=True,pkl=True)
    show_and_save(dfs["kendall"], model_dir, f"df_kendall_all_{model}", csv=True,pkl=True)

    # Plots
    if not dfs["rank"].empty:
        show_and_save(plot_rank_stability(strip_data_prefix(dfs["rank"])),
                      model_dir, f"{model}_rank_stability")

    if not dfs["kendall"].empty:
        show_and_save(plot_kendall_w(strip_data_prefix(dfs["kendall"])),
                      model_dir, f"{model}_kendall")

    if not dfs["add"].empty:
        show_and_save(plot_mae_boxplot(strip_data_prefix(dfs["add"])),
                      model_dir, f"{model}_mae")

    if not dfs["min"].empty:
        show_and_save(plot_cardinality_ratio(strip_data_prefix(dfs["min"])),
                      model_dir, f"{model}_cardinality_ratio")

# -----------------------------------------
# METRICS TABLE
# -----------------------------------------
def create_combined_metric_table_compact(
    df_rank1: pd.DataFrame,   # rank_stability + most_common_rank
    df_rank: pd.DataFrame,    # kendalls_w
    df_mae: pd.DataFrame,     # MAE_gt1
    df_min: pd.DataFrame,     # Ratio + tau
    metric_mae: str = "MAE_gt1",
    metric_rank: str = "kendalls_w",
    metric_rs: str = "rank_stability",
    tau_for_cr: float = 0.1,
) -> pd.DataFrame:
    """
    Compact combined table.

    Rows:    model × dataset
    Columns: W, RS, RS_1, RS_2, MAE, CR_1
    """

    if df_rank.empty or df_rank1.empty or df_mae.empty or df_min.empty:
        return pd.DataFrame()


    df_rank  = normalize_model_dataset_names(df_rank)
    df_rank1 = normalize_model_dataset_names(df_rank1)
    df_mae   = normalize_model_dataset_names(df_mae)
    df_min   = normalize_model_dataset_names(df_min)

    # Metric computations
    w = (
        df_rank
        .groupby(["model", "dataset"], observed=True)[metric_rank]
        .apply(mean_pm_std)
        .rename(r"$\overline{W}$")
    )

    rs = (
        df_rank1
        .groupby(["model", "dataset"], observed=True)[metric_rs]
        .apply(mean_pm_std)
        .rename(r"$\overline{\mathrm{RS}}$")
    )

    rs1 = (
        df_rank1[df_rank1["most_common_rank"] == 1]
        .groupby(["model", "dataset"], observed=True)[metric_rs]
        .agg(["mean", "std"])
    )

    rs1 = rs1.apply(
        lambda r: f"{r['mean']:.2f} $\\pm$ {r['std']:.2f}"
        if not pd.isna(r["mean"])
        else "--",
        axis=1,
    ).rename(r"$\overline{\mathrm{RS}_1}$")

    rs2 = (
        df_rank1[df_rank1["most_common_rank"] == 2]
        .groupby(["model", "dataset"], observed=True)[metric_rs]
        .agg(["mean", "std"])
    )

    rs2 = rs2.apply(
        lambda r: f"{r['mean']:.2f} $\\pm$ {r['std']:.2f}"
        if not pd.isna(r["mean"])
        else "--",
        axis=1,
    ).rename(r"$\overline{\mathrm{RS}_2}$")

    mae = (
        df_mae
        .groupby(["model", "dataset"], observed=True)[metric_mae]
        .apply(mean_pm_std)
        .rename(r"$\overline{\mathrm{MAE}}$")
    )


    # CR_1
    df_min = df_min[df_min["tau"] == tau_for_cr]
    df_min["is_minimal"] = (df_min["Ratio"].round(2) == 1.00).astype(float)

    cr1 = (
        df_min
        .groupby(["model", "dataset"], observed=True)["is_minimal"]
        .mean()
        .rename(r"$\mathrm{CR}_1$")
    )

    # Combine (same spirit as combined table)
    table = pd.concat([rs, rs1, rs2, w, mae, cr1], axis=1).reset_index()

    table = table.sort_values(["model", "dataset"])
    table.iloc[:, 2:] = table.iloc[:, 2:].round(2)

    return table


# -----------------------------------------
# PERMUTATION TEST
# -----------------------------------------
def permutation_test_rank1_vs_rest(
    df: pd.DataFrame,
    n_perm: int = 5000,
    seed: int = 42,
):
    """
    Graph-level permutation test.

    H1: mean delta(G) > 0
    where:
        delta(G) = RS(rank=1) - RS(rank>1)
    """

    rng = np.random.default_rng(seed)

    # build graph-level arrays
    graphs = []

    for _, gdf in df.groupby("graph_id"):
        rs = gdf["rank_stability"].to_numpy()
        is_r1 = (gdf["most_common_rank"] == 1).to_numpy()

        if is_r1.sum() == 0 or is_r1.sum() == len(is_r1):
            continue

        graphs.append((rs, is_r1))

    n_graphs = len(graphs)

    if n_graphs < 3:
        return {
            "n_graphs": n_graphs,
            "mean_delta": np.nan,
            "p_perm": np.nan,
        }

    # observed statistic
    deltas_obs = np.array([
        rs[is_r1].mean() - rs[~is_r1].mean()
        for rs, is_r1 in graphs
    ])

    T_obs = deltas_obs.mean()

    # permutation
    T_perm = np.empty(n_perm)

    for b in range(n_perm):
        deltas_b = []

        for rs, is_r1 in graphs:
            perm = rng.permutation(is_r1)
            deltas_b.append(
                rs[perm].mean() - rs[~perm].mean()
            )

        T_perm[b] = np.mean(deltas_b)

    p_value = (np.sum(T_perm >= T_obs) + 1) / (n_perm + 1)

    return {
        "n_graphs": n_graphs,
        "mean_delta": float(T_obs),
        "p_perm": float(p_value),
    }

def load_or_compute_rs_perm_result(
    *,
    model: str,
    dataset: str,
    gdf: pd.DataFrame,
    base_dir: Path,
    n_perm: int = 5000,
    seed: int = 42,
):
    """
    Load cached permutation result or compute it.
    """

    model_safe = model.replace(":", "_")
    dataset_safe = dataset.replace("/", "_")

    out_dir = base_dir / model_safe / dataset_safe
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "rs_perm_result.pkl"

    if out_path.exists():
        logging.info(f"Loaded permutation result: {out_path}")
        with open(out_path, "rb") as f:
            return pickle.load(f)

    logging.info(f"Computing permutation test for {model} / {dataset}")

    perm_res = permutation_test_rank1_vs_rest(
        gdf,
        n_perm=n_perm,
        seed=seed
    )

    graphs = prepare_graph_arrays(gdf)
    f_dom = dominance_fraction_from_graphs(graphs)

    result = {
        "model": model,
        "dataset": dataset,
        "n_graphs": perm_res["n_graphs"],
        "mean_delta": perm_res["mean_delta"],
        "p_perm": perm_res["p_perm"],
        "dominant_frac": f_dom,
    }

    with open(out_path, "wb") as f:
        pickle.dump(result, f)

    logging.info(f"Saved permutation result: {out_path}")

    return result


def main():

    args = parse_args()
    setup_logging(args.log_level)

    RESULTS_DIR = args.results_dir
    SUMMARY_DIR = Path("summary_results")
    MODEL_DIR = SUMMARY_DIR / "per_model"
    TABLE_DIR = SUMMARY_DIR / "tables"

    ensure_dirs(MODEL_DIR, TABLE_DIR)

    global_collectors = {"min": [], "add": [], "rank": [], "kendall": []}

    # PER MODEL
    for model in args.models:

        logging.info(f"→ Model: {model}")
        model_dir = MODEL_DIR / model
        model_dir.mkdir(exist_ok=True)

        dfs = load_model_data(model, args.datasets, RESULTS_DIR)

        process_model(model, model_dir, args.tex_dir, dfs)

        for k in global_collectors:
            if not dfs[k].empty:
                global_collectors[k].append(dfs[k])

    # GLOBAL DATA
    df_min_all = pd.concat(global_collectors["min"], ignore_index=True)
    df_add_all = pd.concat(global_collectors["add"], ignore_index=True)
    df_rank_all = pd.concat(global_collectors["rank"], ignore_index=True)
    df_kendall_all = pd.concat(global_collectors["kendall"], ignore_index=True)

    show_and_save(df_min_all, SUMMARY_DIR, "df_min_all", csv=True, pkl=True)
    show_and_save(df_add_all, SUMMARY_DIR, "df_add_all", csv=True, pkl=True)
    show_and_save(df_rank_all, SUMMARY_DIR, "df_ranks_all", csv=True, pkl=True)
    show_and_save(df_kendall_all, SUMMARY_DIR, "df_kendall_all", csv=True, pkl=True)

    # PERMUTATION
    if not df_rank_all.empty:

        significance_rows = []

        for (model, dataset), gdf in df_rank_all.groupby(
                ["model", "dataset"], observed=True
        ):
            try:
                res = load_or_compute_rs_perm_result(
                    model=model,
                    dataset=dataset,
                    gdf=gdf,
                    base_dir=MODEL_DIR,
                    n_perm=5000,
                    seed=42,
                )
                significance_rows.append(res)

            except Exception as e:
                logging.error(
                    f"Permutation test failed for {model} / {dataset}: {e}"
                )

        df_rs_significance = pd.DataFrame(significance_rows)

        if not df_rs_significance.empty:
            # select + rename columns
            df_rs_significance = df_rs_significance[
                [
                    "model",
                    "dataset",
                    "n_graphs",
                    "mean_delta",
                    "dominant_frac",
                    "p_perm",
                ]
            ]

            # save
            show_and_save(
                df_rs_significance,
                TABLE_DIR,
                "rank_stability_perm_significance",
                csv=True,
            )

    # METRICS TABLE
    metrics_compact = create_combined_metric_table_compact(
        df_rank_all,
        df_kendall_all,
        df_add_all,
        df_min_all,
    )

    show_and_save(
        metrics_compact,
        TABLE_DIR,
        "combined_metrics_compact",
        csv=True,
    )


if __name__ == "__main__":
    main()