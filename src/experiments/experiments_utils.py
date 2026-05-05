"""
experiments/experiments_utils.py

Shared utilities for running experiments, logging results,
and saving figures and tables.

"""
from IPython.display import display
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np
import random
import re, os
import seaborn as sns
from datetime import datetime
import logging
from scipy.stats import mannwhitneyu
from scipy.stats import binomtest
import math
from scipy.stats import wilcoxon
import pickle
import hashlib
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------
# RUN MANAGEMENT / PATHS
# ---------------------------------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


RUN_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_(?P<model>.+)_(?P<dataset>data_.+)$"
)

def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")

def make_experiment_dir(base, dataset_path, model) -> Path:
    base = Path(base)
    dataset_path = Path(dataset_path)

    dataset_name = dataset_path.stem if dataset_path.suffix else dataset_path.name

    out = base / slugify(dataset_name) / slugify(model)
    out.mkdir(parents=True, exist_ok=True)

    return out

def make_run_folder(dataset: str, model: str, results_dir) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"{timestamp}_{model.replace(':', '_')}_{dataset}"
    run_folder = results_dir / folder_name
    run_folder.mkdir(parents=True, exist_ok=False)
    return run_folder

def latest_run_for_dataset(
    model: str,
    dataset: str,
    results_dir: Path
) -> Optional[Path]:
    """
    Finds the latest run folder of the form:
    <timestamp>_<model>_<dataset>
    """

    candidates = []

    for p in results_dir.iterdir():
        if not p.is_dir():
            continue

        m = RUN_RE.match(p.name)
        if not m:
            continue

        if m.group("model") == model and m.group("dataset") == dataset:
            candidates.append(p)

    if not candidates:
        return None

    # lexicographic sort works because timestamp is prefix
    return sorted(candidates, key=lambda p: p.name, reverse=True)[0]

# ---------------------------------------------------------
# LOGGING
# ---------------------------------------------------------
class SafeFormatter(logging.Formatter):
    def format(self, record):
        if not hasattr(record, "graph_id"):
            record.graph_id = "-"
        return super().format(record)


def setup_logger(name: str, log_dir: Path | None = None, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = SafeFormatter(
        "[%(asctime)s] [%(levelname)s] [%(graph_id)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / f"{name}.log")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

# ---------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------
def get_available_datasets():
    folders = os.listdir("results/minimality")

    datasets = [
        f.replace("data_", "")
        for f in folders
        if f.startswith("data_")
    ]

    return sorted(datasets)

def load_dataset_results(dataset_name, model):
    paths = {
        "df_min": f"results/minimality/data_{dataset_name}/{model}/df_min.pkl",
        "df_add": f"results/additivity/data_{dataset_name}/{model}/df_add.pkl",
        "rank_stability": f"results/consistency/data_{dataset_name}/{model}/pi_convergence/rank_stability_all_nodes.pkl",
        "rank_agreement": f"results/consistency/data_{dataset_name}/{model}/pi_convergence/rank_agreement_all_graphs.pkl",
    }

    dfs = {}

    for key, path in paths.items():
        if os.path.exists(path):
            df = pd.read_pickle(path)
            df["dataset"] = dataset_name
            dfs[key] = df
        else:
            print(f"Missing: {path}")
            dfs[key] = None

    return dfs

# ---------------------------------------------------------
# SAVING AND SUMMARIZE OUTPUTS
# ---------------------------------------------------------
def show_and_save(obj,  path: str = "", filename: str = None,
                  csv: bool = False, pkl: bool = False, tex: bool = False,
                  dpi: int = 200, display_obj: bool = False):

    """
    Displays and saves DataFrames or Matplotlib figures.

    This function standardizes output handling across all experiments
    to ensure consistent result storage and reproducibility.
    """
    save_path = Path(path)
    save_path.mkdir(parents=True, exist_ok=True)

    if isinstance(obj, pd.DataFrame):
        # Display DataFrame if requested
        if display_obj:
            display(obj.head(5))

        if any([csv, pkl, tex]) and filename is None:
            raise ValueError("Filename must be provided for saving DataFrames.")

        # Save in requested formats
        if csv:
            file_csv = save_path / f"{filename}.csv"
            obj.to_csv(file_csv, index=False)
            print(f"CSV saved: {file_csv}")
        if pkl:
            file_pkl = save_path / f"{filename}.pkl"
            obj.to_pickle(file_pkl)
            print(f"Pickle saved: {file_pkl}")
        if tex:
            file_tex = save_path / f"{filename}.tex"
            df_safe = obj.copy()
            # fix _
            df_safe.columns = [c.replace("_", r"\_") for c in df_safe.columns]
            for col in df_safe.select_dtypes(include="object").columns:
                df_safe[col] = df_safe[col].apply(lambda x: x.replace("_", r"\_") if isinstance(x, str) else x)
            df_safe.to_latex(file_tex, index=False)
            print(f"LaTeX saved: {file_tex}")

    elif isinstance(obj, plt.Figure):
        if any([csv, pkl, tex]):
            raise ValueError("Saving options csv/pkl/tex not valid for Figures. Only PNG is supported.")
        if filename is None:
            raise ValueError("Filename must be provided for saving Figures.")

        file_path = save_path / f"{filename}.png"
        obj.savefig(file_path, dpi=dpi, bbox_inches="tight")
        if display_obj:
            display(obj)
        plt.close(obj)
        print(f"Figure saved: {file_path}")

    else:
        raise TypeError("Object must be a pd.DataFrame or matplotlib.figure.Figure.")

def summarize_rank_stability(graphs, graph_ids=None, pi_convergence_dir=None) -> pd.DataFrame:
    """
    Summarize the rank stability measures across all graphs.

    Parameters:
    - graphs: list of graph objects
    - graph_ids: optional, int, str, Path, or list; if None, automatically detected
    - pi_convergence_dir: Path to folder containing graph subfolders
    """
    if pi_convergence_dir is None:
        raise ValueError("pi_convergence_dir must be provided")

    pi_convergence_dir = Path(pi_convergence_dir)
    graph_ids = _normalize_graph_ids(graph_ids)

    # Automatically find all subfolders named “graph_*” if None
    if graph_ids is None:
        graph_ids = sorted([p.name for p in pi_convergence_dir.iterdir() if p.is_dir() and p.name.startswith("graph_")])

    rows = []

    for graph_id in graph_ids:
        # Bestimme Index und Ordnername
        if isinstance(graph_id, int):
            idx = graph_id
            folder_id = f"graph_{idx+1}"
        elif isinstance(graph_id, Path):
            folder_id = graph_id.name
            idx = int(folder_id.split("_")[1]) - 1
        else:  # string
            folder_id = graph_id
            idx = int(folder_id.split("_")[1]) - 1

        # Safetycheck
        if idx < 0 or idx >= len(graphs):
            print(f"Skipping {folder_id} (index out of range)")
            continue

        graph_dir = pi_convergence_dir / folder_id
        rank_stability_file = graph_dir / "rank_stability.pkl"

        if not rank_stability_file.exists():
            print(f"Skipping {folder_id} (no rank_stability.pkl)")
            continue

        df = pd.read_pickle(rank_stability_file)

        tmp = df[["node_id", "most_common_rank", "rank_stability"]].copy()
        tmp["graph_id"] = folder_id
        tmp["n_nodes"] = df["node_id"].nunique()

        rows.append(tmp)

    if not rows:
        return pd.DataFrame()

    df_all = pd.concat(rows, ignore_index=True)

    if 'show_and_save' in globals():
        show_and_save(df_all, filename="rank_stability_all_nodes", path=pi_convergence_dir, csv=True, pkl=True)

    return df_all

def summarize_rank_agreement(graphs, graph_ids=None, pi_convergence_dir=None) -> pd.DataFrame:
    """
    Summarize rank agreement across all graphs.

    Parameters:
    - graphs: list of graph objects
    - graph_ids: optional, int, str, Path, or list; if None, automatically detected
    - pi_convergence_dir: Path to folder containing graph subfolders (required)
    """
    if pi_convergence_dir is None:
        raise ValueError("pi_convergence_dir must be provided")

    pi_convergence_dir = Path(pi_convergence_dir)
    graph_ids = _normalize_graph_ids(graph_ids)

    if graph_ids is None:
        graph_ids = sorted([p.name for p in pi_convergence_dir.iterdir() if p.is_dir() and p.name.startswith("graph_")])

    rows = []

    for graph_id in graph_ids:
        if isinstance(graph_id, int):
            idx = graph_id
            folder_id = f"graph_{idx+1}"
        elif isinstance(graph_id, Path):
            folder_id = graph_id.name
            idx = int(folder_id.split("_")[1]) - 1
        else:  # string
            folder_id = graph_id
            idx = int(folder_id.split("_")[1]) - 1

        if idx < 0 or idx >= len(graphs):
            print(f"Skipping {folder_id} (index out of range)")
            continue

        graph = graphs[idx]
        n_nodes = len(graph.nodes)

        graph_dir = pi_convergence_dir / folder_id
        rank_agreement_file = graph_dir / "rank_agreement.pkl"

        if not rank_agreement_file.exists():
            print(f"Skipping {folder_id} (no rank_agreement.pkl)")
            continue

        df = pd.read_pickle(rank_agreement_file)

        tmp = df[["kendalls_w"]].copy()
        tmp["graph_id"] = folder_id
        tmp["n_nodes"] = n_nodes

        rows.append(tmp)

    if not rows:
        return pd.DataFrame()

    df_all = pd.concat(rows, ignore_index=True)

    if 'show_and_save' in globals():
        show_and_save(df_all, filename="rank_agreement_all_graphs", path=pi_convergence_dir, csv=True, pkl=True)

    return df_all

# ---------------------------------------------------
# PLOT FUNCTIONS: EVALUATION PER MODEL X DATASETS
# ---------------------------------------------------
def plot_rank_stability(df_rank_stability_all):

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.lineplot(
        data=df_rank_stability_all,
        x="most_common_rank",
        y="rank_stability",
        hue="dataset",
        errorbar="ci",
        marker="o",
        ax=ax
    )

    min_rank = int(df_rank_stability_all["most_common_rank"].min())
    max_rank = int(df_rank_stability_all["most_common_rank"].max())
    ax.set_xticks(np.arange(min_rank, max_rank + 1, 1))

    ax.set_xlabel("Rank")
    ax.set_ylabel("Mean Rank Stability (%) ± 95% CI")
    ax.set_ylim(0, 100)
    #ax.set_title("Rank Stability")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(title="Dataset")

    fig.tight_layout()
    return fig

def plot_kendall_w(df_kendall_all):
    fig, ax = plt.subplots(figsize=(8, 5))

    sns.lineplot(
        data=df_kendall_all,
        x="n_nodes",
        y="kendalls_w",
        hue="dataset",
        errorbar="ci",
        marker="o",
        ax=ax
    )

    min_nodes = int(df_kendall_all["n_nodes"].min())
    max_nodes = int(df_kendall_all["n_nodes"].max())
    ax.set_xticks(np.arange(min_nodes, max_nodes + 1, 1))

    ax.set_xlabel("Graph Size")
    ax.set_ylabel("Mean Kendall's W ± 95% CI")
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(title="Dataset")

    fig.tight_layout()
    return fig

def plot_mae_boxplot(df_add_all):
    fig, ax = plt.subplots(figsize=(8, 5))
    order = (
        df_add_all.groupby("dataset", observed=True)["MAE_gt1"]
        .median()
        .sort_values()
        .index
    )

    sns.boxplot(
        data=df_add_all,
        y="dataset",
        x="MAE",
        order=order,
        orient="h",
        ax=ax
    )

    ax.set_xlabel("Mean Absolute Error (MAE)")
    ax.set_ylabel("Dataset")
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    fig.tight_layout()
    return fig

def plot_cardinality_ratio(df_min_all, tau_values=None):
    df = df_min_all.copy()

    # Filter for tau
    if tau_values is not None:
        if not isinstance(tau_values, (list, tuple, set)):
            tau_values = [tau_values]
        df = df[df["tau"].isin(tau_values)]

    if df.empty:
        raise ValueError(
            "No data left after filtering by tau_values. "
            "Check that the specified tau values exist in the dataframe."
        )


    df["Ratio_plot"] = df["Ratio"].round(2)

    na_mask = df["Ratio_plot"].isna()
    numeric_vals = sorted(df.loc[~na_mask, "Ratio_plot"].unique())

    categories = [f"{v:.2f}" for v in numeric_vals]
    if na_mask.any():
        categories.append("N/A")

    df.loc[na_mask, "Ratio_plot"] = "N/A"
    df.loc[~na_mask, "Ratio_plot"] = df.loc[~na_mask, "Ratio_plot"].map(lambda x: f"{x:.2f}")

    # categorical variable with a defined order
    df["Ratio_plot"] = pd.Categorical(df["Ratio_plot"],
                                      categories=categories,
                                      ordered=True)

    unique_taus = sorted(df["tau"].unique())
    datasets = df["dataset"].unique()
    palette = {ds: f"C{i}" for i, ds in enumerate(datasets)}
    palette["N/A"] = "red"

    # Plot
    if len(unique_taus) == 1:
        tau_str = unique_taus[0]
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(
            data=df,
            x="Ratio_plot",
            hue="dataset",
            ax=ax,
            order=categories,
            palette={**palette, "N/A": "red"},
        )
        ax.set_xlabel("Cardinality Ratio")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Cardinality Ratios for $\\tau = {tau_str}$")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.legend(title="Dataset")
        fig.tight_layout()

    else:
        n_cols = min(3, len(unique_taus))
        g = sns.FacetGrid(df, col="tau", col_wrap=n_cols, sharey=True, height=4)

        def countplot_with_palette(data, **kwargs):
            sns.countplot(
                data=data,
                x="Ratio_plot",
                hue="dataset",
                order=categories,
                palette={**palette, "N/A": "red"},
                dodge=True,
                **kwargs
            )

        g.map_dataframe(countplot_with_palette)
        g.set_axis_labels("Cardinality Ratio", "Frequency")
        g.add_legend(title="Dataset")
        g.set_titles(r"$\tau = {col_name}$")
        fig = g.figure
        fig.tight_layout()

    return fig

# -------------------------------------------------
# PERMUTATION TEST
# -------------------------------------------------
def format_p_value(p: float, max_decimals: int = 3) -> str:
    """
    Format p-values for publication with adaptive precision.
    """
    if p == 0 or p < 1e-300:
        return r"$<10^{-300}$"

    if p < 1e-3:
        exponent = int(math.floor(math.log10(p)))
        return rf"$<10^{{{exponent}}}$"

    if p < 0.01:
        return f"{p:.{max_decimals}f}"

    return f"{p:.2f}"

def prepare_graph_arrays(df: pd.DataFrame):
    """
    Precompute per-graph arrays:
    - rs: rank stability values
    - is_rank1: boolean mask
    """
    graphs = []

    for _, gdf in df.groupby("graph_id"):
        rs = gdf["rank_stability"].to_numpy()
        is_r1 = (gdf["most_common_rank"] == 1).to_numpy()

        # skip degenerate cases
        if is_r1.sum() == 0 or is_r1.sum() == len(is_r1):
            continue

        graphs.append((rs, is_r1))

    return graphs

def dominance_fraction_from_graphs(graphs):
    """
    Fraction of graphs with DELTA(G) > 0,
    where DELTA(G) = mean RS(rank=1) - mean RS(rank>1).
    """

    if len(graphs) == 0:
        return np.nan

    deltas = [
        rs[is_r1].mean() - rs[~is_r1].mean()
        for rs, is_r1 in graphs
    ]

    return (np.array(deltas) > 0).mean()

