"""
evaluation/additivity.py

Empirical evaluation of the additivity assumption used in the paper.

This module analyzes whether the semantic dissimilarity induced by
a set of node interventions can be approximated by the sum of
individual node-level contributions.

"""

import itertools
import math
import random

import numpy as np
import seaborn as sns
import pandas as pd

from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from IPython.display import display

from graphrag.utils.graph_model import Graph
from counterfactual.relevance_distribution import (
    approx_node_relevance_distribution,
    compute_dissimilarity,
)
from counterfactual.cache_utils import (
    OpsInput,
    normalize_ops,
    key_for_graph,
    key_for_modification,
)
from counterfactual.GraphUtils import GraphUtils
from experiments.experiments_utils import show_and_save
import logging
base_logger = logging.getLogger("experiment.additivity")

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
MAX_SUBSET_SIZE = 3
N_SAMPLES_PER_SIZE = 200

# ---------------------------------------------------------
# SUBSET HELPERS
# ---------------------------------------------------------
def all_subsets(nodes, k):
    """
    Generate all possible subsets of size k from the given nodes.
    """
    return list(itertools.combinations(nodes, k))

def sample_subsets(nodes, k, n):
    """
    Randomly sample up to n subsets of size k from the given nodes.
    If the total number of subsets is smaller than n, returns all subsets.
    """
    total = int(math.comb(len(nodes), k))
    if total <= n:
        return all_subsets(nodes, k)

    seen = set()
    samples = []
    while len(samples) < n:
        s = tuple(sorted(random.sample(nodes, k)))
        if s not in seen:
            seen.add(s)
            samples.append(s)
    return samples

# -----------------------------------------------------------
# ADDITIVITY EVALUATION
# -----------------------------------------------------------
def evaluate_additivity(
    graphs: list[Graph],
    questions: list[str],
    model: str,
    operations: OpsInput = None,
    raw_deltas: dict[int, dict] | None = None,
    max_subset_size: int | None = None
) -> pd.DataFrame:
    """

    Empirically evaluate the additivity assumption.

    The additivity assumption states that the semantic dissimilarity
    induced by jointly removing a set of nodes N can be approximated
    by the sum of individual node-level dissimilarities.

    This function compares:
            y_true = DELTA(N; G', q)
            y_hat  = sum_{n ∈ N} delta(n; G', q)

    across different subset sizes and graphs.
    """

    base_logger.info("Start additivity evaluation | model=%s | graphs=%d", model, len(graphs))

    operations_norm = normalize_ops(operations)
    all_rows = []

    for idx, (graph, question) in enumerate(zip(graphs, questions)):
        graph_id = f"graph_{idx+1}"
        logger = logging.LoggerAdapter(base_logger, {"graph_id": graph_id})

        nodes = [n.id for n in graph.nodes]
        V = len(nodes)

        logger.info("Evaluating additivity | Graph %s | nodes=%d", graph_id, V)

        key_orig = key_for_graph(graph)

        # Node-level deltas
        logger.debug("Computing node deltas for %s", graph_id)
        deltas = (
            raw_deltas[idx]
            if raw_deltas is not None
            else approx_node_relevance_distribution(
                graph, question, model, operations_norm, "raw"
            )
        )

        delta_sum = {
            nid: sum(deltas[nid][op] for op in operations_norm)
            for nid in nodes
        }

        mod_graph_cache = {}
        rows = []

        # Max Subset Size
        K_max = V if max_subset_size is None else min(max_subset_size, V)
        logger.info("Graph %s | K_max=%d", graph_id, K_max)

        for k in range(1, K_max + 1):
            total_subsets_k = math.comb(V, k)
            logger.debug("Evaluating subsets | k=%d", k)

            # For large graphs, it is possible to select only a random subset of all possible nodes
            # to keep the computation tractable.
            if max_subset_size is not None and total_subsets_k > N_SAMPLES_PER_SIZE:
                subsets = sample_subsets(nodes, k, N_SAMPLES_PER_SIZE)
                logger.debug("Graph %s | k=%d | sampling %d/%d subsets", graph_id, k, len(subsets), total_subsets_k)
            else:
                subsets = all_subsets(nodes, k)
                logger.debug("Graph %s | k=%d | evaluating ALL %d subsets", graph_id, k, len(subsets))

            for N in subsets:
                key = tuple(sorted(N))
                if key not in mod_graph_cache:
                    g_mod = graph
                    for nid in key:
                        for op in operations_norm:
                            g_mod = GraphUtils.alpha(g_mod, nid, operation=op)
                    mod_graph_cache[key] = g_mod

                g_mod = mod_graph_cache[key]
                key_mod = key_for_modification(
                    graph, operations_norm, key, mode="parallel"
                )

                res = compute_dissimilarity(
                    graph, g_mod, question, model, key_orig, key_mod
                )

                y_true = float(res["dissimilarity"])
                # Compare true dissimilarity with additive approximation
                y_hat = sum(delta_sum[nid] for nid in key)

                rows.append({
                    "subset": key,
                    "subset_size": k,
                    "y_true": y_true,
                    "y_hat": y_hat,
                    "abs_error": abs(y_true - y_hat)
                })

        df_subsets = pd.DataFrame(rows)

        total_possible = 2**V - 1 if max_subset_size is None else sum(
            math.comb(V, kk) for kk in range(1, K_max + 1)
        )

        # MAE over all subsets
        MAE_all = df_subsets["abs_error"].mean() if not df_subsets.empty else 0
        # MAE over subsets with size > 1
        MAE_gt1 = df_subsets[df_subsets["subset_size"] > 1]["abs_error"].mean() \
            if not df_subsets[df_subsets["subset_size"] > 1].empty else 0

        logger.info("Additivity result | Graph %s, MAE=%.4f | MAE_gt1=%.4f", graph_id, MAE_all, MAE_gt1)

        all_rows.append(pd.DataFrame({
            "graph_id": [graph_id],
            "question": [question],
            "n_nodes": [V],
            "max_subset_size": [K_max],
            "total_subsets": [total_possible],
            "n_subsets": [len(df_subsets)],
            "coverage": [len(df_subsets)/total_possible],
            "MAE": [MAE_all],
            "MAE_gt1": [MAE_gt1],
            "details": [df_subsets]
        }))

    base_logger.info("Finished additivity evaluation")
    return pd.concat(all_rows, ignore_index=True)

def print_additivity_summary(df_add: pd.DataFrame):
    display(df_add[["graph_id", "n_nodes", "MAE"]])

def analyze_additivity_subsets(df_add, graph_ids, add_subsets_dir, display_obj=True, color_by_subset=False):
    """
    Automatically create scatterplot for selected graphs.
    """
    for graph_id in graph_ids:
        # folder
        subsets_graph_dir = add_subsets_dir / f"{graph_id}"
        subsets_graph_dir.mkdir(exist_ok=True)

        # plot & table
        fig_add_subsets, table_add_subsets = plot_additivity_subsets(df_add.round(4), graph_id=graph_id, color_by_subset_size=color_by_subset)

        # Save
        show_and_save(fig_add_subsets, filename=f"add_subsets_{graph_id}", path=subsets_graph_dir, display_obj=display_obj)
        show_and_save(table_add_subsets, filename=f"add_subsets_{graph_id}", path=subsets_graph_dir,
                      csv=True, tex=True, display_obj=display_obj)

        print(f"Processed additivity subsets for {graph_id}")


# --------------------------------------------
# PLOT FUNCTION:  Additivity of node subsets
# --------------------------------------------
def plot_additivity_subsets(
        df_add: pd.DataFrame,
        graph_id: str,
        color_by_subset_size: bool = True
) -> tuple[plt.Figure, pd.DataFrame]:
    """
    Scatter plot of predicted vs. true dissimilarities for all node subsets for a selected graph.
    """
    row = df_add[df_add["graph_id"] == graph_id]

    if row.empty:
        raise ValueError(f"Graph '{graph_id}' not found in results.")
    if len(row) > 1:
        raise ValueError(f"Multiple rows found for graph_id '{graph_id}'. Graph_id must be unique.")

    df_details = row.iloc[0]["details"]

    if df_details is None or df_details.empty:
        print(f"No subset data to plot for graph '{graph_id}'.")

    # metrics
    # mae = row.iloc[0]["MAE"]
    mae_gt1 = row.iloc[0]["MAE_gt1"]
    n_nodes = row.iloc[0]["n_nodes"]
    n_subsets = row.iloc[0]["n_subsets"]

    # scatterplot
    fig_scatter, ax = plt.subplots(figsize=(7, 6))

    if color_by_subset_size:
        # discrete colors
        unique_sizes = sorted(df_details["subset_size"].unique())
        cmap = plt.get_cmap("viridis", len(unique_sizes))
        size_to_color = {size: cmap(i) for i, size in enumerate(unique_sizes)}
        colors = df_details["subset_size"].map(size_to_color)
        ax.scatter(
            df_details["y_hat"], df_details["y_true"],
            c=colors, alpha=0.8, s=40
        )

        # legend
        for size in unique_sizes:
            if size == 1:
                ax.scatter([], [], c=[size_to_color[size]], label=f"{size} node", s=40)
            else:
                ax.scatter([], [], c=[size_to_color[size]], label=f"{size} nodes", s=40)

        ax.legend(title="Subset Size", bbox_to_anchor=(1.05, 1), loc='upper left', handletextpad=0.1)

    else:
        # one colored scatter
        ax.scatter(
            df_details["y_hat"], df_details["y_true"],
            color="skyblue", alpha=0.8, s=40
        )

    maxv = max(df_details["y_hat"].max(), df_details["y_true"].max(), 1.0)
    ax.plot([0, maxv], [0, maxv], "r--", linewidth=1)

    ax.set_xlabel("Predicted Dissimilarity")
    ax.set_ylabel("True Dissimilarity")
    ax.set_title(f"True vs Predicted Dissimilarity - {graph_id}")
    ax.grid(True)

    # metrics below scatterplot
    metrics_text = (
       # f"MAE = {mae:.4f}    "
        f"MAE (Subsets>1) = {mae_gt1:.4f}    "
        f"Number of Nodes = {n_nodes}    "
        f"Number of Subsets = {n_subsets}"
    )
    ax.text(
        0.0, -0.18, metrics_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11
    )

    plt.tight_layout(rect=(0, 0.15, 1, 1))

    # table
    df_display = df_details[[
        "subset", "subset_size", "y_true", "y_hat", "abs_error"
    ]].copy().round(4)

    return fig_scatter, df_display

# ----------------------------------------
# PLOT FUNCTION: MAE Distribution
# ----------------------------------------
def plot_mae_distributions(df_add: pd.DataFrame, mae_column: str = "MAE") -> plt.Figure:
    """
    Histogram of Mean Absolute Error (MAE) across all graphs.
    mae_column: Column name to use for MAE values ("MAE" or "MAE_gt1")
    """
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))

    sns.histplot(data=df_add, x=mae_column, bins=30, kde=True, ax=ax)
    ax.set_title(f"Histogram of MAE")
    ax.set_xlabel("Mean Absolute Error (MAE)")
    ax.set_ylabel("Frequency")

    plt.tight_layout()
    return fig

# ----------------------------------------
# PLOT FUNCTION: MAE by Graph Size
# ----------------------------------------
def plot_mae_by_graph_size(df_add: pd.DataFrame, mae_column: str = "MAE") -> plt.Figure:
    """
    Boxplot of MAE grouped by graph size.
    mae_column: Column name to use for MAE values ("MAE" or "MAE_gt1")
    """
    sizes = sorted(df_add["n_nodes"].unique())

    data = [df_add[df_add["n_nodes"] == n][mae_column].dropna() for n in sizes]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot(data)
    ax.set_xticks(range(1, len(sizes)+1))
    ax.set_xticklabels(sizes)
    ax.set_xlabel("Graph Size")
    ax.set_ylabel("Mean Absolute Error (MAE)")
    ax.set_title(f"Boxplot of MAE by Graph Size")
    ax.grid(True)
    plt.tight_layout()
    return fig

# ----------------------------------------
# PLOT FUNCTION: MAE vs Subset Size
# ----------------------------------------
def plot_mae_by_subset_size(df_add: pd.DataFrame, use_subset_gt1: bool = True) -> plt.Figure:
    """
    Boxplot of MAE for subsets of different sizes.
    use_subset_gt1: If True, use only abs_error for subsets with size >1, else use all
    """
    sns.set_theme(style="whitegrid")
    mae_rows = []

    for _, row in df_add.iterrows():
        df_sub = row["details"].copy()
        df_sub["graph_id"] = row["graph_id"]
        if use_subset_gt1:
            df_sub = df_sub[df_sub["subset_size"] > 1]
        mae_rows.append(df_sub[["graph_id", "subset_size", "abs_error"]])

    df_mae = pd.concat(mae_rows, ignore_index=True)

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    sns.boxplot(data=df_mae, x="subset_size", y="abs_error", ax=ax, color="lightblue")
    ax.set_xlabel("Subset Size")
    ax.set_ylabel("Mean Absolute Error (MAE)")
    title_suffix = " (Subsets>1)" if use_subset_gt1 else ""
    ax.set_title(f"Boxplot of MAE vs Subset Size{title_suffix}")
    ax.grid(True)

    plt.tight_layout()
    return fig


