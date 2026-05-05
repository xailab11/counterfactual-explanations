"""
evaluation/minimality.py

Evaluation of the minimality of counterfactual explanations.

This module compares the greedy counterfactual search strategy
against an optimal exhaustive search in terms of:
- cardinality of selected node sets,
- achieved semantic dissimilarity.
"""
import pandas as pd
import matplotlib.pyplot as plt
from counterfactual.counterfactual_explanation import greedy_counterfactual, optimal_counterfactual
from counterfactual.relevance_distribution import approx_node_relevance_distribution
from counterfactual.cache_utils import OpsInput, normalize_ops
from graphrag.utils.graph_model import Graph
from IPython.display import display
import matplotlib.colors as mcolors
import numpy as np
from pathlib import Path
import logging
base_logger = logging.getLogger("experiment.minimality")

# --------------------------------------
# MINIMALITY EVALUATION
# --------------------------------------
def evaluate_minimality(
    graphs: list[Graph],
    questions: list[str],
    tau: float,
    model=None,
    operations: OpsInput = None,
    debug: bool = False
) -> pd.DataFrame:
    """
    Compare greedy and optimal counterfactual explanations.
    """
    base_logger.info("Start minimality evaluation | tau=%.2f | graphs=%d", tau, len(graphs))

    assert len(graphs) == len(questions), \
        "graphs and questions must have same length"

    operations_norm = normalize_ops(operations)
    all_rows = []

    for idx, (graph, question) in enumerate(zip(graphs, questions)):
        graph_id = f"graph_{idx+1}"

        logger = logging.LoggerAdapter(base_logger, {"graph_id": graph_id})
        logger.info("Evaluating minimality| Graph %s", graph_id)

        # Compute relevance distributions ONCE
        pi_norm = approx_node_relevance_distribution(
            graph, question, model, operations_norm, "normalized"
        )
        pi_raw = approx_node_relevance_distribution(
            graph, question, model, operations_norm, "raw"
        )

        logger.debug("Graph %s | running greedy CF", graph_id)
        res_greedy = greedy_counterfactual(
            graph, tau, pi_norm, question, model,
            operations=operations_norm, debug=debug
        )

        logger.debug("Graph %s | running optimal CF", graph_id)
        res_optimal = optimal_counterfactual(
            graph, tau, question, model,
            operations=operations_norm, debug=debug
        )

        greedy_nodes = res_greedy.get("selected_nodes", [])
        optimal_nodes = res_optimal.get("selected_nodes", []) if res_optimal is not None else []

        len_greedy = len(greedy_nodes)
        len_optimal = len(optimal_nodes)
        # Ratio > 1 indicates that the greedy approach selects more nodes
        # than the optimal counterfactual explanation.
        ratio = len_greedy / len_optimal if len_optimal > 0 else float("nan")

        set_greedy = set(greedy_nodes)
        set_optimal = set(optimal_nodes)
        if set_greedy or set_optimal:
            node_overlap = len(set_greedy & set_optimal) / len(set_greedy | set_optimal)
        else:
            node_overlap = float("nan")

        raw_deltas_optimal = [
            pi_raw[nid]["remove"] for nid in optimal_nodes
        ]
        sum_delta_optimal = sum(raw_deltas_optimal)
        y_true = res_optimal.get("max_dissimilarity", None) if res_optimal is not None else None
        y_max_possible = res_optimal.get("max_score_reached", None) if res_optimal is not None else None

        if debug:
            print(f"=== Graph {graph_id} ===")
            print("Greedy size:", len_greedy)
            print("Optimal size:", len_optimal)
            print("Ratio:", ratio)

        logger.info("Minimality result | Graph %s | greedy=%d | optimal=%d | ratio=%.3f", graph_id, len_greedy, len_optimal, ratio)

        all_rows.append({
            "graph_id": graph_id,
            "Num_nodes": len(pi_raw) ,
            "question": question,
            "tau": tau,
            "Greedy_size": len_greedy,
            "Greedy_nodes": greedy_nodes,
            "Optimal_size": len_optimal,
            "Optimal_nodes": optimal_nodes,
            "Ratio": ratio,
            "Node_overlap": node_overlap,
            "Dissimilarity": y_true,
            "Max_dissimilarity": y_max_possible,
            "Raw_deltas": raw_deltas_optimal,
            "Sum_raw_deltas": sum_delta_optimal
        })

    base_logger.info("Minimality evaluation completed")
    return pd.DataFrame(all_rows)

def merge_minimality_dfs(exp_dir: Path, file_name: str = "df_min"):
    all_dfs = []

    # CSV files
    for csv_file in exp_dir.glob("df_min_*.csv"):
        df = pd.read_csv(csv_file)
        all_dfs.append(df)

    # PKL files
    for pkl_file in exp_dir.glob("df_min_*.pkl"):
        df = pd.read_pickle(pkl_file)
        all_dfs.append(df)

    if not all_dfs:
        raise ValueError(f"No df_min files found in {exp_dir}")

    # Concatenate all
    df_min_all = pd.concat(all_dfs, ignore_index=True)

    # Drop duplicates based on 'graph_id' and 'tau' only
    df_min_all = df_min_all.drop_duplicates(subset=["graph_id", "tau"])

    # Save combined
    df_min_all.to_csv(exp_dir / f"{file_name}.csv", index=False)
    df_min_all.to_pickle(exp_dir / f"{file_name}.pkl")

    print(f"Merged {len(all_dfs)} files into {file_name}.csv and {file_name}.pkl")
    #return df_min_all

def print_minimality_summary(df_min):
    display(df_min[["graph_id", "Num_nodes", "Greedy_size", "Optimal_size", "Ratio","Greedy_nodes", "Optimal_nodes", "Dissimilarity"]])

def approx_ratio_distribution(df_min: pd.DataFrame):
    ratio_counts = df_min["Ratio"].value_counts()
    ratio_share = df_min["Ratio"].value_counts(normalize=True)
    print("Counts per ratio:\n", ratio_counts)
    print("Relative frequency:\n", ratio_share)
    #return ratio_share

# -------------------------------------------------
# PLOT FUNCTION: CARDINALITY RATIO SCATTERPLOT
# -------------------------------------------------
def plot_minimality_scatter(df_min: pd.DataFrame, color_by_size: bool = False):
    """
    Visualizes the relationship between predicted and true dissimilarity
    to assess the quality of the additive approximation.
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    if color_by_size and "Num_nodes" in df_min.columns:
        c_vals = df_min["Num_nodes"].astype(int)
        unique_sizes = sorted(c_vals.unique())

        # discrete colors
        cmap = plt.get_cmap("viridis", len(unique_sizes))
        size_to_color = {size: cmap(i) for i, size in enumerate(unique_sizes)}
        colors = c_vals.map(size_to_color)

        # scatterplot
        ax.scatter(df_min["Sum_raw_deltas"], df_min["Dissimilarity"],
                   c=colors, alpha=0.7, s=40)

        # legend
        for size in unique_sizes:
            if size == 1:
                ax.scatter([], [], c=[size_to_color[size]], label=f"{size} node", s=40)
            else:
                ax.scatter([], [], c=[size_to_color[size]], label=f"{size} nodes", s=40)

        ax.legend(title="Graph Size", bbox_to_anchor=(1.05, 1), loc='upper left',  handletextpad=0.1)

    else:
        ax.scatter(df_min["Sum_raw_deltas"], df_min["Dissimilarity"], alpha=0.7, s=40)

    maxv = max(df_min["Sum_raw_deltas"].max(), df_min["Dissimilarity"].max(), 1.0)
    ax.plot([0, maxv], [0, maxv], 'r--', label='y=x')

    ax.set_xlabel("Predicted Dissimilarity")
    ax.set_ylabel("True Dissimilarity")
    ax.set_title("True vs Predicted Dissimilarity")
    ax.grid(True)

    fig.tight_layout()
    return fig

# -------------------------------------------------
# PLOT FUNCTION: CARDINALITY RATIO BARPLOT
# -------------------------------------------------
def plot_ratio_distribution(df_min, stacked=False):
    """
    Plot discrete ratio distribution.
    """
    if stacked:
        # Count per Ratio per Num_nodes
        pivot_df = df_min.groupby(["Ratio", "Num_nodes"]).size().unstack(fill_value=0).sort_index()

        cmap = plt.get_cmap("viridis_r")
        colors = cmap(np.linspace(0, 1, len(pivot_df.columns)))

        fig, ax = plt.subplots(figsize=(8,5))
        bottom = np.zeros(len(pivot_df), dtype=float)

        # fix x positions
        x_positions = np.arange(len(pivot_df))
        x_labels = [f"{r:.2f}" for r in pivot_df.index]

        for i, col in enumerate(pivot_df.columns):
            heights = pivot_df[col].to_numpy(dtype=float)
            bars = ax.bar(
                x=x_positions,
                height=heights,
                bottom=bottom,
                color=colors[i],
                edgecolor='black',
                label=f"{col} nodes"
            )

            # Counts mittig auf Segment schreiben
            for j in range(len(heights)):
                if heights[j] > 0:
                    ax.text(
                        x_positions[j],
                        bottom[j] + heights[j]/2,
                        str(int(heights[j])),
                        ha='center',
                        va='center',
                        fontsize=9,
                        color='white'
                    )

            bottom += heights

        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel("Cardinality Ratio")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of Cardinality Ratios")
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.legend(title="Graph Size", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        return fig

    else:
        # simple bars
        ratio_counts = df_min["Ratio"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(7,5))

        x_positions = np.arange(len(ratio_counts))
        x_labels = [f"{r:.2f}" for r in ratio_counts.index]

        heights = ratio_counts.to_numpy(dtype=float)
        ax.bar(x=x_positions, height=heights, color='skyblue', edgecolor='black')

        for i in range(len(heights)):
            ax.text(x_positions[i], heights[i] + 0.5, str(int(heights[i])), ha='center', va='bottom')

        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel("Cardinality Ratio")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of Cardinality Ratios")
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        return fig


