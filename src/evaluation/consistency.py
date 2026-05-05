"""
evaluation/consistency.py

Empirical analysis of the stability and consistency of node relevance
distributions across repeated LLM runs.

This module investigates whether the relevance distribution
and the induced node rankings are robust under different LLM runs.
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.stats import kendalltau
from matplotlib.colors import ListedColormap, BoundaryNorm
from experiments.experiments_utils import show_and_save
from pathlib import Path

from graphrag.utils.graph_model import Graph
from counterfactual.relevance_distribution import approx_node_relevance_distribution
from counterfactual.cache_utils import (
    VERBALIZATION_CACHE,
    EMBEDDING_CACHE,
    DISSIMILARITY_CACHE,
    OpsInput,
    normalize_ops,
)
import logging
base_logger = logging.getLogger("experiment.consistency")

# ------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------
def gini(x: np.ndarray) -> float:
     """
     Gini coefficient of a probability distribution.
     Measures concentration of relevance distribution: low -> relevance evenly distributed, high -> few nodes dominate.
     """
     if np.allclose(x.sum(), 0):
         return 0.0
     x = np.sort(x)
     n = len(x)
     index = np.arange(1, n + 1)
     return (2 * np.sum(index * x) / (n * np.sum(x))) - (n + 1) / n

def entropy(x: np.ndarray) -> float:
    """
    Shannon entropy of a probability distribution.
    Measures dispersion:
     low --> relevance concentrates on few nodes,
     high --> relevance spread over many nodes.
    """
    x = x[x > 0]
    return -np.sum(x * np.log(x + 1e-12))

def get_node_ranks(df_pi_runs: pd.DataFrame) -> pd.DataFrame:
    """
    Converts raw pi values into ranks per run.

    - Highest pi  --> rank 1
    - Ties get identical rank --> method = 'dense'
    """
    df_ranks = df_pi_runs.rank(
        axis=1,
        method='dense',
        ascending=False
    ).astype(int)
    return df_ranks

def compute_node_rank_stability(df_ranks: pd.DataFrame) -> pd.DataFrame:
    """
    Computes node-level rank stability across runs.
        Metrics:
        - std_rank            --> how much the rank fluctuates
        - range               --> worst - best rank
        - most_common_rank    --> modal rank
        - rank_stability      --> % of runs where node kept its modal rank

    High rank_stability --> the node consistently occupies the same importance position.
    """
    n_runs = df_ranks.shape[0]

    std_rank = df_ranks.std(axis=0)
    min_rank = df_ranks.min(axis=0)
    max_rank = df_ranks.max(axis=0)
    rank_range = max_rank - min_rank

    most_common_rank = df_ranks.apply(lambda col: col.mode().iloc[0])
    rank_stability = df_ranks.apply(
        lambda col: col.value_counts().max() / n_runs * 100
    )

    df_node = pd.DataFrame({
        "node_id": df_ranks.columns.astype(str),
        "std_rank": std_rank.values,
        "min_rank": min_rank.values,
        "max_rank": max_rank.values,
        "range": rank_range.values,
        "most_common_rank": most_common_rank.values,
        "rank_stability": rank_stability.values,
    })

    return df_node.sort_values("rank_stability", ascending=False)

def compute_node_pi_stability(df_pi_runs: pd.DataFrame) -> pd.DataFrame:
    """
    Computes stability of raw pi values per node.
    Metrics:
    - Mean pi
    - Std  pi
    - CV   = Std / Mean
    Low CV  --> stable relevance
    High CV --> unstable / sensitive node
    """
    df_node_conv = pd.DataFrame({
        "node_id": df_pi_runs.columns.astype(str),
        "Mean": df_pi_runs.mean().values,
        "Std": df_pi_runs.std(ddof=0).values,
    })

    df_node_conv["CV"] = df_node_conv["Std"] / df_node_conv["Mean"]

    df_node_conv = df_node_conv.sort_values("CV", ascending=False)

    return df_node_conv

def compute_rank_agreement(df_ranks: pd.DataFrame) -> dict:
    """
    Measures global agreement of node rankings across runs.

    Metrics:
        - Mean Spearman correlation
        - Mean Kendall-Tau
        - Kendall's W (overall concordance)

        Kendall's W:
            1   --> perfect agreement
            0   --> random rankings
        """
    # Spearman
    spearman_corr = df_ranks.T.corr(method="spearman")
    upper = spearman_corr.values[np.triu_indices_from(spearman_corr, k=1)]
    mean_spearman = float(np.mean(upper))

    # Kendall-Tau
    kendall_values = []
    for i in range(df_ranks.shape[0]):
        for j in range(i + 1, df_ranks.shape[0]):
            tau, _ = kendalltau(df_ranks.iloc[i], df_ranks.iloc[j])
            kendall_values.append(tau)
    mean_kendall_tau = float(np.mean(kendall_values))

    # Kendall's W
    R = df_ranks.values
    m, n = R.shape  # m = Runs, n = Nodes
    rank_sums = R.sum(axis=0)
    S = np.sum((rank_sums - rank_sums.mean()) ** 2)
    kendalls_w = float((12 * S) / (m**2 * (n**3 - n)))

    return pd.DataFrame([{
        "kendalls_w": kendalls_w,
        "mean_spearman": mean_spearman,
        "mean_kendall_tau": mean_kendall_tau
    }])

def estimate_pi_convergence_by_id(
    graphs,
    questions,
    graph_id,
    model,
    n_runs,
    pi_distribution_dir=None
):
    """
    Robust convergence estimation.

    run_0  -> precomputed pi distribution (if available)
    run_1+ -> fresh runs (use_cache=False)

    Includes extensive safety checks.
    """

    # Resolve graph index + folder id
    if isinstance(graph_id, int):
        idx = graph_id
        folder_id = f"graph_{idx+1}"
    else:
        idx = int(str(graph_id).split("_")[1]) - 1
        folder_id = str(graph_id)

    graph = graphs[idx]
    question = questions[idx]

    runs_dicts = []
    base_pi = None

    # Load precomputed distribution
    if pi_distribution_dir is not None:

        dist_path = (
            Path(pi_distribution_dir)
            / "distribution_details"
            / f"{folder_id}_pi.csv"
        )

        if dist_path.exists():

            print(f"[Convergence] Loading base π from {dist_path}")

            try:
                df0 = pd.read_csv(
                    dist_path,
                    dtype={"node_id": str, "pi": float}
                )

            except Exception as e:
                print(f"[ERROR] Could not read {dist_path}: {e}")
                df0 = None

            if df0 is not None:

                # Column check
                if not {"node_id", "pi"}.issubset(df0.columns):
                    raise ValueError(
                        f"{dist_path} must contain columns ['node_id','pi']"
                    )

                # Remove duplicate nodes
                if df0["node_id"].duplicated().any():
                    print(
                        f"[WARNING] Duplicate node IDs in {folder_id} distribution. "
                        "Keeping first occurrence."
                    )
                    df0 = df0.drop_duplicates(subset="node_id")

                # Remove invalid pi values
                df0 = df0[np.isfinite(df0["pi"])]


                # Handle probability mass correctly
                s = df0["pi"].sum()

                if s < 0:
                    # This is genuinely invalid
                    raise ValueError(
                        f"Negative total probability mass in {dist_path}"
                    )

                if s == 0:
                    # Legitimate degenerate distribution:
                    # no node removal had any effect
                    print(
                        f"[INFO] {folder_id}: Loaded π distribution with zero total mass "
                        "(node removals had no measurable effect)."
                    )

                else:
                    # Regular probabilistic case
                    # Normalize only if numerically necessary
                    if not np.isclose(s, 1.0):
                        df0["pi"] = df0["pi"] / s

                # Convert to dict
                base_pi = dict(zip(df0["node_id"], df0["pi"]))

                print(
                    f"[Convergence] {folder_id}: "
                    f"{len(base_pi)} nodes loaded, π sum={sum(base_pi.values()):.4f}"
                )

                runs_dicts.append(base_pi)

        else:
            print(f"[Convergence] No precomputed π for {folder_id}")

    # Fresh runs
    remaining = n_runs - 1 if base_pi is not None else n_runs

    for r in range(remaining):

        print(f"[Convergence] Computing run_{len(runs_dicts)} for {folder_id}")

        pi = approx_node_relevance_distribution(
            graph,
            question,
            model,
            return_type="normalized",
            use_cache=False
        )

        flat = {
            str(nid): float(v["remove"])
            for nid, v in pi.items()
        }

        runs_dicts.append(flat)

    # Align nodes across runs
    all_nodes = sorted(
        set().union(*[d.keys() for d in runs_dicts])
    )

    aligned_rows = []

    for d in runs_dicts:

        row = {nid: d.get(nid, 0.0) for nid in all_nodes}
        aligned_rows.append(row)

    df = pd.DataFrame(aligned_rows, columns=all_nodes)
    df.index = [f"run_{i}" for i in range(len(aligned_rows))]

    df = df.astype(float)

    # Final safety checks
    if df.isna().any().any():
        print(f"[WARNING] NaNs detected in {folder_id}")

    if not np.isfinite(df.values).all():
        print(f"[WARNING] Non-finite values detected in {folder_id}")

    print(
        f"[Convergence] Final matrix {folder_id}: "
        f"{df.shape[0]} runs × {df.shape[1]} nodes"
    )

    return df

def _normalize_graph_ids(graph_ids):
    """
    Helper function: converts graph_ids into a list.
    Accepts int, str, Path, or list of these types.
    """
    if graph_ids is None:
        return None
    if isinstance(graph_ids, (int, str, Path)):
        return [graph_ids]
    if isinstance(graph_ids, list):
        return graph_ids
    raise TypeError(f"Unsupported type for graph_ids: {type(graph_ids)}")

# -----------------------------------------------------------
# CONSISTENCY EVALUATION
# -----------------------------------------------------------
def analyze_convergence(
    graphs,
    questions,
    model,
    pi_convergence_dir,
    graph_ids=None,
    reload_existing=True,
    n_runs=10,
    display_obj=False
):
    """
    Main convergence pipeline.

    Computes pi/ranks, rank stability, pi stability, rank agreement,
    saves results, and plots for each selected graph.
    """

    base_logger.info("Start convergence analysis | model=%s | n_runs=%d", model, n_runs)

    if graph_ids is None:
        n_graphs = len(graphs) if not isinstance(graphs, dict) else len(graphs.keys())
        max_graphs = min(n_graphs, 100)
        graph_ids = list(range(max_graphs))  # 0-based indices
        base_logger.info("No graph_ids provided → using first %d graphs", len(graph_ids))

    for graph_id in graph_ids:

        # Specify the index and graph_id for folders/plots
        if isinstance(graph_id, int):
            idx = graph_id
            folder_id = f"graph_{idx+1}"
        else:
            idx = int(graph_id.split("_")[1]) - 1
            folder_id = graph_id

        logger = logging.LoggerAdapter(base_logger, {"graph_id": graph_id})
        logger.info("Starting convergence analysis")

        # Make folder
        graph_dir = pi_convergence_dir / folder_id
        graph_dir.mkdir(exist_ok=True)

        # Path to files
        node_ranks_file = graph_dir / "node_ranks.pkl"
        pi_runs_file = graph_dir / "pi_runs.pkl"
        pi_stability_file = graph_dir / "pi_stability.pkl"
        rank_stability_file = graph_dir / "rank_stability.pkl"
        rank_agreement_file = graph_dir / "rank_agreement.pkl"

        # Load existing or compute
        if reload_existing and pi_runs_file.exists():
            logger.info("Loading cached convergence artifacts")
            df_pi_runs = pd.read_pickle(pi_runs_file)
            pi_stability = pd.read_pickle(pi_stability_file)
            df_node_ranks = pd.read_pickle(node_ranks_file)
            rank_stability = pd.read_pickle(rank_stability_file)
            rank_agreement = pd.read_pickle(rank_agreement_file)
            print(f"Loaded existing data for {folder_id}")
        else:
            logger.info("Graph %s | computing convergence (n_runs=%d)", folder_id, n_runs)
            df_pi_runs = estimate_pi_convergence_by_id(graphs, questions, graph_id, model, n_runs, pi_distribution_dir=(pi_convergence_dir.parent / "pi_distribution"))
            logger.debug("Computing rank and stability metrics")
            df_node_ranks = get_node_ranks(df_pi_runs)
            pi_stability = compute_node_pi_stability(df_pi_runs)
            rank_stability = compute_node_rank_stability(df_node_ranks)
            rank_agreement = compute_rank_agreement(df_node_ranks)

            # Save results
            logger.debug("Saving convergence results")
            show_and_save(df_node_ranks, filename="node_ranks", path=graph_dir, csv=True, pkl=True, display_obj=False)
            show_and_save(df_pi_runs, filename="pi_runs", path=graph_dir, csv=True, pkl=True, display_obj=False)
            show_and_save(pi_stability, filename="pi_stability", path=graph_dir, csv=True, pkl=True, display_obj=False)
            show_and_save(rank_stability, filename="rank_stability", path=graph_dir, csv=True, pkl=True, display_obj=False)
            show_and_save(rank_agreement, filename="rank_agreement", path=graph_dir, csv=True, pkl=True, display_obj=False)
            print(f"Computed and saved data for {folder_id}")

        # Plots
        logger.info("Generating convergence plots")
        fig_pi_convergence = plot_pi_convergence_heatmap_with_cv(df_pi_runs, pi_stability, folder_id)
        fig_rank_convergence = plot_heatmap_rank_stability(df_node_ranks, rank_stability, folder_id)

        show_and_save(fig_pi_convergence, filename=f"pi_convergence_{folder_id}", path=graph_dir, display_obj=display_obj)
        show_and_save(fig_rank_convergence, filename=f"rank_convergence_{folder_id}", path=graph_dir, display_obj=display_obj)
        logger.info("Convergence analysis finished")

    base_logger.info("All convergence analyses finished")

# ------------------------------------------------------------
# DISTRIBUTIONAL ANALYSIS OF Pi PER GRAPH
# ------------------------------------------------------------
def compute_pi_descriptive_measures(
    graphs: list[Graph],
    questions: list[str],
    model: str,
    operations: OpsInput = None,
    clear_cache: bool | None = False
) -> pd.DataFrame:
    """
    Computes node relevance distribution and their descriptive statistics for each graph.

        Each graph is summarized by:

        - Top1_Relevance  --> dominance of most relevant node
        - Gini_Index      --> inequality of relevance distribution
        - Entropy         --> spread of distribution
        - CV              --> dispersion

        Interpretation:
        - High Top1 + high Gini --> strong concentration on few nodes
        - High Entropy          --> more distributed reasoning
        """
    base_logger.info("Starting PI descriptive measures | graphs=%d | model=%s | clear_cache=%s", len(graphs), model,
                     clear_cache)

    operations = normalize_ops(operations)
    rows = []

    if clear_cache:
        base_logger.warning("Clearing caches before PI computation")
        VERBALIZATION_CACHE.clear()
        EMBEDDING_CACHE.clear()
        DISSIMILARITY_CACHE.clear()

    for idx, (graph, question) in enumerate(zip(graphs, questions)):
        graph_id = f"graph_{idx + 1}"
        logger = logging.LoggerAdapter(base_logger, {"graph_id": graph_id})
        logger.info("Computing PI distribution | Graph %s", graph_id)

        pi = approx_node_relevance_distribution(
            graph,
            question,
            model,
            operations,
            return_type="normalized",
            use_cache = True
        )

        #values = np.array([float(v["remove"]) for v in pi.values()])
        #values = np.sort(values)[::-1]  # descending

        df_pi = pd.DataFrame.from_dict(
            {
                nid: {
                    "pi": float(v["remove"]),
                }
                for nid, v in pi.items()
            },
            orient="index",
        )
        df_pi.index.name = "node_id"
        df_pi = df_pi.sort_values("pi", ascending=False)

        values = df_pi["pi"].values

        n = len(values)
        mean_val = values.mean()
        std_val = values.std(ddof=0)  # Population Std
        cv_val = std_val / mean_val if mean_val > 0 else np.nan # Coefficient of Variation

        logger.debug("Stats | n=%d | top1=%.4f | gini=%.4f | entropy=%.4f | cv=%.4f", n, values[0], gini(values),
                     entropy(values), cv_val)

        row = {
            "graph_id": f"graph_{idx+1}",
            "n_nodes": n,

            # the maximal probability of a single node
            "Top1_Relevance": values[0],

            # form/inequality
            "Gini_Index": gini(values),
            "Entropy": entropy(values),

            # dispersion
            "CV": cv_val,

        }

        rows.append(row)
        row["pi"] = df_pi

    base_logger.info("Finished PI descriptive measures")
    return pd.DataFrame(rows)

# ------------------------------------------------------------
# PLOT FUNCTION: NODE RELEVANCE HEATMAP
# ------------------------------------------------------------
def plot_pi_convergence_heatmap(df_pi_runs: pd.DataFrame) -> plt.Figure:
    """
    Heatmap of node relevance distribution across different LLM runs.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    sns.heatmap(
        df_pi_runs.T,
        cmap="viridis",
        cbar_kws={"label": "π"},
        ax=ax
    )

    ax.set_xlabel("LLM run")
    ax.set_ylabel("Node ID")
    ax.set_title("Convergence of Node Relevance Distribution")

    plt.tight_layout()
    return fig

def plot_pi_convergence_heatmap_with_cv(
    df_pi_runs: pd.DataFrame,
    df_node_conv: pd.DataFrame,
    graph_id: str
) -> plt.Figure:
    """
    Heatmap of node relevance distribution across different LLM runs
    + barplot of CV per Node.
    Heatmap und Barplot haben identische Breitenverhältnisse wie in rank_stability.
    """
    df_pi_runs.columns = df_pi_runs.columns.astype(str)

    # Node order sorted by CV in descending order
    node_order = df_node_conv.sort_values("CV", ascending=False)["node_id"].tolist()
    df_heatmap = df_pi_runs[node_order].T
    df_node_conv_indexed = df_node_conv.set_index("node_id").loc[node_order]

    # Consistent Layout
    fig, axes = plt.subplots(
        ncols=2,
        figsize=(16, 6),
        gridspec_kw={"width_ratios": [3, 1]},
        sharey=False
    )

    # Heatmap
    sns.heatmap(
        df_heatmap,
        cmap="viridis",
        cbar_kws={"label": "$\pi$", "shrink": 0.8},
        ax=axes[0],
        annot=True,
        fmt=".2f",
        annot_kws={"size": 8},
        linewidths=0.5,
        square=False,
    )
    axes[0].set_xlabel("LLM Run")
    axes[0].set_ylabel("Node ID")
    axes[0].set_title(f"Relevance Distribution Across Runs for {graph_id}")
    axes[0].set_yticklabels(node_order, rotation=0)

    # CV Barplot
    axes[1].barh(
        df_node_conv_indexed.index,
        df_node_conv_indexed["CV"],
        color="tab:red"
    )
    axes[1].set_xlabel("Coefficient of Variation")
    axes[1].set_ylabel("Node ID")
    axes[1].set_title("$\pi$ Instability per Node")
    axes[1].invert_yaxis()  # gleiche Orientierung wie Heatmap

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    return fig

# ------------------------------------------------------------
# PLOT FUNCTION: NODE RANK STABILITY
# ------------------------------------------------------------
def plot_heatmap_rank_stability(
    df_ranks: pd.DataFrame,
    df_rank_stability: pd.DataFrame,
    graph_id: str,
) -> plt.Figure:
    """
    Heatmap of ranks across different LLM runs
    + barplot of rank stability per Node.
    Heatmap und Barplot haben identische Breitenverhältnisse wie in pi_stability.
    """
    df_ranks.columns = df_ranks.columns.astype(str)

    # Node order sorted by rank stability in ascending order
    node_order = df_rank_stability.sort_values("rank_stability", ascending=True)["node_id"].tolist()

    df_ranks_sorted = df_ranks[node_order]
    df_node_sorted = df_rank_stability.set_index("node_id").loc[node_order]

    # Consistent Layout
    fig, axes = plt.subplots(
        ncols=2,
        figsize=(16, 6),
        gridspec_kw={"width_ratios": [3, 1]},
        sharey=False
    )

    # Discrete Colormap for ranks
    max_rank = int(df_ranks_sorted.max().max())
    min_rank = int(df_ranks_sorted.min().min())
    n_ranks = max_rank - min_rank + 1
    viridis = plt.get_cmap("viridis_r", n_ranks)
    cmap = ListedColormap([viridis(i) for i in range(n_ranks)])
    norm = BoundaryNorm(np.arange(min_rank - 0.5, max_rank + 1.5, 1), cmap.N)

    sns.heatmap(
        df_ranks_sorted.T,
        annot=True,
        fmt=".0f",
        cmap=cmap,
        norm=norm,
        cbar_kws={
            "ticks": np.arange(min_rank, max_rank + 1),
            "label": "Rank"
        },
        linewidths=0.5,
        ax=axes[0]
    )
    axes[0].set_title(f"Node Ranks Across Runs for {graph_id}")
    axes[0].set_xlabel("LLM Run")
    axes[0].set_ylabel("Node ID")
    axes[0].set_yticklabels(node_order, rotation=0)

    # Barplot der Rank-Stabilität
    axes[1].barh(
        df_node_sorted.index,
        df_node_sorted["rank_stability"]
    )
    axes[1].set_xlim(0, 100)
    axes[1].set_title("Rank Stability per Node")
    axes[1].set_xlabel("Rank Stability %")
    axes[1].set_ylabel("Node ID")
    axes[1].invert_yaxis()

    plt.tight_layout()
    return fig

def plot_node_rank_stability_aggregated(df):
    grouped = df.groupby('n_nodes')['rank_stability'].agg(['mean','std']).reset_index()

    fig, ax = plt.subplots(figsize=(10,4))
    ax.errorbar(
        grouped['n_nodes'],
        grouped['mean'],
        yerr=grouped['std'],
        fmt='o-',
        capsize=5
    )
    ax.set_xlabel("Graph Size (number of nodes)")
    ax.set_ylabel("Mean Node Rank Stability (%)")
    ax.set_title("Aggregated Node-Level Rank Stability")
    ax.set_ylim(0, 100)
    fig.tight_layout()
    return fig

def plot_node_rank_stability(df_rank_all):
    df = df_rank_all.copy()
    df["most_common_rank"] = df["most_common_rank"].astype(int)

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.boxplot(
        data=df,
        x="most_common_rank",
        y="rank_stability",
        ax=ax
    )

    sns.stripplot(
        data=df,
        x="most_common_rank",
        y="rank_stability",
        color="black",
        alpha=0.4,
        ax=ax
    )

    ax.set_xlabel("Rank")
    ax.set_ylabel("Rank Stability (%)")
    #ax.set_title("Rank Stability by Rank Position")
    ax.set_ylim(0, 100)

    plt.tight_layout()
    return fig

# ------------------------------------------------------------
# PLOT FUNCTION: KENDALLS W
# ------------------------------------------------------------
def plot_all_kendalls_w(df):
    """
    Barplot Kendall's W per Graph.
    """
    fig, ax = plt.subplots(figsize=(12, 4))

    ax.bar(df["graph_id"], df["kendalls_w"])
    ax.set_xticklabels(df["graph_id"], rotation=90)
    ax.set_ylabel("Kendall's W")
    ax.set_xlabel("Graph ID")
    ax.set_title("Ranking Stability per Graph")
    ax.set_ylim(0, 1)

    fig.tight_layout()
    return fig

def plot_kendalls_w_aggregated(df):
    grouped = df.groupby("n_nodes")["kendalls_w"].agg(["mean", "std"])

    fig, ax = plt.subplots()

    x_values = grouped.index.to_list()

    ax.errorbar(
        x_values,
        grouped["mean"],
        yerr=grouped["std"],
        fmt='o-',
        capsize=5
    )

    ax.set_xlabel("Graph Size")
    ax.set_ylabel("Mean Kendall's W")
    ax.set_title("Rank Stability per Graph")
    ax.set_ylim(0, 1)

    ax.set_xticks(x_values)

    fig.tight_layout()

    return fig

def plot_kendalls_w_boxplot(df):
    grouped = [group["kendalls_w"].values
               for _, group in df.groupby("n_nodes")]
    labels = sorted(df["n_nodes"].unique())

    fig, ax = plt.subplots()

    ax.boxplot(grouped)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)

    ax.set_xlabel("Graph Size")
    ax.set_ylabel("Kendall's W")
    ax.set_title("Rank Stability per Graph")
    ax.set_ylim(0, 1)

    fig.tight_layout()

    return fig

