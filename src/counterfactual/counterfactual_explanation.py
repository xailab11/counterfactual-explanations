"""
counterfactual/counterfactual_explanation.py

This module implements counterfactual explanation methods.

It provides:
- A greedy heuristic for computing minimal counterfactual node sets,
- An exact (optimal) counterfactual search,
- Utility functions for inspecting and visualizing counterfactual graphs.

"""
from counterfactual.relevance_distribution import compute_dissimilarity
from counterfactual.cache_utils import cached_verbalization, key_for_graph, key_for_modification, OpsInput, normalize_ops
from counterfactual.GraphUtils import GraphUtils
from graphrag.serialization import print_subgraph
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random
import logging
import itertools

from graphrag.utils.graph_model import Graph

logging.basicConfig(level=logging.INFO)

EPS = 1e-12

# ----------------------------------------------------------------
# GREEDY COUNTERFACTUAL
# ----------------------------------------------------------------
def greedy_counterfactual(
    g_original: Graph,
    tau: float,
    relevance_distribution: dict[str, dict[str, float]],
    question: str,
    model: str,
    operations: OpsInput= None,
    debug=False
):
    """
    Compute a counterfactual explanation using a greedy node selection strategy.
    Starting from the original subgraph, nodes are iteratively selected according to their relevance scores
    and modified using the given intervention operations until the semantic dissimilarity exceeds the threshold.

    Returns
    -------
    dict
        Dictionary containing:
        - g_before: original subgraph
        - g_modified_after: counterfactually modified subgraph
        - answer_before: original verbalized answer
        - answer_modified: counterfactual answer
        - selected_nodes: nodes selected for intervention
        - selected_operations: applied operations
        - max_dissimilarity: achieved semantic change
    """
    operations = normalize_ops(operations)

    # initial status
    g_before = g_original.model_copy()
    g_current = g_original.model_copy()

    # for empty selected nodes
    key_current = key_for_graph(g_current)
    vkey_orig = ("verb", key_current)
    answer_before = cached_verbalization(g_before, question, model, vkey_orig)
    answer_modified = answer_before

    current_dissimilarity = 0.0
    selected_nops = []

    # Sort nodes by descending relevance
    node_relevance = {nid: max(v.values()) for nid, v in relevance_distribution.items()}
    sorted_nodes = sorted(
        g_original.nodes,
        key=lambda n: node_relevance.get(n.id, 0.0),
        reverse=True
    )

    if debug:
        print("=== Greedy Debug Start ===")
        print("Node order:", [(n.id, node_relevance.get(n.id, 0)) for n in sorted_nodes])


    for node in sorted_nodes:

        if current_dissimilarity >= tau:
            break

        best_score = current_dissimilarity
        best_graph = None
        best_answer = None
        best_op = None

        if debug:
            print(f"\nEvaluating node {node.id}")

        # Try all allowed operations for the current node
        for op in operations:

            # removed nodes + actual test node
            mod_nodes = tuple([nid for nid, _ in selected_nops] + [node.id])
            mod_ops = tuple([o for _, o in selected_nops] + [op])

            # Key for function compute_dissimilarity
            #mode = "single" if len(mod_nodes) == 1 else "greedy"
            key_mod = key_for_modification(g_original, mod_ops, mod_nodes, mode="greedy")

            g_test = GraphUtils.alpha(g_current.model_copy(), node.id, operation=op)

            # compute semantic dissimilarity between current and modified graph
            result = compute_dissimilarity(
                g_current,
                g_test,
                question,
                model,
                key_current,
                key_mod
            )

            score = result["dissimilarity"]

            if debug:
                print(f"   op={op}, score={score:.6f}")

            if score > best_score + EPS:
                best_score = score
                best_op = op
                best_graph = g_test
                best_answer = result["answer_modified"]

        # if an improvement exists --> apply the best
        if best_op is not None:
            selected_nops.append((node.id, best_op))
            g_current = best_graph
            key_current = key_for_modification(g_original, tuple(op for _, op in selected_nops),
                                               tuple(nid for nid, _ in selected_nops), mode = "greedy")
            answer_modified = best_answer
            current_dissimilarity = best_score

            if debug:
                print(f" → Applied {best_op} on {node.id}, new dis={best_score:.6f}")

    if debug:
        print("\n=== Greedy Debug End ===")

    selected_nodes = [nid for nid, op in selected_nops]
    selected_operations = [op for nid, op in selected_nops]

    return {
        "g_before": g_before,
        "answer_before": answer_before,
        "max_dissimilarity": current_dissimilarity,
        "selected_operations_nodes": selected_nops,
        "selected_nodes": selected_nodes,
        "selected_operations": selected_operations,
        "g_modified_after": g_current,
        "answer_modified": answer_modified
    }

# ----------------------------------------------------------------
# OPTIMAL COUNTERFACTUALAl
# ----------------------------------------------------------------
def optimal_counterfactual(
    g_original: Graph,
    tau: float,
    question: str,
    model: str,
    operations: OpsInput = None,
    debug=False
):
    """
    Compute an optimal counterfactual explanation by exhaustive search.
    This method enumerates all node subsets and all corresponding operation assignments to find a
    counterfactual explanation of minimal cardinality that reaches the threshold.
    """
    operations = normalize_ops(operations)

    nodes = [n.id for n in g_original.nodes]
    key_orig = key_for_graph(g_original)

    # baseline verbalization
    vkey_orig = ("verb", key_orig)
    answer_before = cached_verbalization(g_original, question, model, vkey_orig)

    max_score_reached = 0.0

    if debug:
        print("=== Optimal Search Start ===")

    # search by increasing subset size
    for k in range(1, len(nodes) + 1):
        if debug:
            print(f"\nTesting subsets of size {k}")

        for subset in itertools.combinations(nodes, k):
            # for each subset, consider ALL operation assignments
            for ops_tuple in itertools.product(operations, repeat=k):

                # Apply all modifications PARALLEL on the original graph
                g_mod = g_original.model_copy()
                for nid, op in zip(subset, ops_tuple):
                    g_mod = GraphUtils.alpha(g_mod, nid, operation=op)

                # construct cache key (parallel)
                #if len(subset) == 1:
                #    key_mod = ("mod", "single", ops_tuple[0], subset)
                # else:
                #    key_mod = ("mod", "parallel", ops_tuple, subset)
                key_mod = key_for_modification(g_original, ops_tuple, subset, mode="parallel")

                # compute dissimilarity
                res = compute_dissimilarity(
                    g_original,
                    g_mod,
                    question,
                    model,
                    key_orig,
                    key_mod
                )

                score = float(res["dissimilarity"])
                max_score_reached = max(max_score_reached, score)

                if debug:
                    print(f"subset={subset}, ops={ops_tuple}, dis={score:.4f}")

                # found minimal optimal solution
                if score >= tau - EPS:
                    if debug:
                        print("\n=== Optimal Solution Found ===")

                    selected_nodes = list(subset)
                    selected_operations = list(ops_tuple)

                    return {
                        "g_before": g_original,
                        "answer_before": answer_before,
                        "max_dissimilarity": score,  # unify key name
                        "max_score_reached": max_score_reached,
                        "selected_operations_nodes": list(zip(subset, ops_tuple)),
                        "selected_operations": selected_operations,
                        "selected_nodes": selected_nodes,  # optional
                        "g_modified_after": g_mod,
                        "answer_modified": res["answer_modified"]
                    }

    if debug:
        print("No solution reached threshold.")
    return None

# ----------------------------------------------------------------
# PRINT COUNTERFACTUAL RESULTS
# ----------------------------------------------------------------
def print_counterfactual_results(results: dict):
    """
    Print the results of greedy_counterfactual or optimal_counterfactual
    in a clean, readable format.
    """
    g_before = results.get("g_before")
    g_after = results.get("g_modified_after")
    answer_before = results.get("answer_before")
    answer_after = results.get("answer_modified")
    dis = results.get("max_dissimilarity", results.get("dissimilarity", None))
    selected_nodes_ops = results.get("selected_operations_nodes", [])
    selected_nodes = results.get("selected_nodes", [])
    selected_operations = results.get("selected_operations", [])

    print("\n=== ORIGINAL SUBGRAPH ===")
    print_subgraph(g_before)

    print("\n=== MODIFIED SUBGRAPH ===")
    print_subgraph(g_after)

    print("\n=== ANSWER BEFORE ===")
    print(answer_before)

    print("\n=== ANSWER AFTER ===")
    print(answer_after)

    if dis is not None:
        print(f"\n=== FINAL DISSIMILARITY: {dis:.4f} ===")

    print("\n=== SELECTED NODES & OPERATIONS ===")
    if not selected_nodes_ops:
        print("(none)")
    else:
        for nid, op in selected_nodes_ops:
            print(f"- Node {nid} → operation '{op}'")

    print("\n=== SELECTED NODES ===")
    if selected_nodes:
        print(selected_nodes)
    else:
        print("(none)")

    print("\n=== SELECTED OPERATIONS ===")
    if selected_operations:
        print(selected_operations)
    else:
        print("(none)")

# ----------------------------------------------------------------
# PLOT FUNCTION
# ----------------------------------------------------------------
def plot_counterfactual_graphs(results: dict):
    """
    Plot the original and modified subgraph, highlighting nodes selected
    by either greedy or optimal counterfactual search.
    selected_operation_nodes format: [(node_id, operation), ...]
    """
    g_before = results["g_before"]
    g_after = results["g_modified_after"]
    selected_ids = {nid for nid, _ in results.get("selected_operations_nodes", [])}

    # Convert Subgraphs to NetworkX
    G_before = nx_graph_from_subgraph(g_before)
    G_after = nx_graph_from_subgraph(g_after)

    pos = nx.spring_layout(G_before, seed=42)

    node_labels_before = {n: make_node_label(n, G_before) for n in G_before.nodes()}
    node_labels_after = {n: make_node_label(n, G_after) for n in G_after.nodes()}

    plt.figure(figsize=(16, 7))

    # Before
    plt.subplot(1, 2, 1)
    colors_before = ["red" if n in selected_ids else "skyblue" for n in G_before.nodes()]
    nx.draw(G_before, pos, with_labels=False, node_color=colors_before, node_size=1800, arrowsize=20)
    nx.draw_networkx_labels(G_before, pos, labels=node_labels_before, font_size=9)
    edge_labels_before = {(u, v): d["label"] for u, v, d in G_before.edges(data=True)}
    nx.draw_networkx_edge_labels(G_before, pos, edge_labels=edge_labels_before, font_size=8)
    plt.title("Original Graph")

    # After
    plt.subplot(1, 2, 2)
    colors_after = ["red" if n in selected_ids else "skyblue" for n in G_after.nodes()]
    nx.draw(G_after, pos, with_labels=False, node_color=colors_after, node_size=1800, arrowsize=20)
    nx.draw_networkx_labels(G_after, pos, labels=node_labels_after, font_size=9)
    edge_labels_after = {(u, v): d["label"] for u, v, d in G_after.edges(data=True)}
    nx.draw_networkx_edge_labels(G_after, pos, edge_labels=edge_labels_after, font_size=8)
    plt.title("Modified Graph")

    plt.tight_layout()
    plt.show()

def nx_graph_from_subgraph(subgraph):
    g = nx.DiGraph()
    for n in subgraph.nodes:
        g.add_node(n.id, labels=n.labels, properties=n.properties)
    for e in subgraph.edges:
        g.add_edge(e.start, e.end, label=e.label)
    return g

# Helper to create detailed node labels
def make_node_label(node_id, g):
    labels = ", ".join(g.nodes[node_id]["labels"])
    props = ", ".join(f"{k}: {v}" for k, v in g.nodes[node_id]["properties"].items())
    return f"{node_id}\n{labels}\n{props}"



