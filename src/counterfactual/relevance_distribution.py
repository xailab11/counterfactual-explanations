"""
counterfactual/relevance_distribution.py

Compute node relevance distributions based on semantic dissimilarity
between original and perturbed graphs.

Relevance is estimated by measuring how strongly node-level interventions
affect the model's response and is optionally normalized to a probability distribution.
"""
from counterfactual.GraphUtils import GraphUtils
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict
from graphrag.utils.graph_model import Graph
import matplotlib.pyplot as plt
from graphrag.verbalization import generate_verbalization
from graphrag.serialization import serialize_subgraph
from counterfactual.cache_utils import  OpsInput, normalize_ops, key_for_graph, key_for_modification, GraphKey,  DISSIMILARITY_CACHE, cached_verbalization, cached_embedding, embedding_model

# ---------------------------------------------------------
# COMPUTE DISSIMILARITY
# ---------------------------------------------------------
def compute_dissimilarity(
    graph_orig: Graph,
    graph_mod: Graph,
    question: str,
    model: str,
    key_orig: GraphKey,
    key_mod: GraphKey,
    use_cache: bool | None = True
) -> dict[str, str | float]:
    """
    Computes cosine-dissimilarity between original and modified graph verbalizations.
    graph_orig: cached always if available
    graph_mod: only uses cache if use_cache=True
    """
    #print("use_cache:", use_cache)

    # original graph always cache verbalization
    vkey_orig = ("verb", key_orig)
    a_orig = cached_verbalization(graph_orig, question, model, vkey_orig)

    ekey_orig = ("emb", key_orig)
    e_orig = cached_embedding(a_orig, ekey_orig)

    # modified graph
    if use_cache:
        vkey_mod = ("verb", key_mod)
        ekey_mod = ("emb", key_mod)

        a_mod = cached_verbalization(graph_mod, question, model, vkey_mod)
        e_mod = cached_embedding(a_mod, ekey_mod)
    else:
        serialization = serialize_subgraph(graph_mod)
        a_mod = generate_verbalization(serialization, question, model)
        e_mod = embedding_model.transform(a_mod)

    # cosine-dissimilarity
    sim = cosine_similarity(e_orig.reshape(1, -1), e_mod.reshape(1, -1))[0, 0]
    #dis = (1 - sim) / 2
    # enforce theoretical bounds
    sim = max(min(sim, 1.0), -1.0)
    dis = (1.0 - sim) / 2.0
    dis = max(dis, 0.0)   # ensures delta >= 0

    result = {
        "answer": a_orig,
        "answer_modified": a_mod,
        "dissimilarity": float(dis),
    }

    # cache result
    if use_cache:
        #print("WRITE To CACHE ID:", id(DISSIMILARITY_CACHE))
        DISSIMILARITY_CACHE[key_mod] = result

    return result

# ---------------------------------------------------------
# NODE RELEVANCE DISTRIBUTION
# ---------------------------------------------------------
def approx_node_relevance_distribution(
    graph: Graph,
    question: str,
    model: str,
    operations: OpsInput = None,
    return_type: str | None = "normalized",
    use_cache: bool | None = True
) ->  dict[str, dict[str, float]] | tuple[dict[str, dict[str, float]], dict] | dict:
    """
    Approximate the node relevance distribution for a given subgraph.
    For each node and each allowed intervention operation, the semantic change induced by occlusion is measured.
    The resulting scores correspond to the node relevance values and their normalized distribution.
    """
    operations = normalize_ops(operations)

    raw_deltas = {}
    all_values = []

    for node in graph.nodes:
        node_id = node.id
        deltas = {}

        for op in operations:
            key = key_for_modification(graph,(op,), (node_id,))
            #if key in DISSIMILARITY_CACHE and use_cache is True:
            #    dis = DISSIMILARITY_CACHE[key]["dissimilarity"]
            #else:
            g_mod = GraphUtils.alpha(graph, node_id, operation=op)

            key_orig = key_for_graph(graph)
            result = compute_dissimilarity(graph, g_mod, question, model, key_orig, key, use_cache)
            dis = result["dissimilarity"]

            deltas[op] = dis
            all_values.append(dis)

        raw_deltas[node_id] = deltas

    # Normalize relevance scores into a probability distribution
    if return_type == "normalized":
        total = sum(all_values)

        if total > 0:
            pi = {
                nid: {op: v / total for op, v in d.items()}
                for nid, d in raw_deltas.items()
            }
        else:
            # fallback
            num_nodes = len(raw_deltas)
            num_ops = len(operations)
            uniform = 1.0 / (num_nodes * num_ops)

            pi = {
                nid: {op: uniform for op in d}
                for nid, d in raw_deltas.items()
            }

        return pi

    elif return_type == "raw":
        return raw_deltas

    elif return_type == "both":
        total = sum(all_values)

        if total > 0:
            pi = {
                nid: {op: v / total for op, v in d.items()}
                for nid, d in raw_deltas.items()
            }
        else:
            num_nodes = len(raw_deltas)
            num_ops = len(operations)
            uniform = 1.0 / (num_nodes * num_ops)

            pi = {
                nid: {op: uniform for op in d}
                for nid, d in raw_deltas.items()
            }

        return pi, raw_deltas

    else:
        raise ValueError(f"Unknown return_type: {return_type}")

# ----------------------------------------------------------------
# PLOT RELEVANCE DISTRIBUTION OF A SELECTED GRAPH
# ----------------------------------------------------------------
def plot_relevance_distribution(relevance_distribution: Dict[str, Dict[str, float]], op: str = "remove"):
    """
    Plots a relevance distribution as a bar chart.
    """
    # Extract Node IDs + probabilities
    node_ids = list(relevance_distribution.keys())
    probabilities = [float(relevance_distribution[nid].get(op, 0.0)) for nid in node_ids]

    # Plot
    plt.figure(figsize=(10, 5))
    plt.bar(node_ids, probabilities, color='skyblue')
    plt.xlabel('Node IDs')
    plt.ylabel('Probability')
    plt.title(f'Relevance Distribution')
    plt.xticks(rotation=45)
    plt.ylim(0, max(probabilities)*1.1 if probabilities else 1)
    plt.tight_layout()
    plt.show()
