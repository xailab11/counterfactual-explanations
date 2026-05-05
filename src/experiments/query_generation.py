"""
experiments/query_generation.py

Utility script for generating evaluation datasets from a Memgraph
knowledge graph.

The script samples random paths from a loaded knowledge graph,
constructs subgraphs, and generates corresponding natural-language
questions and answers using an LLM.

Usage:
1. Load the desired dataset via Memgraph Lab or Cypher.
2. Set the `dataset` variable to match the loaded graph.
3. Run this script to generate a JSON dataset for experiments.
"""

import random
import json
from pathlib import Path
from typing import List, Tuple
from gqlalchemy.models import Relationship
from graphrag.utils.graph_model import Graph, KGNode, KGEdge
from graphrag.utils.graph_model import  PathSample
from graphrag.serialization import path_to_subgraph, serialize_subgraph
from graphrag.verbalization import generate_question, generate_verbalization
from graphrag.utils.memgraph_connector import MemgraphConnector

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
QUERY_TYPE = "path"
MIN_HOPS = 1
MAX_HOPS = 5
MODEL = "llama3:70b"
dataset = "data_football"
NUM_GRAPHS = 100

# ---------------------------------------------------------
# INTERNAL HELPERS
# ---------------------------------------------------------
def _get_node_id(node) -> str:
    return str(getattr(node, "id", None) or getattr(node, "_id"))

def _get_node_labels(node) -> list[str]:
    return list(getattr(node, "labels", None) or getattr(node, "_labels", []))

def _get_node_props(node) -> dict:
    return dict(getattr(node, "properties", None) or getattr(node, "_properties", {}))

def _get_rel_type(rel: Relationship) -> str:
    typ = getattr(rel, "type", None) or getattr(rel, "_type", None)
    if not typ:
        raise ValueError("Relationship without type")
    return typ

def cypher_literal(v):
    if isinstance(v, str):
        return f"'{v}'"
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "null"
    return str(v)

# ---------------------------------------------------------
# Sample Path
# ---------------------------------------------------------
def sample_n_hop_path(mc: MemgraphConnector, min_hops=MIN_HOPS, max_hops=MAX_HOPS) -> PathSample:
    """
    Randomly sample a simple path from the knowledge graph.
    The path length is chosen uniformly between MIN_HOPS and MAX_HOPS.
    Cycles are avoided to ensure a simple, interpretable subgraph.
    """
    # Random anchor node
    rows = mc.run_query("MATCH (n) RETURN n ORDER BY rand() LIMIT 1")
    if not rows:
        raise RuntimeError("No nodes in graph")

    anchor = rows[0]["n"]
    anchor_label_list = _get_node_labels(anchor)
    if not anchor_label_list:
        raise RuntimeError("Anchor node has no labels")
    anchor_label = random.choice(anchor_label_list)
    anchor_props = _get_node_props(anchor)
    anchor_id = _get_node_id(anchor)

    #  Initialize Path
    node_ids = [anchor_id]
    node_labels = [anchor_label_list]
    node_properties = [anchor_props]
    rel_types = []

    visited = {anchor_id}
    current_id = anchor_id

    # Stepwise traversal
    hops = random.randint(min_hops, max_hops)
    for _ in range(hops):
        query = f"""
        MATCH (n)-[r]-(m)
        WHERE {(f"id(n) = {current_id}" if current_id.isdigit() else f"n.id = {cypher_literal(current_id)}")}
        WITH r, m, coalesce(m.id, id(m)) AS mid
        ORDER BY rand() LIMIT 1
        RETURN r, m, mid
        """
        rows = mc.run_query(query)
        if not rows:
            break  # Dead end

        r = rows[0]["r"]
        m = rows[0]["m"]
        mid = str(rows[0]["mid"])

        # avoid cycles
        if mid in visited:
            break

        rel_types.append(_get_rel_type(r))
        node_ids.append(mid)
        node_labels.append(_get_node_labels(m))
        node_properties.append(_get_node_props(m))

        visited.add(mid)
        current_id = mid

    if len(node_ids) < 2:
        raise RuntimeError("Could not sample enough hops for path")

    return PathSample(
        node_ids=node_ids,
        node_labels=node_labels,
        node_properties=node_properties,
        rel_types=rel_types,
        anchor_label=anchor_label,
        anchor_properties=anchor_props
    )

# ---------------------------------------------------------
# Build Cypher
# ---------------------------------------------------------
def build_cypher(path: PathSample) -> str:
    pattern = ""

    # start node
    start_labels = ":`" + "`,`".join(path.node_labels[0]) + "`" if path.node_labels[0] else ""
    start_props = "{" + ", ".join(f"`{k}`: {cypher_literal(v)}" for k, v in path.node_properties[0].items()) + "}" if path.node_properties[0] else ""
    pattern += f"(n0{start_labels}{start_props})"

    # nodes + relations
    for i in range(len(path.rel_types)):
        rel_label = f"`{path.rel_types[i]}`"
        next_labels = ":`" + "`,`".join(path.node_labels[i + 1]) + "`" if path.node_labels[i + 1] else ""
        next_props = "{" + ", ".join(f"`{k}`: {cypher_literal(v)}" for k, v in path.node_properties[i + 1].items()) + "}" if path.node_properties[i + 1] else ""
        pattern += f"-[r{i}:{rel_label}]-" + f"(n{i+1}{next_labels}{next_props})"

    return f"MATCH {pattern} RETURN {', '.join(f'n{i}, r{i}' for i in range(len(path.rel_types)))}, n{len(path.rel_types)} LIMIT 1"

# ---------------------------------------------------------
# Generate Example
# ---------------------------------------------------------
def generate_single_example(mc: MemgraphConnector, model) -> tuple[dict, dict] | None:
    """
    Generate a single (subgraph, question, answer) triple.

    The question and answer are generated by an LLM conditioned on
    the sampled subgraph. The resulting examples form the basis
    for all downstream experiments.
    """

    try:
        path = sample_n_hop_path(mc)
    except RuntimeError:
        return None

    # subgraph from pathsample
    subgraph = path_to_subgraph(path)
    cypher = build_cypher(path)

    if len(subgraph.edges) < 1:
        return None

    # Question Generation
    components = {
        "labels": subgraph.labels,
        "edges": [
            {
                "relation": e.label,
                "from": subgraph.get_node(e.start).labels,
                "to": subgraph.get_node(e.end).labels,
            }
            for e in subgraph.edges
        ],
        "node_properties": [n.properties for n in subgraph.nodes],
        "n_hops": len(subgraph.edges),
    }

    question = generate_question(QUERY_TYPE, components, model=model)

    serialization = serialize_subgraph(subgraph)
    answer = generate_verbalization(serialization, question, model=MODEL)

    dataset_entry = {
        "question": question,
        "answer": answer,
        "graph": {
            "nodes": [n.model_dump() for n in subgraph.nodes],
            "edges": [e.model_dump() for e in subgraph.edges]
        }
    }

    debug_entry = {
        "question": question,
        "cypher": cypher,
        "answer": answer,
        "retrieved_graph": {
            "nodes": [n.model_dump() for n in subgraph.nodes],
            "edges": [e.model_dump() for e in subgraph.edges]
        },
        "n_hops": len(subgraph.edges),
        "num_nodes": len(subgraph.nodes),
        "num_edges": len(subgraph.edges),
        "labels": subgraph.labels
    }

    return dataset_entry, debug_entry

# --------------------------------------------------------
# Dataset Generation
# ---------------------------------------------------------
def generate_test_data(n: int, model, dataset_file: Path, debug_file: Path) -> None:
    mc = MemgraphConnector()
    dataset = []
    debug_log = []
    attempts = 0

    while len(dataset) < n and attempts < n * 10:
        result = generate_single_example(mc, model)
        if result is not None:
            dataset_entry, debug_entry = result
            dataset.append(dataset_entry)
            debug_log.append(debug_entry)
            print(f"[{len(dataset)}/{n}] generated")
        attempts += 1

    dataset_file.parent.mkdir(parents=True, exist_ok=True)
    debug_file.parent.mkdir(parents=True, exist_ok=True)

    dataset_file.write_text(json.dumps(dataset, indent=2, ensure_ascii=False))
    debug_file.write_text(json.dumps(debug_log, indent=2, ensure_ascii=False))

# ---------------------------------------------------------
# LOAD DATASET --> Graphs + Questions
# ---------------------------------------------------------
def load_dataset(json_file: Path) -> Tuple[List[Graph], List[str]]:
    """
    Load dataset from JSON file and reconstruct Graph objects and questions.
    """
    if not json_file.exists():
        raise FileNotFoundError(f"{json_file} does not exist")

    data = json.loads(json_file.read_text(encoding="utf-8"))

    graphs: List[Graph] = []
    questions: List[str] = []

    for entry in data:
        gdata = entry["graph"]

        # Reconstruct nodes
        nodes = [
            KGNode(
                id=n["id"],
                labels=n.get("labels", []),
                properties=n.get("properties", {}),
            )
            for n in gdata.get("nodes", [])
        ]

        # Reconstruct edges
        edges = [
            KGEdge(
                start=e["start"],
                end=e["end"],
                label=e["label"],
            )
            for e in gdata.get("edges", [])
        ]

        graphs.append(Graph(nodes=nodes, edges=edges))
        questions.append(entry.get("question", ""))

    return graphs, questions

if __name__ == "__main__":

    output_file = Path("data") / f"{dataset}.json"
    debug_file = Path("data/debug") / f"debug_{dataset}.json"

    generate_test_data(NUM_GRAPHS, MODEL, output_file, debug_file)