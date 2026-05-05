"""
graphrag/serialization.py

This module implements serialization of retrieved graph structures
into textual representations suitable for language model input.

It provides:
- conversion from database retrieval results to subgraphs,
- serialization of subgraphs into human-readable strings.
"""
from graphrag.utils.graph_model import Graph,KGEdge,KGNode,PathSample
from typing import List, Dict, Any
from gqlalchemy.models import Node, Relationship
from counterfactual.GraphUtils import GraphUtils

# ----------------------------------------------------------------
# Helper: secure getters for relationships
# ----------------------------------------------------------------
def get_relationship_info(rel: Relationship) -> tuple[str, str, str]:
    """
    Returns the start ID, end ID, and type of relationship.
    First attempts public attributes, otherwise falls back to _-members.
    """
    start = getattr(rel, "start_node_id", None) or getattr(rel, "_start_node_id")
    end = getattr(rel, "end_node_id", None) or getattr(rel, "_end_node_id")
    typ = getattr(rel, "type", None) or getattr(rel, "_type")
    return str(start), str(end), str(typ)

# ----------------------------------------------------------------
# CONVERTS RETRIEVAL TO SUBGRAPH
# ----------------------------------------------------------------
def retrieval_to_subgraph(retrieval: List[Dict[str, Any]]) -> Graph:
    """
    Convert raw database retrieval results into a property-graph subgraph.

    The function processes nodes and relationships returned by a graph
    database query and constructs a unified subgraph representation.
    Duplicate nodes are merged based on node identifiers.

    This function serves as a preprocessing step before serialization
    and verbalization.
    """
    node_index: Dict[str, KGNode] = {}     # id → KGNode
    edges: List[KGEdge] = []
    labels_set: set = set()

    for row in retrieval:
        for value in row.values():

            #  If value is a Relationship  --> generate Edge
            if isinstance(value, Relationship):
                start, end, typ = get_relationship_info(value)
                edges.append(KGEdge(
                    #start=str(value._start_node_id),
                    start = start,
                    end = end,
                    #end=str(value._end_node_id),
                    label= typ
                    #label=value._type
                ))
                continue

            #  If Value is a Node --> generate KGNode
            node_obj = GraphUtils.convert_value_to_node(value)

            # if convert_value_to_node returns nothing --> skip
            if not isinstance(node_obj, KGNode):
                continue

            # Collect labels
            labels_set.update(node_obj.labels)

            # Deduplicate to Node-ID
            if node_obj.id not in node_index:
                node_index[node_obj.id] = node_obj

            else:
                # If the same node occurs multiple times in the retrieval results,
                # merge labels and properties to construct a consistent node representation
                old = node_index[node_obj.id]
                merged_labels = list(set(old.labels + node_obj.labels))
                merged_props = {**old.properties, **node_obj.properties}

                node_index[node_obj.id] = KGNode(
                    id=old.id,
                    labels=merged_labels,
                    properties=merged_props
                )

    return Graph(
        nodes=list(node_index.values()),
        edges=edges,
        labels=list(labels_set)
    )

def path_to_subgraph(path: PathSample) -> Graph:
    nodes = []
    for i, nid in enumerate(path.node_ids):
        node = KGNode(
            id=str(nid),
            labels=path.node_labels[i],
            properties=path.node_properties[i]
        )
        nodes.append(node)

    edges = [
        KGEdge(
            start=str(path.node_ids[i]),
            end=str(path.node_ids[i + 1]),
            label=path.rel_types[i]
        )
        for i in range(len(path.rel_types))
    ]

    return Graph(nodes=nodes, edges=edges)


# ----------------------------------------------------------------
# SERIALIZE SUBGRAPH
# ----------------------------------------------------------------
def serialize_subgraph(subgraph) -> dict:
    """
    Serialize a subgraph into a human-readable textual representation.

    Each edge is rendered as a string that encodes:
    - source node labels and properties,
    - relation label,
    - target node labels and properties.

    """
    nodes = subgraph.nodes if hasattr(subgraph, "nodes") else []
    edges = subgraph.edges if hasattr(subgraph, "edges") else []

    # Node dict: id --> Node
    node_index = {node.id: node for node in nodes}

    serialized_edges = []

    # Serialize each edge independently to allow partial graph modifications
    for edge in edges:
        start_node = node_index.get(edge.start)
        end_node = node_index.get(edge.end)

        if start_node is None or end_node is None:
            continue

        # Labels, join list to string
        start_labels = ", ".join(start_node.labels)
        end_labels = ", ".join(end_node.labels)

        # Properties, Dictionary to readable string
        start_props = ", ".join(f"{k}: {v}" for k, v in start_node.properties.items())
        end_props = ", ".join(f"{k}: {v}" for k, v in end_node.properties.items())

        edge_label = edge.label

        # Final String
        readable = f"{start_labels} ({start_props}) {edge_label} {end_labels} ({end_props})"

        serialized_edges.append(readable)

    return {"answer":serialized_edges}

# ----------------------------------------------------------------
# PRINT SUBGRAPH OVERVIEW
# ----------------------------------------------------------------
def print_subgraph(subgraph: Graph):
    print("\n--- Nodes ---")
    for node in subgraph.nodes:
        if hasattr(node, "id"):
            node_id = node.id
            labels = node.labels
            props = node.properties
        else:
            node_id = node.get("id", str(node))
            labels = node.get("labels", [])
            props = node.get("properties", {})
        print(f"ID: {node_id}")
        print(f"  Labels: {', '.join(labels)}")
        print(f"  Properties: {props}\n")

    print("--- Edges ---")
    if subgraph.edges:
        for edge in subgraph.edges:
            print(f"{edge.start} -[{edge.label}]-> {edge.end}")
    else:
        print("(none)")

    print("\n--- Labels ---")
    print(", ".join(subgraph.labels))
