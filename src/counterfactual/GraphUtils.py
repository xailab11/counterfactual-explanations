"""
counterfactual/GraphUtils.py

Utility functions implementing graph interventions on subgraphs.

This module realizes the node intervention operators:
- node removal (occlusion),
- node replacement,
- node perturbation.
"""
from typing import List, Optional, Literal
from graphrag.utils.graph_model import KGNode, KGEdge, Graph
from graphrag.utils.memgraph_connector import MemgraphConnector
from gqlalchemy.models import Node
from copy import deepcopy
import random



class GraphUtils:

    @staticmethod
    def convert_value_to_node(value: any) -> KGNode:

        if isinstance(value, Node):
            node_id = getattr(value, "id", None) or getattr(value, "_id")
            labels = getattr(value, "labels", None) or getattr(value, "_labels", [])
            props = getattr(value, "properties", None) or getattr(value, "_properties", {})

            return KGNode(
                id=str(node_id),
                labels=list(labels),
                properties=dict(props)
            )
        elif isinstance(value, KGNode):
            return value
        elif isinstance(value, dict):
            return KGNode(
                id=str(value.get("id") or value.get("_id") or str(value)),
                labels=list(value.get("labels") or value.get("_labels") or []),
                properties=dict(value.get("properties") or value)
            )
        else:
            return KGNode(
                id=str(value),
                labels=["Unknown"],
                properties={"value": str(value)}
            )

    @staticmethod
    def get_node_by_id(nodes: List[KGNode], node_id: str) -> Optional[KGNode]:
        for n in nodes:
            if n.id == node_id:
                return n
        return None


    @staticmethod
    def remove_node(graph: Graph, node_id: str) -> Graph:
        g = deepcopy(graph)
        # remove node
        g.nodes = [n for n in g.nodes if n.id != node_id]
        # remove edge
        g.edges = [e for e in g.edges if e.start != node_id and e.end != node_id]
        return g


    @staticmethod
    def select_random_node(
            g: Graph,
            mode: Literal["local", "global"],
            exclude_id: str | None = None
    ) -> KGNode:

        if mode == "local":
            candidates = g.nodes
            candidates = [n for n in candidates if n.id != exclude_id]

        elif mode == "global":
            mg = MemgraphConnector()
            all_records = mg.run_query("MATCH (n) RETURN n")

            global_nodes = [
                GraphUtils.convert_value_to_node(r["n"])
                for r in all_records
            ]

            subgraph_ids = {n.id for n in g.nodes}

            candidates = [
                n for n in global_nodes
                if n.id not in subgraph_ids
            ]


        else:
            raise ValueError("mode must be local/global")


        if not candidates:
            raise ValueError("no possible candidates")

        chosen = random.choice(candidates)
        print(f"[DEBUG] selected: {chosen.id} mode={mode}")
        return chosen

    @staticmethod
    def replace_perturbate_node(
            graph: Graph,
            node_id: str,
            mode: Literal["local", "global"]
    ) -> Graph:

        g = deepcopy(graph)

        target = g.get_node(node_id)
        if target is None:
            print(f"[WARN] Node {node_id} not found in subgraph.")
            return g

        # choose random node
        fresh = GraphUtils.select_random_node(
            g,
            mode=mode,
            exclude_id=node_id
        )

        if fresh is None:
            print("[WARN] No replacement candidate found.")
            return g

        # if mode global --> add node
        if mode == "global" and g.get_node(fresh.id) is None:
            g.nodes.append(fresh)

        # edges rewiring
        new_edges = []
        for e in g.edges:
            start = fresh.id if e.start == node_id else e.start
            end = fresh.id if e.end == node_id else e.end
            new_edges.append(KGEdge(start=start, label=e.label, end=end))
        g.edges = new_edges

        # remove target
        g.nodes = [n for n in g.nodes if n.id != node_id]

        print(f"[DEBUG] replaced {node_id} -> {fresh.id} mode={mode}")

        return g

    @staticmethod
    def alpha(
        graph: Graph,
        node_id: str,
        operation: str | None
    ) -> Graph:
        """
        Apply a node-level intervention to a graph.

        Parameters
        ----------
        graph : Graph
            Input subgraph G'.
        node_id : str
            Target node of the intervention.
        operation : str
            Intervention type: 'remove', 'replace', or 'perturbate'.

        Returns
        ------
        Graph
            Modified subgraph α(G', {node_id}).
        """
        if operation in ("remove", None):
            return GraphUtils.remove_node(graph, node_id)
        elif operation == "replace":
            return GraphUtils.replace_perturbate_node(graph, node_id, mode = "global")
        elif operation == "perturbate":
            return GraphUtils.replace_perturbate_node(graph, node_id, mode = "local")
        else:
            raise ValueError(f"Unknown operation: {operation}")


