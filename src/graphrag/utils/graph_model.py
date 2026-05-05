"""
Data structures for representing property graphs and subgraphs.

These classes provide a minimal abstraction layer between retrieval,
serialization, and counterfactual intervention code.
"""

from pydantic import BaseModel
from typing import List, Dict, Any

class KGNode(BaseModel):
    id: str
    labels: List[str]
    properties: Dict[str, Any]

class KGEdge(BaseModel):
    start: str
    label: str
    end: str

class Graph(BaseModel):
    nodes: List[KGNode]
    edges: List[KGEdge]
    #labels: List[str]
    @property
    def labels(self) -> List[str]:
        """Automatically return all unique labels from nodes."""
        all_labels = set()
        for n in self.nodes:
            all_labels.update(n.labels)
        return sorted(all_labels)

    def node_ids(self) -> List[str]:
        return [n.id for n in self.nodes]

    def get_node(self, node_id: str) -> KGNode | None:
        for n in self.nodes:
            if n.id == node_id:
                return n
        return None

class PathSample(BaseModel):
    node_ids: List[str]
    node_labels: List[List[str]]
    node_properties: List[Dict]
    rel_types: List[str]
    anchor_label: str
    anchor_properties: Dict



