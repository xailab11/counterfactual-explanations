"""
counterfactual/cache_utils.py

This module provides caching utilities.

It supports caching of:
- verbalized answers obtained from subgraphs,
- vector embeddings of generated answers,
- semantic dissimilarities between baseline and counterfactual answers.

The cache is keyed by graph structure, applied node interventions,
and generation mode, which allows efficient reuse of computations.
"""

from sklearn_embeddings import SentenceTransformerEmbedding
from graphrag.verbalization import generate_verbalization
from typing import Union, Tuple, Iterable, List
from graphrag.utils.graph_model import Graph
import numpy as np
import pickle
from pathlib import Path
from graphrag.serialization import serialize_subgraph
from sklearn_embeddings import SentenceTransformerEmbedding
from contextlib import contextmanager

# ---------------------------------------------------------
# CACHE PATH
# ---------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = SRC_DIR / "experiments"
CACHES_BASE_DIR = EXPERIMENTS_DIR / "caches"
# ---------------------------------------------------------
# DATATYPE ALIASES
# ---------------------------------------------------------
NodeIDs = Tuple[str, ...]

# Operations applied during graph interventions
# e.g. ("remove"), ("remove", "replace")
Ops = Tuple[str, ...]
OpsInput = Union[str, Tuple[str], Iterable[str], List[str], None]

def normalize_ops(ops: OpsInput) -> Ops:
    """
    Normalize operation input to a canonical tuple representation.
    This ensures consistent cache keys independent of how operations are provided by the caller.
    """
    if ops is None:
        return ("remove",)
    if isinstance(ops, str):
        return (ops,)
    return tuple(ops)

# ---------------------------------------------------------
# Cache key definitions
# ---------------------------------------------------------
# Key for an unmodified (original) subgraph:
# ("orig", ("node1", "node2", ...))
OrigGraphKey = Tuple[str, NodeIDs]

# Key for a modified subgraph:# ("mod", graph_nodes, mode, operations, target_nodes)
# ("orig", ("1","2"))
ModGraphKey = Tuple[str, str, Ops, NodeIDs]

GraphKey = Union[OrigGraphKey, ModGraphKey]

# All caches are keyed by a (type, GraphKey) pair,
# e.g. ("verb", GraphKey) or ("emb", GraphKey
CacheKey = Tuple[str, GraphKey]

# ---------------------------------------------------------
# EMBEDDING MODEL
# ---------------------------------------------------------
embedding_model = SentenceTransformerEmbedding(model="all-MiniLM-L6-v2")

# ---------------------------------------------------------
# DEFINE CACHES
# ---------------------------------------------------------
VERBALIZATION_CACHE = {}
EMBEDDING_CACHE = {}
DISSIMILARITY_CACHE = {}

# ---------------------------------------------------------
# SAVE CACHES
# ---------------------------------------------------------
CACHE_FILE = Path("cache_store.pkl")

def save_caches(filepath: Path | str = CACHE_FILE) -> None:
    """
    Saves all caches.
    """
    data = {
        "VERBALIZATION_CACHE": VERBALIZATION_CACHE,
        "EMBEDDING_CACHE": EMBEDDING_CACHE,
        "DISSIMILARITY_CACHE": DISSIMILARITY_CACHE,
    }

    with open(filepath, "wb") as f:
        pickle.dump(data, f)

    print(f"Caches gespeichert nach: {filepath}")


def load_caches(filepath: Path | str = CACHE_FILE) -> None:
    """
    Load all caches.
    """
    global VERBALIZATION_CACHE, EMBEDDING_CACHE, DISSIMILARITY_CACHE

    filepath = Path(filepath)

    if not filepath.exists():
        print("Keine Cache-Datei gefunden — starte mit leeren Caches.")
        return

    with open(filepath, "rb") as f:
        data = pickle.load(f)

    VERBALIZATION_CACHE.clear()
    VERBALIZATION_CACHE.update(data.get("VERBALIZATION_CACHE", {}))

    EMBEDDING_CACHE.clear()
    EMBEDDING_CACHE.update(data.get("EMBEDDING_CACHE", {}))

    DISSIMILARITY_CACHE.clear()
    DISSIMILARITY_CACHE.update(data.get("DISSIMILARITY_CACHE", {}))

    print(f"Caches geladen aus: {filepath}")


# ---------------------------------------------------------
# HELPER FUNCTIONS TO GENERATE KEYS
# ---------------------------------------------------------
def key_for_graph(graph: Graph) -> OrigGraphKey:
    """
    Generate a unique, order-invariant cache key for an original subgraph.
    The key is based solely on the sorted node identifiers,
    which uniquely characterize the induced subgraph.
    """
    node_ids = tuple(sorted([n.id for n in graph.nodes]))
    return ("orig", node_ids)

def key_for_modification(
    graph: Graph,
    ops: Ops,
    node_ids: NodeIDs,
    mode: str | None=None
) -> ModGraphKey:
    """
    Generate a cache key for a modified subgraph produced by a node intervention.
    Parameters
    graph : Graph,
            The original subgraph G'.
    ops : Ops
           Applied intervention operations (e.g., remove).
    node_ids : NodeIDs
           Target nodes of the intervention.
    mode : str
            Intervention mode (single, parallel, greedy).
    Single-node interventions are always treated with mode='single'.
    """
    if len(node_ids) == 1 or mode is None:
        mode = "single"

    graph_nodes = tuple(sorted([n.id for n in graph.nodes])) # for unambiguity

    return ("mod", graph_nodes, mode,  ops, node_ids)

# ---------------------------------------------------------
# VERBALIZATION CACHE
# ---------------------------------------------------------
def cached_verbalization(graph: Graph,
                         question: str,
                         model: str,
                         key: CacheKey) -> str:
    """
    Return the cached verbalization result for a given subgraph and query.
    Verbalizations are cached to avoid repeated calls to the LLM during node relevance computation.
    """
    serialization = serialize_subgraph(graph)
    if key not in VERBALIZATION_CACHE:
        VERBALIZATION_CACHE[key] = generate_verbalization(serialization, question, model)
    return VERBALIZATION_CACHE[key]

# ---------------------------------------------------------
# EMBEDDING CACHE
# ---------------------------------------------------------
def cached_embedding(text: str, key: CacheKey) -> np.ndarray:
    """
    Return the cached embedding vector for a generated answer.
    """
    if key in EMBEDDING_CACHE:
        return EMBEDDING_CACHE[key]

    emb = embedding_model.transform(text)
    EMBEDDING_CACHE[key] = emb
    return emb

# ---------------------------------------------------------
# PRINT CACHE
# ---------------------------------------------------------
def print_caches():
    """
    Print all caches:
    - VERBALIZATION_CACHE: key --> str
    - EMBEDDING_CACHE: key -- embedding shape/dtype
    - DISSIMILARITY_CACHE: key --> dissimilarity
    """
    caches = {
        "VERBALIZATION_CACHE": VERBALIZATION_CACHE,
        "EMBEDDING_CACHE": EMBEDDING_CACHE,
        "DISSIMILARITY_CACHE": DISSIMILARITY_CACHE
    }

    for name, cache in caches.items():
        print(f"\n{name} (size: {len(cache)}):")
        if not cache:
            print("  <empty>")
            continue

        for i, (k, v) in enumerate(cache.items()):
            if i >= 10:
                print("  ...")
                break

            print(f"  Key: {k}")

            if name == "VERBALIZATION_CACHE":
                preview = v.replace("\n", "\\n")
                print(f"    Verbalization: '{preview[:80]}{'...' if len(preview) > 80 else ''}'")
            elif name == "EMBEDDING_CACHE":
                if isinstance(v, np.ndarray):
                    print(f"    Embedding: shape={v.shape}, dtype={v.dtype}")
                else:
                    print(f"    Embedding: {type(v)}")
            elif name == "DISSIMILARITY_CACHE":
                dissimilarity = v.get("dissimilarity", None) if isinstance(v, dict) else v
                print(f"    Dissimilarity: {dissimilarity:.6f}" if dissimilarity is not None else f"    {v}")


@contextmanager
def cache_manager(data: str, model: str, filename: str = "cache_store.pkl"):
    """
    Context manager for experiment-specific caching.
    Ensures that cached verbalizations and embeddings are reused
    across runs and automatically persisted to disk.
    """
    cache_dir = CACHES_BASE_DIR / data / model
    print(f"[Cache] Using cache dir: {cache_dir.resolve()}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / filename
    load_caches(cache_file)
    try:
        yield
    finally:
        save_caches(cache_file)


