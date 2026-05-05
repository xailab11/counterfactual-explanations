# Counterfactuals for Subgraph Verbalizations

We study **node-level counterfactual explanations** for GraphRAG systems.
Given a retrieved subgraph and a natural-language query, the goal is to identify those
graph elements whose modification leads to a significant semantic change in the generated answer.

---

## Overview

The repository consists of four main components:

- **`graphrag/`**: Graph serialization and verbalization utilities, mapping retrieved subgraphs to text,
- **`counterfactual/`**: Core methods for computing node relevance and generating counterfactual explanations,
- **`evaluation/`**: Empirical analyses of method properties (e.g., consistency, additivity, minimality),
- **`experiments/`**: Dataset generation and experimental pipelines for reproducibility.


---

## Scope and Abstraction

This repository focuses on **counterfactual explanations for subgraph verbalizations** in GraphRAG systems.
Graph retrieval, ranking, and query planning are treated as upstream processes and are therefore not part of this implementation. 
The scope is restricted to the downstream pipeline:

> **retrieved subgraph → verbalization → node intervention on subgraph nodes
> → semantic comparison → counterfactual explanation**


---

## GraphRAG Utilities and Verbalization (`graphrag/`)

This directory contains supporting components for **subgraph-based
verbalization in GraphRAG systems**. It provides data structures,
serialization, and natural-language generation utilities that
are required by the counterfactual explanation pipeline.


#### `serialization.py`

Provides utilities for converting graph retrieval results into a unified subgraph representation
and serializing them into human-readable text for downstream verbalization.


#### `verbalization.py`

Implements natural-language verbalization of serialized subgraphs using 
large language models, with support for multiple backends (e.g., Ollama, Azure OpenAI) 
and cached model reuse.


#### `utils/graph_model.py`

Defines data structures for representing property graphs,
including nodes with labels and properties, edges with relation labels,
and induced subgraphs used throughout the project.

#### `utils/memgraph_connector.py`

Provides a minimal interface for interacting with a Memgraph database,
including starting a Docker container if required and executing Cypher
queries. This component is used only for data access and is orthogonal
to counterfactual reasoning.

#### `utils/ollama_launcher.py`

Convenience utility that ensures a local Ollama service is running
before verbalization is performed. This module is optional and not part
of the proposed method.

---

## Counterfactual Explanations (`counterfactual/`)

This directory contains the core implementation of **node-level
counterfactual explanations** for GraphRAG systems.

The modules operationalize the theoretical framework introduced in the
paper, focusing on graph interventions, semantic change, and the
construction of minimal counterfactual explanations.

#### `cache_utils.py`

Provides caching utilities for:

- subgraph verbalizations,
- answer embeddings,
- semantic dissimilarities.

Caching is essential for efficient experimentation, as verbalization and
embedding steps are computationally expensive. Cache keys are
constructed based on subgraph structure, applied interventions, and
generation mode.

#### `GraphUtils.py`

Implements node-level graph interventions, including:

- **node removal (occlusion)**: deletes nodes and all incident edges,
- **node replacement**: replaces nodes with randomly sampled nodes from outside the subgraph,
- **node perturbation**: replaces nodes with other nodes sampled from within the subgraph.

#### `relevance_distribution.py`

Computes node relevance scores via node-level interventions and the induced semantic change in the verbalized answer, 
producing a normalized relevance distribution. Semantic change is measured using cosine distance.

#### `counterfactual_explanation.py`

Implements algorithms for generating counterfactual explanations, including a greedy heuristic and an optimal (exhaustive) search.


---

## Evaluation (`evaluation/`)

This directory contains **empirical analyses** that evaluate assumptions and
properties of the proposed counterfactual explanation method.

The evaluation module is organized around three scripts,
each addressing a specific research question:

- `consistency.py`: analyzes the stability and robustness of node
  relevance distributions and rankings across repeated runs.

- `additivity.py`: evaluates the additivity assumption by comparing
  the semantic effect of joint node interventions to the sum of
  individual node effects.

- `minimality.py`: compares greedy counterfactual explanations to
  optimal solutions with respect to cardinality and achieved semantic change.

All evaluation procedures are optional and can be reproduced independently.


---

## Experiments (`experiments/`)

This directory contains all scripts required to **generate datasets, run
experiments, and reproduce the empirical results**.

The experimental code orchestrates the full pipeline and builds on the
core components implemented in `graphrag/` and `counterfactual/`. 

---

#### `query_generation.py`

Utility script for **dataset generation**.

**Usage**:

1. Load a dataset into Memgraph (e.g., via Memgraph Lab or Cypher).
2. Set the `dataset` variable to match the loaded graph.
3. Run the script to generate a dataset for downstream experiments.


---

#### `run_experiments.py`

Main entry point for running all experiments reported in the paper.
This script executes the full experimental pipeline for selected
datasets and models.
Experiments can be controlled via CLI arguments to select
specific datasets, models, or output directories.

---

#### `summary_results/experiments_utils.py`

Provides utilities for logging, directory management, and formatting experimental outputs.

---

## Installation

The code is written in Python and tested with Python 3.11.

Dependencies can be installed via:

```bash
pip install -r requirements.txt
```

The experiments rely on external services for graph storage and verbalization (e.g., Memgraph and Ollama). These tools
can be installed independently by following their official documentation.

### LLM Setup

By default, we use **local models via Ollama**.

1. Install Ollama: https://ollama.com
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

2. Pull a model:

```bash
ollama pull gemma4:latest
```
3. Start the ollama service: 
```bash
ollama serve 
```

### Memgraph Lab Setup

See https://memgraph.com/docs/memgraph-lab#quick-start.

Linux/macOS:
```bash
curl https://install.memgraph.com | sh
```
Windows: 
```bash
iwr https://windows.memgraph.com | iex
```

## Usage

The main entry point for running experiments is the script
`experiments/run_experiments.py`.


Experiments can be executed from the command line. For example, to run all evaluations for a specific dataset and model:
```bash
PYTHONPATH=src python -m experiments.run_experiments \
  --model llama3:70b \
  --dataset data_hetionet
```

Multiple datasets and models can be specified by repeating the flags:
```bash
PYTHONPATH=src python -m experiments.run_experiments \
  --model gemma4:latest \
  --model llama3:70b \
  --dataset data_hetionet \
  --dataset data_football
```

If no command-line arguments are provided, the script runs all default datasets
and models defined in `run_experiments.py`.

Datasets can be generated using `experiments/query_generation.py`.

--- 

## License

This project is released under the MIT License.
See the `LICENSE` file for details.

