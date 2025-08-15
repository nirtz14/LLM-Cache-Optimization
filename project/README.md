# Project Plan: Enhanced Semantic Cache for LLMs

This `project` directory contains the skeleton implementation of an enhanced
semantic caching layer for large‑language models (LLMs).  The design follows
the plan agreed upon earlier in this conversation and aims to address the
limitations of naïve LLM caches by adding:

* **Similarity thresholding** with a tunable parameter `tau` (τ) to decide
  whether a cached response is close enough to a new query to be reused.
* **Context chain awareness**, ensuring that responses are only reused when
  the query belongs to the same conversation context as the cached entry.
* **Configurable architecture** exposing cache size, embedding model and
  threshold parameters via a `Config` dataclass.
* **Clean API surface** providing `put` and `get` methods for inserting and
  retrieving cached responses.

At this stage the code here is a starting point.  It defines the core
building blocks (`config.py`, `policy.py`, `context.py`, `adapter.py`) and a
basic test to verify that the modules load correctly.  Future work will
extend these placeholders with real similarity computations (using cosine
distance over embeddings), context tracking and persistence layers, as well
as benchmarks and more comprehensive tests.

## Directory Structure

```
project/
  ├─├─ README.md           # This file
  ├─├─ __init__.py         # Marks the project as a Python package
  ├─├─ config.py           # Configuration dataclass
  ├─├─ policy.py           # Threshold policy skeleton
  ├─├─ context.py          # Context chain skeleton
  ├─├─ adapter.py          # Semantic cache adapter skeleton
  └── tests/
      ├─├─ __init__.py
      └── test_skeleton.py  # Basic sanity tests
```

To develop further:

1. Flesh out the classes in `policy.py` and `context.py` with real logic for
   similarity measurement and context chain management.
2. Implement the `SemanticCache` adapter in `adapter.py` to integrate
   threshold policies, context checking and persistence.
3. Add benchmarks and proper unit/integration tests under `project/tests/`.
4. Connect to GPTCache by wrapping its API rather than modifying it.
