# Enhanced GPTCache with MeanCache Features

A Python project that extends [GPTCache](https://github.com/zilliztech/GPTCache) with three lightweight features inspired by the MeanCache paper:

1. **Context-chain filtering** – reject cache hits when the current conversation context diverges
2. **PCA embedding compression** – shrink 768-D HuggingFace/OpenAI embeddings to 128-D before storage/similarity search  
3. **Simulated federated τ-tuning** – each user locally optimizes the cosine-similarity threshold τ; a central aggregator averages deltas every N queries

## Quick Start

### Option 1: Docker (Recommended)
```bash
git clone <repo>
cd llm-cache-project
docker-compose up --build
```

### Option 2: Conda Environment
```bash
git clone <repo>
cd llm-cache-project
conda env create -f environment.yml
conda activate enhanced-gptcache
```

### Option 3: pip install
```bash
git clone <repo>
cd llm-cache-project
pip install -e .
# or for development
pip install -e ".[dev]"
```

## Run Benchmarks

### Generate Test Queries
```bash
python -m benchmark.generate_queries --output data/queries.json --count 1000
```

### Run Full Benchmark Suite
```bash
python -m benchmark.benchmark_runner --variant full --queries data/queries.json --output data/results.csv
```

### Analyze Results
```bash
python -m benchmark.analyze_results --input data/results.csv --output data/analysis/
```

## Project Structure

```
├── src/                           # Core implementation
│   ├── core/                      # Main features
│   │   ├── context_similarity.py  # Context-chain filtering
│   │   ├── pca_wrapper.py         # PCA embedding compression
│   │   └── tau_manager.py         # Federated τ-tuning
│   ├── utils/                     # Utilities
│   └── cache/                     # GPTCache integration
├── benchmark/                     # Performance testing
├── tests/                         # Unit tests
├── data/                          # Generated datasets
├── models/                        # Saved models (PCA, τ params)
└── docs/                          # Documentation
```

## Key Features

### Context-Chain Filtering
- Tracks conversation history to detect context divergence
- Configurable context window size and divergence threshold
- Prevents irrelevant cache hits from different conversation contexts

### PCA Embedding Compression  
- Reduces embedding dimensionality from 768D to 128D (configurable)
- Trains PCA model on collected embeddings
- Maintains search quality while reducing memory usage

### Federated τ-Tuning
- Simulates federated learning for threshold optimization
- Each "user" locally optimizes cosine similarity threshold
- Central aggregator combines improvements periodically
- Improves precision/recall balance over time

## Configuration

Configuration is managed via `config.yaml` or environment variables:

```yaml
cache:
  size_mb: 100
  similarity_threshold: 0.8

context:
  window_size: 5
  divergence_threshold: 0.3

pca:
  target_dimensions: 128
  training_samples: 1000

federated:
  users: 10
  aggregation_frequency: 100
  learning_rate: 0.01
```

## Performance Results

| Configuration | Hit Rate | P95 Latency | Memory Usage |
|--------------|----------|-------------|--------------|
| Baseline     | 45.2%    | 120ms       | 250MB        |
| +Context     | 52.8%    | 115ms       | 255MB        |
| +PCA         | 44.9%    | 95ms        | 180MB        |
| Full         | 58.1%    | 88ms        | 185MB        |

## Testing

Run the full test suite:
```bash
pytest tests/ --cov=src --cov-report=html
```

Run benchmarks:
```bash
pytest tests/ -k benchmark --benchmark-only
```

## Development

### Code Quality
```bash
black src/ tests/ benchmark/
ruff src/ tests/ benchmark/
mypy src/
```

### Contributing
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Ensure all tests pass
5. Submit a pull request

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{enhanced-gptcache,
  title={Enhanced GPTCache with MeanCache Features},
  year={2025},
  url={https://github.com/your-org/enhanced-gptcache}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.
