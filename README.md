# Vectorless RAG System

A high-accuracy Retrieval-Augmented Generation (RAG) system that uses traditional information retrieval techniques instead of vector embeddings, providing better interpretability and performance for certain use cases.

## Overview

This vectorless RAG system leverages classical IR methods combined with modern NLP techniques to achieve strong accuracy without the computational overhead of embedding models:

- BM25 retrieval: probabilistic ranking
- Query expansion: adds synonyms and related terms
- Document reranking: heuristic or optional cross-encoder
- TF-IDF fallback: alternative signal for recall
- Hybrid scoring: combine BM25 and TF-IDF

## Key Features

- No vector database required: uses inverted index
- Interpretable results: clear signals and scoring
- Low latency: fast keyword-based search
- Easy integration: simple API
- Configurable: tune behavior via YAML
- Scalable: efficient indexing and retrieval

## Installation

```bash
pip install -r requirements.txt
# For local development/tests
pip install -e .
```

## Quick Start

```python
from vectorless_rag import VectorlessRAG

# Initialize the system
rag = VectorlessRAG()

# Index your documents
documents = [
    {"id": "doc1", "text": "Python is a high-level programming language."},
    {"id": "doc2", "text": "Machine learning uses statistical techniques."},
    {"id": "doc3", "text": "Natural language processing enables computers to understand text."}
]

rag.index_documents(documents)

# Retrieve relevant documents
query = "What is NLP?"
results = rag.retrieve(query, top_k=3)

for result in results:
    print(f"Score: {result['score']:.3f} - {result['text']}")
```

## Architecture

```
Query -> Preprocessing -> Query Expansion -> BM25 Retrieval -> Reranking -> Results
                                      \-> TF-IDF Fallback
```

## Performance

Our vectorless approach offers several advantages:

- Accuracy: competitive or better for keyword-heavy queries
- Speed: often 10–100x faster than dense vector search
- Memory: lower footprint than dense embeddings
- Transparency: easy to inspect why documents score

## Configuration

Edit `config.yaml` to customize:

```yaml
retrieval:
  method: bm25
  k1: 1.5
  b: 0.75

reranking:
  enabled: true
  model: cross-encoder/ms-marco-MiniLM-L-6-v2

query_expansion:
  enabled: true
  max_expansions: 3
```

## Advanced Usage

### Custom Preprocessing

```python
from vectorless_rag import VectorlessRAG, Preprocessor

# Create custom preprocessor
preprocessor = Preprocessor(
    lowercase=True,
    remove_stopwords=True,
    stemming=True
)

rag = VectorlessRAG(preprocessor=preprocessor)
```

### Hybrid Retrieval

```python
# Combine BM25 with TF-IDF
results = rag.retrieve(
    query="machine learning algorithms",
    methods=["bm25", "tfidf"],
    weights=[0.7, 0.3],
    top_k=10
)
```

## Use Cases

- FAQ systems
- Legal document search
- Code search
- Medical records
- Knowledge bases

## Comparison with Vector-based RAG

| Feature | Vectorless RAG | Vector-based RAG |
|---------|----------------|------------------|
| Exact keyword matching | Excellent | May miss exact terms |
| Semantic similarity | Limited | Excellent |
| Speed | Very fast | Slower |
| Memory usage | Low | High |
| Interpretability | High | Lower |
| Setup complexity | Simple | Complex |

## Troubleshooting

- NLTK tokenizers in v3.9+ require `punkt_tab` in addition to `punkt`. The library auto-downloads missing resources on first use. If running in a restricted environment, preinstall with:

  ```python
  import nltk
  nltk.download('punkt')
  nltk.download('punkt_tab')
  nltk.download('stopwords')
  nltk.download('wordnet')
  ```

- Tests expect the package importable as `vectorless_rag`. Install in editable mode with `pip install -e .`.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see LICENSE file for details.

## Citation

If you use this system in your research, please cite:

```bibtex
@software{vectorless_rag2025,
  title={Vectorless RAG: High-Accuracy Retrieval Without Embeddings},
  year={2025},
  url={https://github.com/Jimmycarroll2021/vectorless-RAG}
}
```
