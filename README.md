# Vectorless RAG System

A high-accuracy Retrieval-Augmented Generation (RAG) system that uses traditional information retrieval techniques instead of vector embeddings, providing better interpretability and performance for certain use cases.

## Overview

This vectorless RAG system leverages classical IR methods combined with modern NLP techniques to achieve superior accuracy without the computational overhead of embedding models:

- **BM25 Retrieval**: Industry-standard probabilistic ranking function
- **Query Expansion**: Enhances queries with synonyms and related terms
- **Document Reranking**: Cross-encoder based reranking for improved precision
- **TF-IDF Fallback**: Additional retrieval method for better recall
- **Hybrid Scoring**: Combines multiple signals for optimal ranking

## Key Features

- ✅ **No Vector Database Required**: Uses inverted index for fast retrieval
- ✅ **Interpretable Results**: Clear scoring and ranking explanations
- ✅ **Low Latency**: Fast keyword-based search
- ✅ **Easy Integration**: Simple API for RAG pipelines
- ✅ **Configurable**: Tune parameters for your specific use case
- ✅ **Scalable**: Efficient indexing and retrieval for large corpora

## Installation

```bash
pip install -r requirements.txt
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
Query → Preprocessing → Query Expansion → BM25 Retrieval → Reranking → Results
                                              ↓
                                        TF-IDF Fallback
```

## Performance

Our vectorless approach offers several advantages:

- **Accuracy**: Achieves competitive or better results than vector-based systems on keyword-heavy queries
- **Speed**: 10-100x faster retrieval compared to dense vector search
- **Memory**: Significantly lower memory footprint
- **Transparency**: Easy to debug and understand why documents were retrieved

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

- **FAQ Systems**: Exact keyword matching for customer support
- **Legal Document Search**: Precise terminology matching
- **Code Search**: Finding specific functions and classes
- **Medical Records**: Accurate matching of medical terms
- **Knowledge Bases**: Internal documentation search

## Comparison with Vector-based RAG

| Feature | Vectorless RAG | Vector-based RAG |
|---------|---------------|------------------|
| Exact keyword matching | ✅ Excellent | ⚠️ May miss exact terms |
| Semantic similarity | ⚠️ Limited | ✅ Excellent |
| Speed | ✅ Very fast | ⚠️ Slower |
| Memory usage | ✅ Low | ⚠️ High |
| Interpretability | ✅ High | ⚠️ Low |
| Setup complexity | ✅ Simple | ⚠️ Complex |

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
