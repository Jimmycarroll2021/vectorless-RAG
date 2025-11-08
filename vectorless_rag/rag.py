"""
Main Vectorless RAG system implementation.
"""

from typing import List, Dict, Optional, Union
import yaml
from pathlib import Path

from .preprocessor import Preprocessor
from .retrievers.bm25 import BM25Retriever
from .retrievers.tfidf import TFIDFRetriever
from .query_expansion import QueryExpander
from .reranker import Reranker


class VectorlessRAG:
    """
    Main RAG system using traditional IR methods without vector embeddings.

    Combines BM25/TF-IDF retrieval, query expansion, and reranking for
    high-accuracy document retrieval.
    """

    def __init__(
        self,
        preprocessor: Optional[Preprocessor] = None,
        config_path: Optional[str] = None
    ):
        """
        Initialize the Vectorless RAG system.

        Args:
            preprocessor: Custom preprocessor (uses default if None)
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = self._load_config(config_path)

        # Initialize preprocessor
        self.preprocessor = preprocessor or Preprocessor(
            lowercase=True,
            remove_stopwords=True,
            remove_punctuation=True,
            stemming=False,
            lemmatization=True,
            min_token_length=2
        )

        # Initialize retrievers
        self.bm25 = BM25Retriever(
            k1=self.config.get('retrieval', {}).get('k1', 1.5),
            b=self.config.get('retrieval', {}).get('b', 0.75)
        )
        self.tfidf = TFIDFRetriever()

        # Initialize query expander
        self.query_expander = QueryExpander(
            max_expansions=self.config.get('query_expansion', {}).get('max_expansions', 3)
        )

        # Initialize reranker
        self.reranker = Reranker(
            exact_match_boost=self.config.get('reranking', {}).get('exact_match_boost', 2.0),
            title_boost=self.config.get('reranking', {}).get('title_boost', 1.5)
        )

        # Storage
        self.documents: List[Dict] = []
        self.tokenized_docs: List[List[str]] = []

    def _load_config(self, config_path: Optional[str] = None) -> Dict:
        """
        Load configuration from file or use defaults.

        Args:
            config_path: Path to config file

        Returns:
            Configuration dictionary
        """
        default_config = {
            'retrieval': {
                'method': 'bm25',
                'k1': 1.5,
                'b': 0.75
            },
            'query_expansion': {
                'enabled': True,
                'max_expansions': 3
            },
            'reranking': {
                'enabled': True,
                'exact_match_boost': 2.0,
                'title_boost': 1.5
            }
        }

        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                # Merge with defaults
                return {**default_config, **config}

        return default_config

    def index_documents(self, documents: List[Dict]):
        """
        Index documents for retrieval.

        Args:
            documents: List of document dictionaries with 'id' and 'text' fields
                      Optional fields: 'title', 'timestamp', 'metadata'
        """
        self.documents = documents

        # Preprocess all documents
        self.tokenized_docs = []
        for doc in documents:
            text = doc.get('text', '')
            tokens = self.preprocessor.tokenize(text)
            self.tokenized_docs.append(tokens)

        # Index documents in retrievers
        self.bm25.index(documents, self.tokenized_docs)
        self.tfidf.index(documents, self.tokenized_docs)

        print(f"Indexed {len(documents)} documents successfully.")

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        method: str = 'bm25',
        expand_query: Optional[bool] = None,
        rerank: Optional[bool] = None
    ) -> List[Dict]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: Query string
            top_k: Number of top documents to retrieve
            method: Retrieval method ('bm25', 'tfidf', or 'hybrid')
            expand_query: Whether to expand query (uses config default if None)
            rerank: Whether to rerank results (uses config default if None)

        Returns:
            List of retrieved documents with scores
        """
        if not self.documents:
            print("Warning: No documents indexed. Please index documents first.")
            return []

        # Use config defaults if not specified
        if expand_query is None:
            expand_query = self.config.get('query_expansion', {}).get('enabled', True)
        if rerank is None:
            rerank = self.config.get('reranking', {}).get('enabled', True)

        # Preprocess query
        query_tokens = self.preprocessor.tokenize(query)

        # Query expansion
        if expand_query:
            query_tokens = self.query_expander.expand(query_tokens)

        # Retrieve documents
        if method == 'bm25':
            results = self._retrieve_bm25(query_tokens, top_k * 2 if rerank else top_k)
        elif method == 'tfidf':
            results = self._retrieve_tfidf(query_tokens, top_k * 2 if rerank else top_k)
        elif method == 'hybrid':
            results = self._retrieve_hybrid(query_tokens, top_k * 2 if rerank else top_k)
        else:
            raise ValueError(f"Unknown retrieval method: {method}")

        # Rerank if enabled
        if rerank and results:
            results = self._rerank_results(query, results, top_k)

        return results[:top_k]

    def _retrieve_bm25(self, query_tokens: List[str], top_k: int) -> List[Dict]:
        """Retrieve using BM25."""
        scored_docs = self.bm25.retrieve(query_tokens, top_k)

        results = []
        for doc_idx, score in scored_docs:
            doc = self.documents[doc_idx].copy()
            doc['score'] = score
            doc['method'] = 'bm25'
            results.append(doc)

        return results

    def _retrieve_tfidf(self, query_tokens: List[str], top_k: int) -> List[Dict]:
        """Retrieve using TF-IDF."""
        scored_docs = self.tfidf.retrieve(query_tokens, top_k)

        results = []
        for doc_idx, score in scored_docs:
            doc = self.documents[doc_idx].copy()
            doc['score'] = score
            doc['method'] = 'tfidf'
            results.append(doc)

        return results

    def _retrieve_hybrid(
        self,
        query_tokens: List[str],
        top_k: int,
        bm25_weight: float = 0.7,
        tfidf_weight: float = 0.3
    ) -> List[Dict]:
        """
        Retrieve using hybrid approach combining BM25 and TF-IDF.

        Args:
            query_tokens: Tokenized query
            top_k: Number of documents to retrieve
            bm25_weight: Weight for BM25 scores
            tfidf_weight: Weight for TF-IDF scores

        Returns:
            List of retrieved documents with combined scores
        """
        # Get results from both methods
        bm25_results = self.bm25.retrieve(query_tokens, top_k * 2)
        tfidf_results = self.tfidf.retrieve(query_tokens, top_k * 2)

        # Normalize scores to [0, 1]
        bm25_scores = self._normalize_scores([score for _, score in bm25_results])
        tfidf_scores = self._normalize_scores([score for _, score in tfidf_results])

        # Create score dictionaries
        bm25_dict = {idx: score for (idx, _), score in zip(bm25_results, bm25_scores)}
        tfidf_dict = {idx: score for (idx, _), score in zip(tfidf_results, tfidf_scores)}

        # Combine scores
        all_doc_indices = set(bm25_dict.keys()) | set(tfidf_dict.keys())
        combined_scores = []

        for idx in all_doc_indices:
            bm25_score = bm25_dict.get(idx, 0.0)
            tfidf_score = tfidf_dict.get(idx, 0.0)
            combined_score = bm25_weight * bm25_score + tfidf_weight * tfidf_score
            combined_scores.append((idx, combined_score))

        # Sort by combined score
        combined_scores.sort(key=lambda x: x[1], reverse=True)

        # Return top-k documents
        results = []
        for doc_idx, score in combined_scores[:top_k]:
            doc = self.documents[doc_idx].copy()
            doc['score'] = score
            doc['method'] = 'hybrid'
            results.append(doc)

        return results

    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to [0, 1] range."""
        if not scores:
            return []

        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            return [1.0] * len(scores)

        return [(s - min_score) / (max_score - min_score) for s in scores]

    def _rerank_results(self, query: str, results: List[Dict], top_k: int) -> List[Dict]:
        """Rerank retrieved documents."""
        documents = results
        initial_scores = [doc['score'] for doc in results]

        reranked = self.reranker.rerank(query, documents, initial_scores, top_k)

        # Update scores in documents
        reranked_docs = []
        for doc, new_score in reranked:
            doc = doc.copy()
            doc['score'] = new_score
            doc['reranked'] = True
            reranked_docs.append(doc)

        return reranked_docs

    def get_statistics(self) -> Dict:
        """Get system statistics."""
        return {
            'num_documents': len(self.documents),
            'bm25_stats': self.bm25.get_statistics(),
            'tfidf_stats': self.tfidf.get_statistics(),
            'config': self.config
        }

    def save_index(self, path: str):
        """Save indexed documents (placeholder for future implementation)."""
        raise NotImplementedError("Index saving not yet implemented")

    def load_index(self, path: str):
        """Load indexed documents (placeholder for future implementation)."""
        raise NotImplementedError("Index loading not yet implemented")
