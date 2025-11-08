"""
BM25 retrieval implementation for vectorless RAG.
"""

import math
from typing import List, Dict, Tuple
from collections import defaultdict


class BM25Retriever:
    """
    BM25 (Best Matching 25) retrieval algorithm.

    A probabilistic ranking function used for document retrieval based on
    term frequency and inverse document frequency.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 retriever.

        Args:
            k1: Term frequency saturation parameter (typically 1.2-2.0)
            b: Length normalization parameter (typically 0.75)
        """
        self.k1 = k1
        self.b = b

        self.documents: List[Dict] = []
        self.tokenized_docs: List[List[str]] = []
        self.doc_lengths: List[int] = []
        self.avg_doc_length: float = 0.0
        self.idf_scores: Dict[str, float] = {}
        self.doc_freqs: Dict[str, int] = defaultdict(int)
        self.num_docs: int = 0

    def index(self, documents: List[Dict], tokenized_docs: List[List[str]]):
        """
        Index documents for BM25 retrieval.

        Args:
            documents: List of document dictionaries with 'id' and 'text'
            tokenized_docs: List of tokenized document texts
        """
        self.documents = documents
        self.tokenized_docs = tokenized_docs
        self.num_docs = len(documents)

        # Calculate document lengths
        self.doc_lengths = [len(doc) for doc in tokenized_docs]
        self.avg_doc_length = sum(self.doc_lengths) / self.num_docs if self.num_docs > 0 else 0

        # Calculate document frequencies
        self.doc_freqs = defaultdict(int)
        for tokens in tokenized_docs:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.doc_freqs[token] += 1

        # Calculate IDF scores
        self._calculate_idf()

    def _calculate_idf(self):
        """Calculate inverse document frequency for all terms."""
        self.idf_scores = {}
        for term, df in self.doc_freqs.items():
            # BM25 IDF formula
            idf = math.log((self.num_docs - df + 0.5) / (df + 0.5) + 1.0)
            self.idf_scores[term] = idf

    def _score_document(self, query_tokens: List[str], doc_idx: int) -> float:
        """
        Calculate BM25 score for a document given a query.

        Args:
            query_tokens: Tokenized query
            doc_idx: Document index

        Returns:
            BM25 score
        """
        score = 0.0
        doc_tokens = self.tokenized_docs[doc_idx]
        doc_length = self.doc_lengths[doc_idx]

        # Count term frequencies in document
        term_freqs = defaultdict(int)
        for token in doc_tokens:
            term_freqs[token] += 1

        # Calculate BM25 score
        for term in query_tokens:
            if term not in self.idf_scores:
                continue

            tf = term_freqs[term]
            idf = self.idf_scores[term]

            # BM25 formula
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))

            score += idf * (numerator / denominator)

        return score

    def retrieve(self, query_tokens: List[str], top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Retrieve top-k documents for a query using BM25.

        Args:
            query_tokens: Tokenized query
            top_k: Number of top documents to retrieve

        Returns:
            List of (document_index, score) tuples sorted by score descending
        """
        if not self.documents:
            return []

        # Score all documents
        scores = []
        for idx in range(self.num_docs):
            score = self._score_document(query_tokens, idx)
            if score > 0:  # Only include documents with non-zero scores
                scores.append((idx, score))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        # Return top-k
        return scores[:top_k]

    def get_statistics(self) -> Dict:
        """Get retriever statistics."""
        return {
            "num_documents": self.num_docs,
            "avg_doc_length": self.avg_doc_length,
            "vocab_size": len(self.idf_scores),
            "k1": self.k1,
            "b": self.b
        }
