"""
TF-IDF retrieval implementation for vectorless RAG.
"""

import math
from typing import List, Dict, Tuple
from collections import defaultdict


class TFIDFRetriever:
    """
    TF-IDF (Term Frequency-Inverse Document Frequency) retrieval algorithm.

    Classic information retrieval method using cosine similarity between
    query and document TF-IDF vectors.
    """

    def __init__(self):
        """Initialize TF-IDF retriever."""
        self.documents: List[Dict] = []
        self.tokenized_docs: List[List[str]] = []
        self.doc_vectors: List[Dict[str, float]] = []
        self.idf_scores: Dict[str, float] = {}
        self.doc_freqs: Dict[str, int] = defaultdict(int)
        self.num_docs: int = 0

    def index(self, documents: List[Dict], tokenized_docs: List[List[str]]):
        """
        Index documents for TF-IDF retrieval.

        Args:
            documents: List of document dictionaries with 'id' and 'text'
            tokenized_docs: List of tokenized document texts
        """
        self.documents = documents
        self.tokenized_docs = tokenized_docs
        self.num_docs = len(documents)

        # Calculate document frequencies
        self.doc_freqs = defaultdict(int)
        for tokens in tokenized_docs:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.doc_freqs[token] += 1

        # Calculate IDF scores
        self._calculate_idf()

        # Build TF-IDF vectors for all documents
        self._build_doc_vectors()

    def _calculate_idf(self):
        """Calculate inverse document frequency for all terms."""
        self.idf_scores = {}
        for term, df in self.doc_freqs.items():
            # Standard IDF formula with smoothing
            idf = math.log((self.num_docs + 1) / (df + 1)) + 1.0
            self.idf_scores[term] = idf

    def _calculate_tf(self, tokens: List[str]) -> Dict[str, float]:
        """
        Calculate term frequency for a list of tokens.

        Args:
            tokens: List of tokens

        Returns:
            Dictionary of term frequencies
        """
        tf = defaultdict(int)
        for token in tokens:
            tf[token] += 1

        # Normalize by document length
        doc_length = len(tokens)
        if doc_length > 0:
            for term in tf:
                tf[term] = tf[term] / doc_length

        return dict(tf)

    def _build_doc_vectors(self):
        """Build TF-IDF vectors for all documents."""
        self.doc_vectors = []

        for tokens in self.tokenized_docs:
            tf = self._calculate_tf(tokens)
            tfidf_vector = {}

            for term, tf_score in tf.items():
                if term in self.idf_scores:
                    tfidf_vector[term] = tf_score * self.idf_scores[term]

            # Normalize vector (L2 normalization)
            norm = math.sqrt(sum(score ** 2 for score in tfidf_vector.values()))
            if norm > 0:
                tfidf_vector = {term: score / norm for term, score in tfidf_vector.items()}

            self.doc_vectors.append(tfidf_vector)

    def _build_query_vector(self, query_tokens: List[str]) -> Dict[str, float]:
        """
        Build TF-IDF vector for a query.

        Args:
            query_tokens: Tokenized query

        Returns:
            Query TF-IDF vector
        """
        tf = self._calculate_tf(query_tokens)
        query_vector = {}

        for term, tf_score in tf.items():
            if term in self.idf_scores:
                query_vector[term] = tf_score * self.idf_scores[term]

        # Normalize vector
        norm = math.sqrt(sum(score ** 2 for score in query_vector.values()))
        if norm > 0:
            query_vector = {term: score / norm for term, score in query_vector.items()}

        return query_vector

    def _cosine_similarity(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity score
        """
        # Since vectors are already normalized, dot product = cosine similarity
        common_terms = set(vec1.keys()) & set(vec2.keys())
        return sum(vec1[term] * vec2[term] for term in common_terms)

    def retrieve(self, query_tokens: List[str], top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Retrieve top-k documents for a query using TF-IDF cosine similarity.

        Args:
            query_tokens: Tokenized query
            top_k: Number of top documents to retrieve

        Returns:
            List of (document_index, score) tuples sorted by score descending
        """
        if not self.documents:
            return []

        # Build query vector
        query_vector = self._build_query_vector(query_tokens)

        # Score all documents
        scores = []
        for idx, doc_vector in enumerate(self.doc_vectors):
            score = self._cosine_similarity(query_vector, doc_vector)
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
            "vocab_size": len(self.idf_scores)
        }
