"""
Reranking module for improved precision in retrieval results.
"""

from typing import List, Dict, Optional
import re


class Reranker:
    """
    Reranks retrieved documents using various scoring signals.

    This implementation uses heuristic-based reranking without requiring
    external ML models, making it fast and dependency-light.
    """

    def __init__(
        self,
        exact_match_boost: float = 2.0,
        title_boost: float = 1.5,
        recency_boost: float = 1.2,
        length_penalty: float = 0.1
    ):
        """
        Initialize reranker.

        Args:
            exact_match_boost: Score multiplier for exact phrase matches
            title_boost: Score multiplier for matches in document titles
            recency_boost: Score multiplier for recent documents (if timestamp available)
            length_penalty: Penalty factor for very long documents
        """
        self.exact_match_boost = exact_match_boost
        self.title_boost = title_boost
        self.recency_boost = recency_boost
        self.length_penalty = length_penalty

    def rerank(
        self,
        query: str,
        documents: List[Dict],
        initial_scores: List[float],
        top_k: Optional[int] = None
    ) -> List[tuple]:
        """
        Rerank documents based on additional signals.

        Args:
            query: Original query string
            documents: List of retrieved documents
            initial_scores: Initial retrieval scores
            top_k: Number of top documents to return after reranking

        Returns:
            List of (document, final_score) tuples sorted by final score
        """
        if not documents:
            return []

        reranked = []

        for doc, initial_score in zip(documents, initial_scores):
            # Start with initial retrieval score
            final_score = initial_score

            # Apply reranking signals
            final_score *= self._exact_match_score(query, doc)
            final_score *= self._title_match_score(query, doc)
            final_score *= self._length_score(doc)
            final_score *= self._recency_score(doc)
            final_score *= self._keyword_density_score(query, doc)

            reranked.append((doc, final_score))

        # Sort by final score descending
        reranked.sort(key=lambda x: x[1], reverse=True)

        # Return top-k if specified
        if top_k is not None:
            reranked = reranked[:top_k]

        return reranked

    def _exact_match_score(self, query: str, document: Dict) -> float:
        """
        Boost score if query appears exactly in document.

        Args:
            query: Query string
            document: Document dictionary

        Returns:
            Score multiplier
        """
        doc_text = document.get('text', '').lower()
        query_lower = query.lower()

        if query_lower in doc_text:
            return self.exact_match_boost
        return 1.0

    def _title_match_score(self, query: str, document: Dict) -> float:
        """
        Boost score if query terms appear in document title.

        Args:
            query: Query string
            document: Document dictionary

        Returns:
            Score multiplier
        """
        title = document.get('title', '').lower()
        if not title:
            return 1.0

        query_terms = set(query.lower().split())
        title_terms = set(title.split())

        # Calculate overlap
        overlap = len(query_terms & title_terms)
        if overlap > 0:
            overlap_ratio = overlap / len(query_terms)
            return 1.0 + (self.title_boost - 1.0) * overlap_ratio

        return 1.0

    def _length_score(self, document: Dict) -> float:
        """
        Apply penalty for very long or very short documents.

        Args:
            document: Document dictionary

        Returns:
            Score multiplier
        """
        text = document.get('text', '')
        length = len(text.split())

        # Optimal length range: 50-500 words
        if 50 <= length <= 500:
            return 1.0
        elif length < 50:
            return 1.0 - self.length_penalty * (50 - length) / 50
        else:  # length > 500
            penalty = min(0.5, self.length_penalty * (length - 500) / 1000)
            return 1.0 - penalty

    def _recency_score(self, document: Dict) -> float:
        """
        Boost more recent documents.

        Args:
            document: Document dictionary

        Returns:
            Score multiplier
        """
        # This is a placeholder - in real implementation, you'd use document timestamps
        timestamp = document.get('timestamp')
        if timestamp is None:
            return 1.0

        # Simple recency boost based on timestamp
        # (Implementation would depend on timestamp format)
        return self.recency_boost

    def _keyword_density_score(self, query: str, document: Dict) -> float:
        """
        Score based on keyword density in document.

        Args:
            query: Query string
            document: Document dictionary

        Returns:
            Score multiplier
        """
        text = document.get('text', '').lower()
        query_terms = query.lower().split()

        if not text or not query_terms:
            return 1.0

        words = text.split()
        total_words = len(words)

        if total_words == 0:
            return 1.0

        # Count query term occurrences
        term_count = sum(words.count(term) for term in query_terms)

        # Calculate density (but cap it to avoid over-weighting)
        density = term_count / total_words
        density_score = min(1.5, 1.0 + density * 10)  # Max 1.5x boost

        return density_score

    def cross_encoder_rerank(
        self,
        query: str,
        documents: List[Dict],
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k: Optional[int] = None
    ) -> List[tuple]:
        """
        Rerank using a cross-encoder model (requires sentence-transformers).

        This is an advanced feature that requires additional dependencies.

        Args:
            query: Query string
            documents: List of documents
            model_name: Cross-encoder model name
            top_k: Number of top documents to return

        Returns:
            List of (document, score) tuples
        """
        try:
            from sentence_transformers import CrossEncoder

            model = CrossEncoder(model_name)

            # Prepare pairs for cross-encoder
            pairs = [(query, doc.get('text', '')) for doc in documents]

            # Get scores
            scores = model.predict(pairs)

            # Combine documents with scores
            reranked = list(zip(documents, scores))

            # Sort by score descending
            reranked.sort(key=lambda x: x[1], reverse=True)

            # Return top-k if specified
            if top_k is not None:
                reranked = reranked[:top_k]

            return reranked

        except ImportError:
            # Fall back to heuristic reranking if sentence-transformers not available
            print("Warning: sentence-transformers not installed. Using heuristic reranking.")
            return self.rerank(query, documents, [1.0] * len(documents), top_k)
