"""
Query expansion module for improved retrieval accuracy.
"""

from typing import List, Set
from nltk.corpus import wordnet
import nltk

# Download WordNet if not already present
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)


class QueryExpander:
    """
    Expands queries with synonyms and related terms to improve recall.
    """

    def __init__(self, max_expansions: int = 3, min_similarity: float = 0.7):
        """
        Initialize query expander.

        Args:
            max_expansions: Maximum number of expansion terms per query term
            min_similarity: Minimum similarity threshold for expansions (not used with WordNet)
        """
        self.max_expansions = max_expansions
        self.min_similarity = min_similarity

    def expand(self, query_tokens: List[str]) -> List[str]:
        """
        Expand query tokens with synonyms.

        Args:
            query_tokens: List of query tokens

        Returns:
            Expanded list of tokens including original and synonyms
        """
        expanded_tokens = list(query_tokens)  # Start with original tokens
        added_terms: Set[str] = set(query_tokens)

        for token in query_tokens:
            synonyms = self._get_synonyms(token)

            # Add top synonyms up to max_expansions
            count = 0
            for synonym in synonyms:
                if synonym not in added_terms and count < self.max_expansions:
                    expanded_tokens.append(synonym)
                    added_terms.add(synonym)
                    count += 1

        return expanded_tokens

    def _get_synonyms(self, word: str) -> List[str]:
        """
        Get synonyms for a word using WordNet.

        Args:
            word: Input word

        Returns:
            List of synonyms
        """
        synonyms = set()

        # Get all synsets for the word
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().lower().replace('_', ' ')

                # Skip the word itself and multi-word expressions
                if synonym != word and ' ' not in synonym:
                    synonyms.add(synonym)

        return list(synonyms)

    def expand_with_phrases(self, query: str, query_tokens: List[str]) -> List[str]:
        """
        Expand query with common phrase variations.

        Args:
            query: Original query string
            query_tokens: Tokenized query

        Returns:
            Expanded tokens with phrase variations
        """
        expanded = self.expand(query_tokens)

        # Add common query transformations
        transformations = {
            'what': ['which', 'describe'],
            'how': ['method', 'way', 'process'],
            'why': ['reason', 'cause', 'purpose'],
            'when': ['time', 'date'],
            'where': ['location', 'place'],
        }

        for token in query_tokens:
            if token in transformations:
                expanded.extend(transformations[token])

        return list(set(expanded))  # Remove duplicates
