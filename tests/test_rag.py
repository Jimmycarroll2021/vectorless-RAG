"""
Tests for the Vectorless RAG system.
"""

import pytest
from vectorless_rag import VectorlessRAG, Preprocessor


class TestVectorlessRAG:
    """Test suite for VectorlessRAG."""

    @pytest.fixture
    def sample_documents(self):
        """Sample documents for testing."""
        return [
            {"id": "1", "text": "Python is a programming language.", "title": "Python"},
            {"id": "2", "text": "Machine learning uses algorithms.", "title": "ML"},
            {"id": "3", "text": "NLP processes natural language.", "title": "NLP"},
        ]

    @pytest.fixture
    def rag(self, sample_documents):
        """Initialize RAG with sample documents."""
        rag = VectorlessRAG()
        rag.index_documents(sample_documents)
        return rag

    def test_initialization(self):
        """Test RAG initialization."""
        rag = VectorlessRAG()
        assert rag is not None
        assert len(rag.documents) == 0

    def test_indexing(self, sample_documents):
        """Test document indexing."""
        rag = VectorlessRAG()
        rag.index_documents(sample_documents)
        assert len(rag.documents) == 3
        assert len(rag.tokenized_docs) == 3

    def test_bm25_retrieval(self, rag):
        """Test BM25 retrieval."""
        results = rag.retrieve("Python programming", top_k=2, method='bm25')
        assert len(results) <= 2
        assert all('score' in r for r in results)
        assert all('id' in r for r in results)

    def test_tfidf_retrieval(self, rag):
        """Test TF-IDF retrieval."""
        results = rag.retrieve("machine learning", top_k=2, method='tfidf')
        assert len(results) <= 2
        assert all('score' in r for r in results)

    def test_hybrid_retrieval(self, rag):
        """Test hybrid retrieval."""
        results = rag.retrieve("natural language", top_k=2, method='hybrid')
        assert len(results) <= 2
        assert all('method' in r for r in results)

    def test_query_expansion(self, rag):
        """Test query expansion."""
        results_no_expansion = rag.retrieve("code", expand_query=False, top_k=3)
        results_with_expansion = rag.retrieve("code", expand_query=True, top_k=3)

        # Both should return results
        assert isinstance(results_no_expansion, list)
        assert isinstance(results_with_expansion, list)

    def test_reranking(self, rag):
        """Test reranking."""
        results_no_rerank = rag.retrieve("Python", rerank=False, top_k=2)
        results_with_rerank = rag.retrieve("Python", rerank=True, top_k=2)

        # Both should return results
        assert len(results_no_rerank) <= 2
        assert len(results_with_rerank) <= 2

    def test_empty_query(self, rag):
        """Test empty query handling."""
        results = rag.retrieve("", top_k=3)
        assert isinstance(results, list)

    def test_no_documents(self):
        """Test retrieval with no documents."""
        rag = VectorlessRAG()
        results = rag.retrieve("test query", top_k=3)
        assert results == []

    def test_statistics(self, rag):
        """Test statistics retrieval."""
        stats = rag.get_statistics()
        assert 'num_documents' in stats
        assert 'bm25_stats' in stats
        assert stats['num_documents'] == 3

    def test_custom_preprocessor(self, sample_documents):
        """Test custom preprocessor."""
        preprocessor = Preprocessor(
            lowercase=True,
            remove_stopwords=False,
            stemming=True
        )
        rag = VectorlessRAG(preprocessor=preprocessor)
        rag.index_documents(sample_documents)

        results = rag.retrieve("programming", top_k=2)
        assert isinstance(results, list)


class TestPreprocessor:
    """Test suite for Preprocessor."""

    def test_initialization(self):
        """Test preprocessor initialization."""
        preprocessor = Preprocessor()
        assert preprocessor is not None

    def test_lowercase(self):
        """Test lowercase conversion."""
        preprocessor = Preprocessor(lowercase=True)
        result = preprocessor.preprocess("HELLO World")
        assert result.islower()

    def test_tokenize(self):
        """Test tokenization."""
        preprocessor = Preprocessor()
        tokens = preprocessor.tokenize("This is a test.")
        assert isinstance(tokens, list)
        assert len(tokens) > 0

    def test_empty_text(self):
        """Test empty text handling."""
        preprocessor = Preprocessor()
        result = preprocessor.preprocess("")
        assert result == ""
        tokens = preprocessor.tokenize("")
        assert tokens == []
