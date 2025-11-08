"""
Vectorless RAG - A high-accuracy RAG system using traditional IR methods.
"""

from .rag import VectorlessRAG
from .preprocessor import Preprocessor

__version__ = "0.1.0"
__all__ = ["VectorlessRAG", "Preprocessor"]
