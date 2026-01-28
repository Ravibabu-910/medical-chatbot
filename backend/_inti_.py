"""
Medical AI Chatbot Backend with GraphRAG

This package implements an advanced RAG (Retrieval Augmented Generation) system
combining vector search and knowledge graph traversal for medical Q&A.

Key Features:
- GraphRAG: Hybrid vector + graph retrieval
- Local AI: No external API calls
- Fast: Preloaded data structures and caching
- Secure: Sensitive data stays local

Modules:
- config: Configuration settings
- embeddings: Local embeddings manager
- data_loader: Excel data loading and processing
- graph_rag: Knowledge graph construction and traversal
- vector_store: Vector database management
- rag_pipeline: Main RAG pipeline with GraphRAG
- memory: Conversation memory
- api: FastAPI endpoints
"""

__version__ = "2.0.0"
__author__ = "Medical AI Team"

from backend.config import Config
from backend.embeddings import EmbeddingsManager
from backend.rag_pipeline import RAGPipeline
from backend.vector_store import VectorStoreManager

__all__ = [
    "Config",
    "EmbeddingsManager",
    "RAGPipeline",
    "VectorStoreManager"
]