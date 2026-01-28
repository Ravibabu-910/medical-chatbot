import os

# Disable SSL verification for local environment
os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"
os.environ["PYTHONHTTPSVERIFY"] = "0"

from langchain_huggingface import HuggingFaceEmbeddings
from backend.config import Config


class EmbeddingsManager:
    """
    Singleton manager for HuggingFace embeddings
    
    Uses local sentence-transformers model for:
    - Document embeddings (vector store)
    - Query embeddings (search)
    - Node embeddings (knowledge graph)
    
    No API keys required - fully local and secure
    """
    
    _embeddings = None

    @staticmethod
    def get():
        """Get or create embeddings instance"""
        if not EmbeddingsManager._embeddings:
            print(f"Loading embedding model: {Config.EMBEDDING_MODEL}")
            
            EmbeddingsManager._embeddings = HuggingFaceEmbeddings(
                model_name=Config.EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            print("✅ Embeddings model loaded")

        return EmbeddingsManager._embeddings