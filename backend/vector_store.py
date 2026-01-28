import os
import logging
from typing import List, Dict

from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

from backend.embeddings import EmbeddingsManager
from backend.data_loader import DataLoader
from backend.graph_rag import KnowledgeGraph
from backend.config import Config

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Enhanced Vector Store Manager with GraphRAG integration"""
    
    _vector_db = None
    _knowledge_graph = None
    
    @staticmethod
    def create_or_load():
        """Create or load vector database and knowledge graph"""
        embeddings = EmbeddingsManager.get()
        
        # Check if both vector DB and graph exist
        vector_exists = os.path.exists(Config.VECTOR_STORE_PATH)
        graph_exists = os.path.exists(Config.GRAPH_PATH)
        
        if vector_exists and graph_exists:
            logger.info("Loading existing vector database and knowledge graph...")
            
            # Load vector database
            VectorStoreManager._vector_db = FAISS.load_local(
                Config.VECTOR_STORE_PATH,
                embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info(f"✅ Vector DB loaded: {VectorStoreManager._vector_db.index.ntotal} vectors")
            
            # Load knowledge graph
            if Config.GRAPH_ENABLED:
                VectorStoreManager._knowledge_graph = KnowledgeGraph()
                if VectorStoreManager._knowledge_graph.load():
                    logger.info("✅ Knowledge Graph loaded")
                else:
                    logger.warning("Failed to load graph, will rebuild")
                    VectorStoreManager._build_new(embeddings)
            
            return VectorStoreManager._vector_db, VectorStoreManager._knowledge_graph
        
        else:
            logger.info("Building new vector database and knowledge graph...")
            return VectorStoreManager._build_new(embeddings)
    
    @staticmethod
    def _build_new(embeddings):
        """Build new vector database and knowledge graph from scratch"""
        logger.info("🔨 Building new vector database and knowledge graph...")
        
        # Create storage directory if needed
        os.makedirs(Config.VECTOR_STORE_PATH.parent, exist_ok=True)
        
        # Load all sheets from Excel
        sheets = DataLoader.load_all_sheets(Config.EXCEL_PATH)
        
        # Create documents for vector store
        documents = DataLoader.create_documents_from_sheets(sheets)
        logger.info(f"Created {len(documents)} documents")
        
        # Build vector database
        VectorStoreManager._vector_db = FAISS.from_documents(documents, embeddings)
        VectorStoreManager._vector_db.save_local(Config.VECTOR_STORE_PATH)
        logger.info(f"✅ Vector DB created: {VectorStoreManager._vector_db.index.ntotal} vectors")
        
        # Build knowledge graph
        if Config.GRAPH_ENABLED:
            VectorStoreManager._knowledge_graph = KnowledgeGraph()
            VectorStoreManager._knowledge_graph.build_from_sheets(sheets, embeddings)
            VectorStoreManager._knowledge_graph.save()
            logger.info("✅ Knowledge Graph created")
        
        return VectorStoreManager._vector_db, VectorStoreManager._knowledge_graph
    
    @staticmethod
    def get_vector_db():
        """Get the loaded vector database"""
        if VectorStoreManager._vector_db is None:
            VectorStoreManager.create_or_load()
        return VectorStoreManager._vector_db
    
    @staticmethod
    def get_knowledge_graph():
        """Get the loaded knowledge graph"""
        if VectorStoreManager._knowledge_graph is None:
            VectorStoreManager.create_or_load()
        return VectorStoreManager._knowledge_graph
    
    @staticmethod
    def search_hybrid(query: str, k: int = 5) -> tuple[List[Document], str]:
        """
        Hybrid search combining vector similarity and graph traversal
        Returns: (documents, graph_context)
        """
        # Vector search
        vector_db = VectorStoreManager.get_vector_db()
        docs = vector_db.similarity_search(query, k=k)
        
        # Graph search
        graph_context = ""
        if Config.GRAPH_ENABLED:
            kg = VectorStoreManager.get_knowledge_graph()
            if kg and kg.embeddings:
                # Get query embedding
                embeddings = EmbeddingsManager.get()
                query_embedding = embeddings.embed_query(query)
                
                # Get relevant subgraph
                subgraph = kg.get_relevant_subgraph(
                    query_embedding,
                    top_k=Config.TOP_K_SIMILAR,
                    max_hops=Config.MAX_GRAPH_HOPS
                )
                
                # Extract context from subgraph
                graph_context = kg.get_context_from_subgraph(subgraph)
        
        return docs, graph_context
    
    @staticmethod
    def rebuild():
        """Force rebuild of vector database and knowledge graph"""
        logger.info("Forcing rebuild of vector database and knowledge graph...")
        
        # Delete existing files
        import shutil
        if os.path.exists(Config.VECTOR_STORE_PATH):
            shutil.rmtree(Config.VECTOR_STORE_PATH)
        if os.path.exists(Config.GRAPH_PATH):
            os.remove(Config.GRAPH_PATH)
        if os.path.exists(Config.GRAPH_EMBEDDINGS_PATH):
            os.remove(Config.GRAPH_EMBEDDINGS_PATH)
        if os.path.exists(Config.COMMUNITY_SUMMARY_PATH):
            os.remove(Config.COMMUNITY_SUMMARY_PATH)
        
        # Rebuild
        embeddings = EmbeddingsManager.get()
        return VectorStoreManager._build_new(embeddings)