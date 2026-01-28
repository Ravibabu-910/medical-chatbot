from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import logging

from backend.rag_pipeline import RAGPipeline
from backend.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize RAG pipeline (will preload if configured)
rag = RAGPipeline()


class Query(BaseModel):
    question: str


class RebuildRequest(BaseModel):
    confirm: bool = False


@router.post("/chat")
def chat(query: Query):
    """
    Main chat endpoint with GraphRAG
    
    Fast response time through:
    - Preloaded data structures
    - Response caching
    - Optimized model inference
    """
    try:
        if not query.question or query.question.strip() == "":
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        logger.info(f"Received query: {query.question[:100]}")
        
        answer = rag.query(query.question)
        
        return {
            "answer": answer,
            "status": "success"
        }
    
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@router.get("/health")
def health_check():
    """Health check endpoint"""
    try:
        stats = rag.get_stats()
        
        return {
            "status": "healthy",
            "model_loaded": True,
            "vector_db_ready": stats["vector_db_size"] > 0,
            "graph_ready": stats["graph_nodes"] > 0,
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }


@router.get("/stats")
def get_stats():
    """Get detailed statistics about the RAG system"""
    try:
        stats = rag.get_stats()
        return {
            "status": "success",
            "data": stats
        }
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clear-cache")
def clear_cache():
    """Clear response cache"""
    try:
        rag.clear_cache()
        return {
            "status": "success",
            "message": "Cache cleared"
        }
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rebuild")
def rebuild_database(request: RebuildRequest):
    """
    Rebuild vector database and knowledge graph
    
    WARNING: This will take time and should only be used when data changes
    """
    if not request.confirm:
        raise HTTPException(
            status_code=400,
            detail="Please set 'confirm: true' to rebuild the database"
        )
    
    try:
        logger.info("Starting database rebuild...")
        
        # Rebuild
        db, kg = VectorStoreManager.rebuild()
        
        # Update RAG pipeline references
        rag.db = db
        rag.kg = kg
        rag.clear_cache()
        
        return {
            "status": "success",
            "message": "Database rebuilt successfully",
            "stats": rag.get_stats()
        }
    
    except Exception as e:
        logger.error(f"Error rebuilding database: {e}")
        raise HTTPException(status_code=500, detail=str(e))