import logging
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from backend.api import router
from backend.config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('chatbot.log')
    ]
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    # Startup
    logger.info("="*60)
    logger.info("🚀 Medical AI Chatbot Starting Up")
    logger.info("="*60)
    logger.info(f"📁 Base Directory: {Config.BASE_DIR}")
    logger.info(f"📊 Excel Path: {Config.EXCEL_PATH}")
    logger.info(f"💾 Vector Store: {Config.VECTOR_STORE_PATH}")
    logger.info(f"🕸️  Knowledge Graph: {Config.GRAPH_PATH}")
    logger.info(f"🤖 LLM Model: {Config.LLM_MODEL}")
    logger.info(f"🔒 Local Only Mode: {Config.LOCAL_ONLY}")
    logger.info(f"⚡ Preload on Start: {Config.PRELOAD_ON_START}")
    logger.info("="*60)
    
    # Create necessary directories
    Config.VECTOR_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    Config.LLM_MODEL.parent.mkdir(parents=True, exist_ok=True)
    Path("data").mkdir(exist_ok=True)
    
    logger.info("✅ All directories ready")
    logger.info("✅ Server is ready to accept requests")
    logger.info("="*60)
    
    yield
    
    # Shutdown
    logger.info("="*60)
    logger.info("🛑 Shutting down Medical AI Chatbot")
    logger.info("="*60)


# Create FastAPI app with lifespan
app = FastAPI(
    title="Medical AI Chatbot with GraphRAG",
    description="Advanced medical chatbot using GraphRAG for enhanced knowledge retrieval",
    version="2.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include router
app.include_router(router, prefix="/api", tags=["Chat"])


@app.get("/")
def home():
    """Root endpoint"""
    return {
        "status": "running",
        "service": "Medical AI Chatbot with GraphRAG",
        "version": "2.0",
        "features": [
            "GraphRAG Knowledge Retrieval",
            "Local AI Model (No API keys)",
            "Vector + Graph Hybrid Search",
            "Community Detection",
            "Response Caching",
            "Conversation Memory"
        ],
        "endpoints": {
            "chat": "/api/chat",
            "health": "/api/health",
            "stats": "/api/stats",
            "clear_cache": "/api/clear-cache",
            "rebuild": "/api/rebuild"
        }
    }


@app.get("/info")
def info():
    """System information"""
    return {
        "config": {
            "graph_enabled": Config.GRAPH_ENABLED,
            "community_detection": Config.COMMUNITY_DETECTION_ENABLED,
            "max_graph_hops": Config.MAX_GRAPH_HOPS,
            "top_k_similar": Config.TOP_K_SIMILAR,
            "local_only": Config.LOCAL_ONLY,
            "cache_enabled": Config.CACHE_ENABLED,
            "preload_on_start": Config.PRELOAD_ON_START
        },
        "security": {
            "sensitive_data_mode": Config.SENSITIVE_DATA_MODE,
            "no_external_apis": Config.LOCAL_ONLY,
            "log_queries": Config.LOG_QUERIES
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting server with uvicorn...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Set to True for development
        log_level="info"
    )