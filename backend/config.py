from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

class Config:
    # Data paths
    EXCEL_PATH = BASE_DIR / "data" / "manual.xlsx"
    
    # Storage paths
    VECTOR_STORE_PATH = BASE_DIR / "storage" / "vector_db"
    GRAPH_PATH = BASE_DIR / "storage" / "graph.gml"
    GRAPH_EMBEDDINGS_PATH = BASE_DIR / "storage" / "graph_embeddings.pkl"
    COMMUNITY_SUMMARY_PATH = BASE_DIR / "storage" / "community_summaries.pkl"
    
    # Model paths (Local - No API keys needed)
    LLM_MODEL = BASE_DIR / "model" / "flan-t5-small"
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    
    # GraphRAG Configuration
    GRAPH_ENABLED = True
    COMMUNITY_DETECTION_ENABLED = True
    MAX_GRAPH_HOPS = 2  # How many connections to traverse
    TOP_K_SIMILAR = 5   # Number of similar nodes to retrieve
    
    # Performance settings
    MAX_HISTORY = 5
    BATCH_SIZE = 32
    CACHE_ENABLED = True
    PRELOAD_ON_START = True  # Load everything at startup for fast responses
    
    # Security settings
    LOCAL_ONLY = True  # No external API calls
    SENSITIVE_DATA_MODE = True
    LOG_QUERIES = False  # Don't log sensitive queries