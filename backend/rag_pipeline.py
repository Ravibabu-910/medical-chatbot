import torch
import logging
from pathlib import Path
from typing import Optional

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

from backend.vector_store import VectorStoreManager
from backend.memory import ChatMemory
from backend.config import Config

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Enhanced RAG Pipeline with GraphRAG integration"""
    
    def __init__(self):
        logger.info("🚀 Initializing Enhanced RAG Pipeline with GraphRAG...")
        
        # Initialize memory
        logger.info("📝 Loading Memory...")
        self.memory = ChatMemory(max_len=Config.MAX_HISTORY)
        
        # Load vector store and knowledge graph (preload on start)
        if Config.PRELOAD_ON_START:
            logger.info("🔄 Preloading Vector Store and Knowledge Graph...")
            self.db, self.kg = VectorStoreManager.create_or_load()
            logger.info("✅ Data structures loaded and ready")
        else:
            self.db = None
            self.kg = None
        
        # Load local AI model
        self._load_model()
        
        # Response cache for faster replies
        self.cache = {} if Config.CACHE_ENABLED else None
        
        logger.info("✅ RAG Pipeline Ready - All systems operational")
    
    def _load_model(self):
        """Load local AI model with optimizations"""
        model_path = str(Config.LLM_MODEL.resolve())
        logger.info(f"🤖 Loading Local AI Model from: {model_path}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=True
            )
            
            # Load model with optimizations
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_path,
                local_files_only=True,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            
            # Set to evaluation mode
            self.model.eval()
            
            # Create pipeline for faster inference
            self.generator = pipeline(
                "text2text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=-1,  # CPU
                batch_size=1
            )
            
            logger.info("✅ AI Model loaded and optimized")
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            logger.info("💡 Make sure the model is downloaded to the correct path")
            raise
    
    def _ensure_data_loaded(self):
        """Ensure vector DB and graph are loaded"""
        if self.db is None or self.kg is None:
            logger.info("Loading data structures on-demand...")
            self.db, self.kg = VectorStoreManager.create_or_load()
    
    def generate(self, prompt: str) -> str:
        """Generate response using local AI model"""
        try:
            # Use pipeline for faster generation
            result = self.generator(
                prompt,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                num_return_sequences=1
            )
            
            return result[0]['generated_text'].strip()
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return "I apologize, but I encountered an error generating a response. Please try again."
    
    def query(self, question: str) -> str:
        """
        Main query method with GraphRAG integration
        
        This method:
        1. Checks cache for fast responses
        2. Performs hybrid search (vector + graph)
        3. Synthesizes context from both sources
        4. Generates AI response
        5. Updates memory
        """
        logger.info(f"📥 Query received: {question[:100]}...")
        
        # Check cache
        if self.cache is not None and question in self.cache:
            logger.info("⚡ Cache hit - returning cached response")
            return self.cache[question]
        
        # Ensure data is loaded
        self._ensure_data_loaded()
        
        # Hybrid search: Vector + Graph
        logger.info("🔍 Performing hybrid search...")
        docs, graph_context = VectorStoreManager.search_hybrid(
            question,
            k=Config.TOP_K_SIMILAR
        )
        
        # Extract vector context
        vector_context = "\n".join([d.page_content for d in docs])
        
        # Combine contexts
        if Config.GRAPH_ENABLED and graph_context:
            combined_context = f"""
KNOWLEDGE BASE (Vector Search):
{vector_context}

RELATED ENTITIES (Knowledge Graph):
{graph_context}
"""
        else:
            combined_context = vector_context
        
        logger.info(f"📊 Context assembled: {len(combined_context)} characters")
        
        # Get conversation history
        history = self.memory.get()
        
        # Build prompt
        prompt = self._build_prompt(question, combined_context, history)
        
        # Generate response
        logger.info("🧠 Generating AI response...")
        answer = self.generate(prompt)
        
        # Update memory
        self.memory.add(question, answer)
        
        # Cache response
        if self.cache is not None:
            self.cache[question] = answer
        
        logger.info("✅ Response generated successfully")
        return answer
    
    def _build_prompt(self, question: str, context: str, history: str) -> str:
        """Build optimized prompt for the model"""
        
        if history:
            prompt = f"""You are a professional medical assistant with access to a comprehensive knowledge base.

Previous Conversation:
{history}

Knowledge Base:
{context}

Current Question: {question}

Provide a clear, accurate, and professional answer based on the knowledge base. If the information is not in the knowledge base, say so clearly.

Answer:"""
        else:
            prompt = f"""You are a professional medical assistant with access to a comprehensive knowledge base.

Knowledge Base:
{context}

Question: {question}

Provide a clear, accurate, and professional answer based on the knowledge base.

Answer:"""
        
        return prompt
    
    def clear_cache(self):
        """Clear response cache"""
        if self.cache is not None:
            self.cache.clear()
            logger.info("Cache cleared")
    
    def get_stats(self) -> dict:
        """Get pipeline statistics"""
        self._ensure_data_loaded()
        
        stats = {
            "vector_db_size": self.db.index.ntotal if self.db else 0,
            "graph_nodes": self.kg.graph.number_of_nodes() if self.kg else 0,
            "graph_edges": self.kg.graph.number_of_edges() if self.kg else 0,
            "cache_size": len(self.cache) if self.cache else 0,
            "history_size": len(self.memory.history)
        }
        
        return stats