# 🏥 Medical AI Chatbot with GraphRAG

An advanced medical Q&A chatbot using **GraphRAG** (Graph-based Retrieval Augmented Generation) for enhanced knowledge retrieval from Excel data. **100% local, no external APIs, fully secure.**

## 🚀 Key Features

### GraphRAG Architecture
- **Hybrid Retrieval**: Combines vector similarity search + knowledge graph traversal
- **Community Detection**: Automatically identifies related medical concepts
- **Multi-hop Reasoning**: Traverses relationships between diseases, medicines, and symptoms
- **Fast Response**: Preloaded data structures and intelligent caching

### Security & Privacy
- ✅ **100% Local**: No OpenAI, no external APIs
- ✅ **Sensitive Data Protection**: Medical data never leaves your machine
- ✅ **No API Keys Required**: Uses local HuggingFace models
- ✅ **HIPAA-Ready**: Designed for sensitive medical information

### Performance Optimizations
- ⚡ **Preloading**: Vector DB and graph loaded at startup
- ⚡ **Response Caching**: Instant replies for repeated questions
- ⚡ **Batch Processing**: Efficient embedding generation
- ⚡ **Optimized Inference**: Fast local AI model

## 📋 Prerequisites

- Python 3.9 or higher
- 4GB+ RAM recommended
- 2GB disk space for models and data

## 🛠️ Installation

### 1. Clone or Download

```bash
# If you're using Git
git clone <your-repo-url>
cd medical-chatbot-graphrag

# Or just extract the files to a folder
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- FastAPI for the API
- LangChain for RAG pipeline
- FAISS for vector storage
- NetworkX for knowledge graphs
- sentence-transformers for embeddings
- PyTorch for AI model
- And all other required packages

### 4. Download the AI Model

You need to download the Flan-T5 model (or any other local model):

**Option A: Automatic Download Script**

```python
# save this as download_model.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "google/flan-t5-small"  # or flan-t5-base for better quality
save_path = "./model/flan-t5-small"

print(f"Downloading {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

print(f"Saving to {save_path}...")
tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)

print("✅ Model downloaded successfully!")
```

Run it:
```bash
python download_model.py
```

**Option B: Manual Download**

1. Go to HuggingFace: https://huggingface.co/google/flan-t5-small
2. Download all files to `model/flan-t5-small/`

### 5. Prepare Your Data

Place your Excel file at `data/manual.xlsx`

**Excel Format Requirements:**
- Can have multiple sheets
- Each sheet will be processed
- Common columns: Disease, Medicine, Symptoms, Treatment, etc.
- The system will automatically create relationships between entities

**Example Excel Structure:**

| Disease | Symptoms | Medicine | Dosage |
|---------|----------|----------|--------|
| Flu | Fever, Cough | Paracetamol | 500mg |
| Diabetes | High Blood Sugar | Metformin | 850mg |

## 🚀 Running the Chatbot

### Start the Server

```bash
# From the project root directory
python main.py
```

Or with uvicorn directly:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### First Run

On first run, the system will:
1. ✅ Load your Excel data
2. ✅ Create vector embeddings (1-2 minutes)
3. ✅ Build knowledge graph with relationships
4. ✅ Detect communities of related concepts
5. ✅ Save everything for fast future loading

**Subsequent runs are instant** - everything loads from disk!

### Check System Status

Open your browser: http://localhost:8000

You should see:
```json
{
  "status": "running",
  "service": "Medical AI Chatbot with GraphRAG",
  "version": "2.0"
}
```

## 📡 API Endpoints

### 1. Chat (Main Endpoint)

```bash
POST http://localhost:8000/api/chat

Body:
{
  "question": "What medicine is used for flu?"
}

Response:
{
  "answer": "Paracetamol is commonly used...",
  "status": "success"
}
```

### 2. Health Check

```bash
GET http://localhost:8000/api/health

Response:
{
  "status": "healthy",
  "vector_db_ready": true,
  "graph_ready": true,
  "stats": {
    "vector_db_size": 150,
    "graph_nodes": 200,
    "graph_edges": 350
  }
}
```

### 3. Get Statistics

```bash
GET http://localhost:8000/api/stats

Response:
{
  "status": "success",
  "data": {
    "vector_db_size": 150,
    "graph_nodes": 200,
    "graph_edges": 350,
    "cache_size": 10,
    "history_size": 5
  }
}
```

### 4. Clear Cache

```bash
POST http://localhost:8000/api/clear-cache

Response:
{
  "status": "success",
  "message": "Cache cleared"
}
```

### 5. Rebuild Database

**Use when you update your Excel data:**

```bash
POST http://localhost:8000/api/rebuild

Body:
{
  "confirm": true
}

Response:
{
  "status": "success",
  "message": "Database rebuilt successfully"
}
```

## 🧪 Testing

### Using cURL

```bash
# Simple test
curl -X POST "http://localhost:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"What are the symptoms of flu?\"}"

# Health check
curl http://localhost:8000/api/health
```

### Using Python

```python
import requests

# Chat request
response = requests.post(
    "http://localhost:8000/api/chat",
    json={"question": "What medicine treats diabetes?"}
)
print(response.json())

# Health check
health = requests.get("http://localhost:8000/api/health")
print(health.json())
```

### Using Postman

1. Import the collection or create new requests
2. POST to `http://localhost:8000/api/chat`
3. Set body to JSON:
```json
{
  "question": "Your question here"
}
```

## 📁 Project Structure

```
medical-chatbot-graphrag/
├── backend/
│   ├── __init__.py
│   ├── config.py              # Configuration settings
│   ├── embeddings.py          # Local embeddings manager
│   ├── data_loader.py         # Excel data processing
│   ├── graph_rag.py           # Knowledge graph (NEW!)
│   ├── vector_store.py        # Vector + Graph management
│   ├── rag_pipeline.py        # Main RAG pipeline
│   ├── memory.py              # Conversation memory
│   └── api.py                 # FastAPI endpoints
├── main.py                     # Server entry point
├── requirements.txt            # Python dependencies
├── data/
│   └── manual.xlsx            # Your Excel data
├── model/
│   └── flan-t5-small/         # Local AI model
└── storage/                    # Auto-generated
    ├── vector_db/             # Vector embeddings
    ├── graph.gml              # Knowledge graph
    ├── graph_embeddings.pkl   # Graph node embeddings
    └── community_summaries.pkl # Community data
```

## ⚙️ Configuration

Edit `backend/config.py` to customize:

```python
class Config:
    # Data paths
    EXCEL_PATH = "data/manual.xlsx"
    
    # Model selection
    LLM_MODEL = "model/flan-t5-small"
    
    # GraphRAG settings
    GRAPH_ENABLED = True              # Enable/disable graph
    MAX_GRAPH_HOPS = 2                # Graph traversal depth
    TOP_K_SIMILAR = 5                 # Results to retrieve
    
    # Performance
    PRELOAD_ON_START = True           # Fast startup
    CACHE_ENABLED = True              # Response caching
    
    # Security
    LOCAL_ONLY = True                 # No external APIs
    SENSITIVE_DATA_MODE = True        # Extra protection
```

## 🔧 Troubleshooting

### Model Not Loading
```
Error: Model not found at model/flan-t5-small
```
**Solution**: Run the download script or manually download the model

### Excel File Not Found
```
Error: Excel file not found at data/manual.xlsx
```
**Solution**: Place your Excel file in the `data/` folder

### Out of Memory
```
Error: RuntimeError: out of memory
```
**Solution**: 
- Use smaller model (flan-t5-small instead of base)
- Reduce batch size in config
- Close other applications

### Slow Responses
**Optimizations**:
1. Enable `PRELOAD_ON_START = True`
2. Enable `CACHE_ENABLED = True`
3. Reduce `MAX_GRAPH_HOPS` to 1
4. Use `flan-t5-small` instead of larger models

### Graph Not Building
```
Warning: Graph has 0 nodes
```
**Solution**: Check Excel column names and format

## 🎯 How GraphRAG Works

### Traditional RAG
```
Question → Vector Search → Top Documents → Generate Answer
```

### GraphRAG (This Implementation)
```
Question → Vector Search + Graph Traversal → 
  ↓
Documents + Related Entities + Relationships → 
  ↓
Enriched Context → Better Answer
```

### Example

**Question:** "What treats diabetes?"

**Vector Search finds:**
- "Diabetes is treated with Metformin"

**Graph adds:**
- Related symptoms: High blood sugar, fatigue
- Related conditions: Type 2 Diabetes, Insulin resistance
- Related medicines: Insulin, Glipizide
- Treatment relationships and dosages

**Result:** Much richer, more comprehensive answer!

## 🔐 Security Features

1. **No External APIs**: Everything runs locally
2. **No Data Leakage**: Data never sent to cloud
3. **Local AI Models**: No OpenAI or cloud dependencies
4. **Secure Storage**: All data stored locally
5. **No Query Logging**: Optional logging can be disabled
6. **CORS Protected**: Configurable CORS policies

## 📊 Performance Metrics

On typical hardware (Intel i5, 8GB RAM):

- **First Load**: 1-2 minutes (builds everything)
- **Subsequent Loads**: 2-3 seconds (loads from disk)
- **Query Response**: 1-3 seconds
- **Cached Response**: <100ms
- **Memory Usage**: ~2GB

## 🚀 Deployment to GitHub

```bash
# Initialize git
git init

# Add files
git add .

# Commit
git commit -m "Initial commit: Medical AI Chatbot with GraphRAG"

# Add remote
git remote add origin <your-github-repo-url>

# Push
git push -u origin main
```

### .gitignore
Create `.gitignore`:
```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/

# Data and Models (large files)
storage/
model/
*.pkl
*.gml

# Logs
*.log

# OS
.DS_Store
Thumbs.db
```

## 📝 Future Enhancements

- [ ] Multiple document formats (PDF, DOCX)
- [ ] Real-time data updates
- [ ] Advanced graph algorithms
- [ ] Multi-language support
- [ ] Web UI interface
- [ ] Docker containerization
- [ ] Batch query processing

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

MIT License - feel free to use for any purpose

## 🆘 Support

For issues or questions:
1. Check the troubleshooting section
2. Review the logs in `chatbot.log`
3. Open an issue on GitHub

## 🎉 Success Checklist

- [ ] Python 3.9+ installed
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Model downloaded to `model/flan-t5-small/`
- [ ] Excel file placed at `data/manual.xlsx`
- [ ] Server starts successfully (`python main.py`)
- [ ] Health check returns "healthy"
- [ ] Test query returns answer
- [ ] Ready for deployment!

---

**Built with ❤️ for secure, fast, and intelligent medical information retrieval**