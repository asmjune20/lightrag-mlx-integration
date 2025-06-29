# LightRAG + MLX Integration

This repository contains a modified version of LightRAG that integrates with Apple's MLX framework for local language model inference, eliminating the need for OpenAI API keys while providing fast, local RAG capabilities.

## 🚀 Features

- **Local LLM Inference**: Uses MLX framework with Apple Silicon optimization
- **Local Embeddings**: Sentence-transformers for vector embeddings (no OpenAI required)
- **Knowledge Graph Disabled**: Optimized for simple document retrieval
- **Naive Query Mode**: Fast vector-based document search
- **Streaming Responses**: Real-time response generation
- **Web UI**: Interactive interface for document management and querying

## 📋 Prerequisites

- macOS with Apple Silicon (M1/M2/M3)
- Python 3.8+
- MLX framework
- Flox environment (optional but recommended)

## 🛠 Installation

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd lightrag_new
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Install MLX and Start MLX Server
```bash
# In a separate terminal or flox environment
uv run mlx_lm.server --model mlx-community/Mistral-7B-Instruct-v0.3-4bit --port 8000
```

### 4. Configure Environment
Create a `.env` file in the project root:
```env
# LLM Configuration
LLM_BINDING=openai
LLM_BINDING_HOST=http://localhost:8000/v1
LLM_BINDING_API_KEY=local
LLM_MODEL_NAME=mlx-community/Mistral-7B-Instruct-v0.3-4bit

# Embedding Configuration
EMBEDDING_BINDING=local
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2

# RAG Configuration
GRAPH_ENABLED=False
DEFAULT_QUERY_MODE=naive
TOP_K=60
MAX_TOKEN_TEXT_CHUNK=4000
```

## 🚀 Usage

### 1. Start LightRAG Server
```bash
python -m lightrag.api.lightrag_server --port 8001
```

### 2. Access Web Interface
Open your browser and navigate to:
```
http://localhost:8001
```

### 3. Upload Documents
- Use the "Documents" tab to upload PDF, TXT, or other supported files
- Documents will be automatically processed and chunked

### 4. Query Documents
- Use the "Retrieval" tab to ask questions about your uploaded documents
- Responses will be generated based on document content using the local MLX model

### API Usage
```bash
# Query via API
curl -X POST "http://localhost:8001/query/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is classical music?",
    "mode": "naive",
    "stream": false
  }'
```

## 🔧 Key Modifications Made

### 1. Local Embedding Support
**File**: `lightrag/llm/local_embedder.py`
- Added sentence-transformers based embedding function
- Eliminates dependency on OpenAI embeddings API

### 2. API Configuration Updates
**Files**: `lightrag/api/config.py`, `lightrag/api/lightrag_server.py`
- Added "local" embedding option
- Updated configuration to support local embeddings

### 3. MLX Server Compatibility
**File**: `lightrag/llm/openai.py`
- Changed from `/v1/completions` to `/v1/chat/completions` endpoint
- Fixed message format for MLX/Mistral compatibility
- Updated API key handling for local inference

### 4. Query Parameter Bug Fix
**File**: `lightrag/api/routers/query_routes.py`
- Fixed `StreamQueryRequest.to_query_params()` method
- Prevented None values from overriding default parameters
- Resolved "bad operand type for unary -: 'NoneType'" error

### 5. Import and Dependency Fixes
**File**: `lightrag/operate.py`
- Added missing imports for cache handling functions
- Fixed parameter validation and error handling

## ⚙️ Configuration Options

### Query Parameters (Web UI)
- **Query Mode**: Set to "Naive" for vector-based retrieval
- **Top K Results**: Number of document chunks to retrieve (default: 60)
- **Max Tokens for Text Unit**: Maximum tokens per chunk (default: 4000)
- **Response Format**: "Multiple Paragraphs", "Single Paragraph", etc.

### Environment Variables
```env
# Core LLM Settings
LLM_BINDING=openai                                    # Use OpenAI-compatible API
LLM_BINDING_HOST=http://localhost:8000/v1            # MLX server endpoint
LLM_MODEL_NAME=mlx-community/Mistral-7B-Instruct-v0.3-4bit

# Embedding Settings
EMBEDDING_BINDING=local                               # Use local embeddings
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2                # Sentence-transformers model

# Performance Settings
TOP_K=60                                             # Default retrieval count
MAX_TOKEN_TEXT_CHUNK=4000                            # Max tokens per chunk
GRAPH_ENABLED=False                                  # Disable knowledge graph
DEFAULT_QUERY_MODE=naive                             # Use naive retrieval
```

## 🏗 Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Browser   │    │  LightRAG API   │    │   MLX Server    │
│   Port 8001     │◄──►│   Port 8001     │◄──►│   Port 8000     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │ Local Embeddings│
                    │ (sentence-trans)│
                    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │ Vector Database │
                    │  (NanoVectorDB) │
                    └─────────────────┘
```

## 🐛 Troubleshooting

### MLX Server Not Responding
```bash
# Check if MLX server is running
curl http://localhost:8000/v1/models

# Restart MLX server
uv run mlx_lm.server --model mlx-community/Mistral-7B-Instruct-v0.3-4bit --port 8000
```

### Vector Search Errors
- Ensure `TOP_K` environment variable is set
- Check that documents have been uploaded and processed
- Verify embedding model is loaded correctly

### Generic Responses
- Check that documents are properly chunked in the vector database
- Verify MLX server is receiving proper context in requests
- Ensure `mode="naive"` is set for vector-based retrieval

## 📊 Performance

### Tested Configuration
- **Hardware**: Apple Silicon (M1/M2/M3)
- **Model**: Mistral-7B-Instruct-v0.3-4bit (MLX optimized)
- **Embedding Model**: all-MiniLM-L6-v2
- **Performance**: ~20-25 tokens/second generation speed

### Memory Usage
- **MLX Model**: ~4-6GB RAM
- **Embedding Model**: ~100-200MB RAM
- **LightRAG Server**: ~200-500MB RAM

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with MLX integration
5. Submit a pull request

## 📄 License

This project maintains the same license as the original LightRAG project.

## 🙏 Acknowledgments

- [LightRAG](https://github.com/HKUDS/LightRAG) - Original RAG framework
- [MLX](https://github.com/ml-explore/mlx) - Apple's machine learning framework
- [sentence-transformers](https://github.com/UKPLab/sentence-transformers) - Local embedding models

## 📚 Related Projects

- [Original LightRAG](https://github.com/HKUDS/LightRAG)
- [MLX Examples](https://github.com/ml-explore/mlx-examples)
- [MLX Community Models](https://huggingface.co/mlx-community) 