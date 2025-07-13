# RAG Chatbot with ChromaDB, Langchain and Verification

A dockerized Python-based chatbot that uses ChromaDB as a RAG (Retrieval-Augmented Generation) backend, Langchain for LLM integration, and bespoke-minicheck for response verification. The system supports OpenAI-compatible APIs and provides both a REST API and a modern web UI for interaction.

## Features

- 🤖 **RAG-powered chatbot** using ChromaDB vector database
- 🔗 **Langchain integration** for seamless LLM interaction
- ✅ **Response verification** using bespoke-minicheck model
- 🔄 **Automatic regeneration** when verification fails
- 🌐 **OpenAI-compatible API support** (OpenAI, Azure OpenAI, local models like Ollama)
- 🔍 **Web search integration** using Tavily Search API
- 🐳 **Fully dockerized** with docker-compose orchestration
- 🌟 **Modern Web UI** with real-time chat and system monitoring
- 📄 **Document ingestion** with automatic text chunking and embedding
- 🔍 **Source attribution** in responses
- 🏥 **Health checks** and monitoring
- 📊 **FastAPI** with automatic OpenAPI documentation

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User/Client   │    │   RAG Chatbot   │    │    ChromaDB     │
│    (Web UI)     │───▶│   (FastAPI)     │───▶│  Vector Store   │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │                       │
                              ▼                       ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │ OpenAI-Compatible│    │ bespoke-minicheck│
                       │   LLM Provider   │    │   Verification   │
                       └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │   Tavily Search │
                       │   API (Web)     │
                       └─────────────────┘
```

## Quick Start

### Prerequisites

- Docker and Docker Compose
- OpenAI API key (or compatible API)
- Ollama with bespoke-minicheck model (for verification)
- **Pre-existing ChromaDB collection** (managed externally)

### Setup

1. **Set up bespoke-minicheck model (for verification):**
   ```bash
   # Install Ollama if not already installed
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Pull the bespoke-minicheck model
   ollama pull bespoke-minicheck:latest
   ```

2. **Clone or create the project structure:**
   ```bash
   mkdir rag-chatbot && cd rag-chatbot
   # Copy all the files from this project
   ```

3. **Set up environment variables:**
   ```bash
   cp env.example .env
   # Edit .env with your configuration
   ```

4. **Configure your environment:**
   Edit the `.env` file with your OpenAI API key:
   ```env
   OPENAI_API_KEY=your_actual_api_key_here
   OPENAI_BASE_URL=https://api.openai.com/v1
   MODEL_NAME=gpt-3.5-turbo
   ENABLE_VERIFICATION=true
   ```

5. **Start the services:**
   ```bash
   docker-compose up --build
   ```

6. **Set up demo data (optional):**
   ```bash
   python demo_setup.py
   ```

7. **Access the application:**
   - **Web UI**: `http://localhost:8080`
   - **API Documentation**: `http://localhost:8080/docs`

## Web UI Features

### 🎯 **Main Interface**
- **Clean, responsive design** with Bootstrap 5
- **Real-time chat interface** with typing indicators
- **System status monitoring** in the sidebar
- **Document management** with easy upload form

### 💬 **Chat Features**
- **Interactive chat bubbles** with user/bot distinction
- **Verification status display** (✅ passed / ⚠️ failed)
- **Source attribution** showing document sources
- **Generation attempt tracking** for transparency
- **Real-time typing indicators** during response generation
- **Message timestamps** and metadata
- **Clear chat functionality**

### 📊 **System Monitoring**
- **Health status indicators** (online/offline)
- **Verification status** (enabled/disabled)
- **Document count** in knowledge base
- **Auto-refresh status** every 30 seconds
- **Manual refresh button** for immediate updates

### 📄 **Document Management**
- **Easy document upload** with content, source, and topic fields
- **Real-time document count** updates
- **Success/error notifications** with toast messages
- **Form validation** and user feedback

### 📱 **Mobile Responsive**
- **Optimized for all screen sizes**
- **Touch-friendly interface**
- **Responsive layout** that adapts to mobile devices

## Environment Variables

### Required Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `OPENAI_API_KEY` | Your OpenAI API key or compatible API key | `sk-...` |

### Optional Variables

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `OPENAI_BASE_URL` | Base URL for OpenAI-compatible API | `https://api.openai.com/v1` | `http://localhost:11434/v1` (Ollama) |
| `MODEL_NAME` | Model to use for chat completion | `gpt-3.5-turbo` | `gpt-4`, `llama2` |
| `COLLECTION_NAME` | ChromaDB collection name | `documents` | `my_knowledge_base` |
| `LOG_LEVEL` | Logging level | `INFO` | `DEBUG`, `WARNING`, `ERROR` |

### Verification Variables

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `ENABLE_VERIFICATION` | Enable/disable response verification | `true` | `false` |
| `VERIFICATION_BASE_URL` | Base URL for verification model API | `http://localhost:11434/v1` | `http://localhost:8080/v1` |
| `VERIFICATION_MODEL` | Model to use for verification | `bespoke-minicheck:latest` | `bespoke-minicheck:v1` |
| `VERIFICATION_API_KEY` | API key for verification model | `dummy` | `your_api_key` |
| `MAX_REGENERATION_ATTEMPTS` | Max attempts to regenerate response | `3` | `5` |

### Tavily Search Variables

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `ENABLE_TAVILY_SEARCH` | Enable/disable web search integration | `false` | `true` |
| `TAVILY_API_KEY` | Your Tavily Search API key | `None` | `tvly-...` |
| `TAVILY_SEARCH_DEPTH` | Search depth: `basic` or `advanced` | `basic` | `advanced` |
| `TAVILY_MAX_RESULTS` | Maximum number of search results | `5` | `10` |
| `TAVILY_INCLUDE_DOMAINS` | Comma-separated domains to include | `None` | `example.com,wikipedia.org` |
| `TAVILY_EXCLUDE_DOMAINS` | Comma-separated domains to exclude | `None` | `spam.com,ads.com` |

### OpenAI-Compatible APIs

The chatbot supports various OpenAI-compatible APIs:

- **OpenAI**: `https://api.openai.com/v1`
- **Azure OpenAI**: `https://your-resource.openai.azure.com/`
- **Ollama**: `http://localhost:11434/v1`
- **LocalAI**: `http://localhost:8080/v1`
- **Anyscale**: `https://api.endpoints.anyscale.com/v1`

## Response Verification

The system uses bespoke-minicheck to verify that generated responses are consistent with the retrieved documents. Here's how it works:

1. **Generate Response**: The RAG system generates a response based on retrieved documents
2. **Verification Check**: The response is verified against source documents using this prompt:
   ```
   Document: {combined_documents}
   Claim: {generated_response}
   ```
3. **Regeneration**: If verification fails, the system regenerates the response (up to max attempts)
4. **Final Response**: Returns the response with verification status and attempt count

### Verification Benefits

- **Accuracy**: Ensures responses are grounded in source documents
- **Reliability**: Reduces hallucinations and incorrect information
- **Transparency**: Shows verification status in response metadata
- **Configurable**: Can be disabled or customized per use case

## Web Search Integration

The system integrates Tavily Search API through LangChain's TavilySearchResults tool to enhance responses with real-time web information. Here's how it works:

1. **Query Processing**: When a user asks a question, the system searches both the knowledge base and the web
2. **Web Search**: Uses LangChain's TavilySearchResults tool to find relevant web content
3. **Context Combination**: Combines knowledge base results with web search results
4. **Response Generation**: Generates a response using both sources of information
5. **Source Attribution**: Clearly indicates which sources are from the knowledge base vs web search

### Web Search Benefits

- **Real-time Information**: Access to current events and latest information
- **Comprehensive Coverage**: Combines internal knowledge with external web data
- **Source Transparency**: Clear distinction between knowledge base and web sources
- **Configurable Search**: Control search depth, result count, and domain filtering
- **Optional Integration**: Can be enabled/disabled via environment variables
- **LangChain Integration**: Uses LangChain's TavilySearchResults tool for seamless integration

### Search Configuration

- **Search Depth**: Choose between `basic` (faster) or `advanced` (more comprehensive)
- **Result Count**: Control how many web results to include (default: 5)
- **Domain Filtering**: Include or exclude specific domains for targeted results
- **API Key Management**: Secure API key configuration via environment variables

### LangChain Integration

The system uses LangChain's `TavilySearchResults` tool for web search integration:

```python
from langchain_community.tools import TavilySearchResults

# Initialize the search tool
tavily_tool = TavilySearchResults(
    api_key="your_tavily_api_key",
    max_results=5,
    search_depth="basic",
    include_domains=["example.com"],
    exclude_domains=["spam.com"]
)

# Use the tool
results = tavily_tool.invoke("your search query")
```

This integration provides:
- **Seamless LangChain compatibility** with existing RAG pipeline
- **Automatic result formatting** as LangChain Documents
- **Consistent API** with other LangChain tools
- **Easy configuration** through environment variables

## API Endpoints

### Base URL: `http://localhost:8080`

#### Web UI
```http
GET /
```
Serves the main web UI interface.

#### Health Check
```http
GET /health
```
Returns the health status of the chatbot, ChromaDB connection, and verification settings.

#### Chat
```http
POST /chat
```
Send a message to the chatbot with verification.

**Request Body:**
```json
{
  "message": "What is Python?"
}
```

**Response:**
```json
{
  "response": "Python is a high-level programming language...",
  "sources": ["Python Documentation: Python Programming"],
  "verification_passed": true,
  "generation_attempts": 1
}
```

#### Add Documents
```http
POST /documents
```
Add documents to the knowledge base.

**Request Body:**
```json
[
  {
    "content": "Your document content here...",
    "metadata": {
      "source": "Document Source",
      "topic": "Document Topic"
    }
  }
]
```

#### Collection Count
```http
GET /collection/count
```
Get the number of documents in the ChromaDB collection.

#### Verification Status
```http
GET /verification/status
```
Get the current verification configuration.

**Response:**
```json
{
  "verification_enabled": true,
  "verification_model": "bespoke-minicheck:latest",
  "verification_base_url": "http://localhost:11434/v1",
  "max_regeneration_attempts": 3
}
```

#### Tavily Search Status
```http
GET /tavily/status
```
Get the current Tavily search configuration.

**Response:**
```json
{
  "tavily_search_enabled": true,
  "tavily_search_depth": "basic",
  "tavily_max_results": 5,
  "tavily_include_domains": ["example.com", "wikipedia.org"],
  "tavily_exclude_domains": ["spam.com"]
}
```

### Interactive API Documentation

Once the service is running, visit:
- **Swagger UI**: `http://localhost:8080/docs`
- **ReDoc**: `http://localhost:8080/redoc`

## Usage Examples

### Using the Web UI

1. **Open your browser** and navigate to `http://localhost:8080`
2. **Check system status** in the sidebar (should show green indicators)
3. **Add documents** using the form in the sidebar
4. **Start chatting** by typing questions in the chat interface
5. **View verification status** for each response (✅ for passed, ⚠️ for failed)
6. **Check sources** displayed below bot responses

### Using curl

**Health check:**
```bash
curl http://localhost:8080/health
```

**Chat with verification:**
```bash
curl -X POST "http://localhost:8080/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "What is machine learning?"}'
```

**Check verification status:**
```bash
curl http://localhost:8080/verification/status
```

### Using Python

```python
import requests

# Chat with the bot
response = requests.post("http://localhost:8080/chat", 
                        json={"message": "Explain RAG"})
result = response.json()

print(f"Response: {result['response']}")
print(f"Verification: {'✅ PASSED' if result['verification_passed'] else '❌ FAILED'}")
print(f"Attempts: {result['generation_attempts']}")

# Check verification status
status = requests.get("http://localhost:8080/verification/status").json()
print(f"Verification enabled: {status['verification_enabled']}")
```

## Development

### Local Development

For local development without Docker:

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start Ollama with bespoke-minicheck:**
   ```bash
   ollama serve
   ollama pull bespoke-minicheck:latest
   ```

3. **Start ChromaDB:**
   ```bash
   docker run -p 8000:8000 chromadb/chroma:latest
   ```

4. **Set environment variables:**
   ```bash
   export OPENAI_API_KEY=your_key_here
   export CHROMA_HOST=localhost
   export ENABLE_VERIFICATION=true
   ```

5. **Run the application:**
   ```bash
   cd app && python -m uvicorn main:app --reload --host 0.0.0.0 --port 8080
   ```

### Demo Setup

Use the demo setup script to populate the knowledge base with sample documents:

```bash
python demo_setup.py
```

This script will:
- Wait for the service to be ready
- Add comprehensive sample documents about Python, ML, FastAPI, ChromaDB, RAG, LangChain, and Docker
- Display system status
- Provide suggested questions to try

### Configuring Verification

#### Disable Verification
```env
ENABLE_VERIFICATION=false
```

#### Use Different Verification Model
```env
VERIFICATION_MODEL=your-custom-model:latest
VERIFICATION_BASE_URL=http://your-model-endpoint/v1
```

#### Adjust Regeneration Attempts
```env
MAX_REGENERATION_ATTEMPTS=5  # Allow more attempts
```

### Customizing the RAG Pipeline

The RAG configuration can be modified in `app/main.py`:

- **Chunk size**: Modify `chunk_size` in `RecursiveCharacterTextSplitter`
- **Retrieval count**: Change `k` in `search_kwargs`
- **Temperature**: Adjust `temperature` in `ChatOpenAI`
- **Verification prompt**: Modify the prompt template in `verify_response`

### Customizing the Web UI

The web UI can be customized by modifying:

- **`app/static/style.css`**: Visual styling and themes
- **`app/static/script.js`**: Interactive functionality and API calls
- **`app/static/index.html`**: Layout and structure

## Monitoring and Logs

### View logs:
```bash
docker-compose logs -f chatbot
```

### Health monitoring:
The services include health checks. Check status with:
```bash
docker-compose ps
```

### Verification Monitoring:
Monitor verification performance in logs:
```bash
docker-compose logs -f chatbot | grep -i verification
```

## Troubleshooting

### Common Issues

1. **Service not starting:**
   - Check if ports 8080 and 8000 are available
   - Verify your OpenAI API key is valid
   - Check logs: `docker-compose logs`

2. **Web UI not loading:**
   - Ensure the service is running on port 8080
   - Check browser console for JavaScript errors
   - Verify static files are properly served

3. **Verification not working:**
   - Ensure Ollama is running and has bespoke-minicheck model
   - Check `VERIFICATION_BASE_URL` points to correct endpoint
   - Verify model name matches: `ollama list`

4. **No responses from chatbot:**
   - Ensure documents are added to the collection
   - Check ChromaDB connection
   - Verify API key and model name

5. **Verification always failing:**
   - Check if verification model is responding correctly
   - Test verification endpoint separately
   - Consider disabling verification temporarily

### Environment-Specific Configurations

**For Ollama with bespoke-minicheck:**
```env
ENABLE_VERIFICATION=true
VERIFICATION_BASE_URL=http://host.docker.internal:11434/v1
VERIFICATION_MODEL=bespoke-minicheck:latest
VERIFICATION_API_KEY=dummy
```

**For Azure OpenAI:**
```env
OPENAI_BASE_URL=https://your-resource.openai.azure.com/
MODEL_NAME=gpt-35-turbo
OPENAI_API_KEY=your_azure_key
```

**Disable verification for testing:**
```env
ENABLE_VERIFICATION=false
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the logs using `docker-compose logs`
3. Test verification separately: `curl http://localhost:11434/v1/chat/completions`
4. Open an issue in the repository 

## ChromaDB Collection Management

⚠️ **Important**: This chatbot does **NOT** automatically create ChromaDB collections. Collections must be created and managed by external software.

### Collection Requirements

The chatbot expects a pre-existing ChromaDB collection with the name specified in the `COLLECTION_NAME` environment variable (default: "documents").

### Creating a Collection Externally

If you need to create a collection manually for testing, you can use this Python script:

```python
import chromadb

# Connect to ChromaDB
client = chromadb.HttpClient(host="localhost", port=8000)

# Create collection
collection_name = "documents"  # or your custom name
collection = client.create_collection(name=collection_name)

print(f"Created collection: {collection_name}")
```

### Error Handling

If the specified collection doesn't exist, the chatbot will:
1. Log an error message
2. Raise a `ValueError` with instructions
3. Fail to start

This ensures that the chatbot only operates on collections managed by your external systems.

## Web UI Features 