import os
import logging
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import chromadb
from chromadb.utils import embedding_functions
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import httpx 

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="RAG Chatbot", description="A chatbot with ChromaDB RAG backend and verification")

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Pydantic models
class ChatMessage(BaseModel):
    message: str
    model: Optional[str] = None
    
class ChatResponse(BaseModel):
    response: str
    sources: Optional[List[str]] = None
    verification_passed: Optional[bool] = None
    generation_attempts: Optional[int] = None

class DocumentInput(BaseModel):
    content: str
    metadata: Optional[dict] = None

class ChromaDBEmbeddingWrapper:
    """Wrapper to make ChromaDB embedding functions compatible with LangChain interface"""
    
    def __init__(self, chromadb_embedding_function):
        self.chromadb_embedding_function = chromadb_embedding_function
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text"""
        # ChromaDB embedding functions expect a list and return a list of embeddings
        embeddings = self.chromadb_embedding_function([text])
        return embeddings[0]
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        return self.chromadb_embedding_function(texts)

class RAGChatbot:
    def __init__(self):
        # Environment variables
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.model_name = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
        self.chroma_host = os.getenv("CHROMA_HOST", "localhost")
        self.chroma_port = int(os.getenv("CHROMA_PORT", "8000"))
        self.collection_name = os.getenv("COLLECTION_NAME", "documents")
        
        # Verification settings
        self.enable_verification = os.getenv("ENABLE_VERIFICATION", "true").lower() == "true"
        self.verification_base_url = os.getenv("VERIFICATION_BASE_URL", "http://localhost:11434/v1")
        self.verification_model = os.getenv("VERIFICATION_MODEL", "bespoke-minicheck:latest")
        self.verification_api_key = os.getenv("VERIFICATION_API_KEY", "dummy")
        self.max_regeneration_attempts = int(os.getenv("MAX_REGENERATION_ATTEMPTS", "3"))
        
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.HttpClient(
            host=self.chroma_host,
            port=self.chroma_port
        )
        
        # Initialize ChromaDB embeddings
        self.chromadb_embeddings = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Create LangChain-compatible wrapper
        self.embeddings = ChromaDBEmbeddingWrapper(self.chromadb_embeddings)
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model_name=self.model_name,
            openai_api_key=self.openai_api_key,
            openai_api_base=self.openai_base_url,
            temperature=1
        )
        
        # Initialize verification LLM if enabled
        if self.enable_verification:
            self.verification_llm = ChatOpenAI(
                model_name=self.verification_model,
                openai_api_key=self.verification_api_key,
                openai_api_base=self.verification_base_url,
                temperature=0.0  # Use deterministic output for verification
            )
            logger.info(f"Verification enabled with model: {self.verification_model}")
        else:
            self.verification_llm = None
            logger.info("Verification disabled")
        
        # Initialize or get collection (don't create if it doesn't exist)
        try:
            self.collection = self.chroma_client.get_collection(name=self.collection_name, embedding_function=self.chromadb_embeddings)
            logger.info(f"Connected to existing collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Collection '{self.collection_name}' not found: {str(e)}")
            raise ValueError(f"Collection '{self.collection_name}' does not exist. Please ensure the collection is created by your external management software before starting the chatbot.")
        
        # Initialize vector store
        self.vector_store = Chroma(
            client=self.chroma_client,
            collection_name=self.collection_name,
            embedding_function=self.embeddings  # Use the LangChain-compatible wrapper
        )
        
        # Initialize retrieval chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 4}),
            return_source_documents=True
        )

        self.qa_chain = (
            {
                "context": self.vector_store.as_retriever(),
                "question": RunnablePassthrough(),
            }
            | self.llm
            | StrOutputParser()
        )
        
        logger.info("RAG Chatbot initialized successfully")
    
    def verify_response(self, claim: str, documents: List[Document]) -> bool:
        """Verify the generated response against source documents using bespoke-minicheck"""
        if not self.enable_verification or not self.verification_llm:
            return True
        
        try:
            # Combine all document content for verification
            combined_document = "\n\n".join([doc.page_content for doc in documents])
            
            # Create verification prompt
            verification_prompt = f"""Document: {combined_document}
Claim: {claim}"""
            
            # Get verification result
            verification_result = self.verification_llm.predict(verification_prompt)
            
            # Check if the response is "Yes" (case-insensitive)
            is_verified = verification_result.strip().lower().startswith("yes")
            
            logger.info(f"Verification result: {verification_result.strip()} -> {'PASSED' if is_verified else 'FAILED'}")
            return is_verified
            
        except Exception as e:
            logger.error(f"Error during verification: {str(e)}")
            # If verification fails, assume the response is valid to avoid blocking
            return True
    
    def add_documents(self, documents: List[DocumentInput]) -> bool:
        """Add documents to the ChromaDB collection"""
        try:
            docs = []
            for doc_input in documents:
                # Split text into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                chunks = text_splitter.split_text(doc_input.content)
                
                for chunk in chunks:
                    docs.append(Document(
                        page_content=chunk,
                        metadata=doc_input.metadata or {}
                    ))
            
            # Add documents to vector store
            self.vector_store.add_documents(docs)
            logger.info(f"Added {len(docs)} document chunks to the collection")
            return True
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            return False
    
    def chat(self, message: str, model: Optional[str] = None) -> ChatResponse:
        """Process a chat message and return response with sources, including verification"""
        try:
            attempts = 0
            verification_passed = True
            # If a model is specified, temporarily use it for this request
            orig_llm = self.llm
            if model and model != self.model_name:
                temp_llm = ChatOpenAI(
                    model_name=model,
                    openai_api_key=self.openai_api_key,
                    openai_api_base=self.openai_base_url,
                    temperature=1
                )
                qa_chain = RetrievalQA.from_chain_type(
                    llm=temp_llm,
                    chain_type="stuff",
                    retriever=self.vector_store.as_retriever(search_kwargs={"k": 4}),
                    return_source_documents=True
                )
            else:
                qa_chain = self.qa_chain
            while attempts < self.max_regeneration_attempts:
                attempts += 1
                # Generate response
                result = qa_chain({"query": message})
                response = result["result"]
                source_documents = result.get("source_documents", [])
                # Prepare sources for response
                sources = []
                for doc in source_documents:
                    if doc.metadata:
                        sources.append(str(doc.metadata))
                    else:
                        sources.append(doc.page_content[:100] + "...")
                # Verify response if enabled
                if self.enable_verification and source_documents:
                    verification_passed = self.verify_response(response, source_documents)
                    if verification_passed:
                        logger.info(f"Response verified successfully on attempt {attempts}")
                        break
                    else:
                        logger.warning(f"Verification failed on attempt {attempts}, regenerating...")
                        if attempts >= self.max_regeneration_attempts:
                            logger.warning(f"Max regeneration attempts ({self.max_regeneration_attempts}) reached")
                            break
                else:
                    # No verification needed or no source documents
                    break
            return ChatResponse(
                response=response, 
                sources=sources,
                verification_passed=verification_passed if self.enable_verification else None,
                generation_attempts=attempts if self.enable_verification else None
            )
        except Exception as e:
            logger.error(f"Error processing chat message: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")

# Initialize chatbot
try:
    chatbot = RAGChatbot()
except Exception as e:
    logger.error(f"Failed to initialize chatbot: {str(e)}")
    chatbot = None

@app.get("/")
async def serve_index():
    """Serve the main web UI"""
    return FileResponse('app/static/index.html')

@app.get("/health")
async def health_check():
    if chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    return {
        "status": "healthy", 
        "chroma_host": chatbot.chroma_host, 
        "model": chatbot.model_name,
        "verification_enabled": chatbot.enable_verification,
        "verification_model": chatbot.verification_model if chatbot.enable_verification else None
    }

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(message: ChatMessage):
    if chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    return chatbot.chat(message.message, model=message.model)

@app.post("/documents")
async def add_documents_endpoint(documents: List[DocumentInput]):
    if chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    success = chatbot.add_documents(documents)
    if success:
        return {"message": f"Successfully added {len(documents)} documents"}
    else:
        raise HTTPException(status_code=500, detail="Failed to add documents")

@app.get("/collection/count")
async def get_collection_count():
    if chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    try:
        count = chatbot.collection.count()
        return {"count": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting collection count: {str(e)}")

@app.get("/verification/status")
async def get_verification_status():
    if chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    return {
        "verification_enabled": chatbot.enable_verification,
        "verification_model": chatbot.verification_model if chatbot.enable_verification else None,
        "verification_base_url": chatbot.verification_base_url if chatbot.enable_verification else None,
        "max_regeneration_attempts": chatbot.max_regeneration_attempts
    }

@app.get("/models")
async def list_models():
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    headers = {"Authorization": f"Bearer {openai_api_key}"}
    url = f"{openai_base_url.rstrip('/')}/models"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Error fetching models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching models: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080) 