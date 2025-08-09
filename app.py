# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import logging
from typing import Dict, Any

from data_ingestion import PDFProcessor
from retriever import VectorStoreManager, RetrieverAgent
from generator import GeneratorAgent
from utils import setup_logging
from config import config

# Pydantic Models
class QuestionRequest(BaseModel):
    question: str

class QuestionResponse(BaseModel):
    answer: str
    sources: list[str]
    context_used: bool

# Global variables for agents
retriever_agent = None
generator_agent = None
logger = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages application lifecycle"""
    global retriever_agent, generator_agent, logger
    
    # Inicialização
    logger = setup_logging()
    logger.info("🚀 Starting RAG Car Agent API")
    
    try:
        # 1. Document processing
        pdf_processor = PDFProcessor()
        documents = pdf_processor.load_and_process(config.PDF_PATH)
        
        # 2. Vector store creation/loading
        vector_manager = VectorStoreManager()
        vectorstore = vector_manager.create_or_load(documents, force_rebuild=False)
        
        # 3. RAG agents initialization
        retriever_agent = RetrieverAgent(vectorstore)
        generator_agent = GeneratorAgent(retriever_agent)
        
        logger.info("✅ RAG system initialized successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Initialization error: {e}")
        raise
    
    # Cleanup (se necessário)
    logger.info("🔄 Shutting down application")

# Criação da aplicação FastAPI
app = FastAPI(
    title="RAG Car Agent API",
    description="RAG system specialized in automotive queries",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/")
async def root():
    """Endpoint de status"""
    return {"message": "🚗 RAG Car Agent API - Automotive Specialist"}

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest) -> QuestionResponse:
    """Processes question using RAG architecture"""
    global generator_agent, logger
    
    if generator_agent is None:
        raise HTTPException(status_code=503, detail="System not yet initialized")
    
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        logger.info(f"📝 Processing question: {request.question}")
        
        # Generate response using RAG architecture
        result = generator_agent.generate_response(request.question)
        
        logger.info(f"✅ Response generated with {len(result['sources'])} sources")
        
        return QuestionResponse(
            answer=result['answer'],
            sources=result['sources'],
            context_used=result['context_used']
        )
        
    except Exception as e:
        logger.error(f"❌ Error processing question: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
