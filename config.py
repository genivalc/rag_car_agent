import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Config:
    # Paths
    PDF_PATH: str = "./data/argo_2023.pdf"
    VECTORSTORE_DIR: str = "./vectorstore/argo"
    
    # Chunking
    MIN_CHUNK_SIZE: int = 200
    MAX_CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 100
    
    # Retrieval
    RETRIEVAL_K: int = 5
    SIMILARITY_THRESHOLD: float = 0.5 
    
    # Models
    EMBEDDING_MODEL: str = "models/embedding-001"
    LLM_MODEL: str = "gemini-2.0-flash-lite"
    LLM_TEMPERATURE: float = 0.3
    
    # API
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY")

config = Config()