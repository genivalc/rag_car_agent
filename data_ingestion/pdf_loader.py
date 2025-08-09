import os
from typing import List
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from .chunker import SemanticChunker
from config import config

class PDFProcessor:
    def __init__(self):
        self.chunker = SemanticChunker(
            min_chunk_size=config.MIN_CHUNK_SIZE,
            max_chunk_size=config.MAX_CHUNK_SIZE,
            overlap=config.CHUNK_OVERLAP
        )
    
    def load_and_process(self, pdf_path: str) -> List[Document]:
        """Loads PDF and applies semantic chunking"""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        print(f"📄 Loading PDF: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        print(f"✂️ Applying semantic chunking")
        chunks = self.chunker.chunk_documents(documents)
        
        print(f"📊 Generated {len(chunks)} semantic chunks")
        return chunks