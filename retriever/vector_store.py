import os
from typing import List
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from config import config

class VectorStoreManager:
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(model=config.EMBEDDING_MODEL)
        self.vectorstore = None
    
    def create_or_load(self, documents: List[Document], force_rebuild: bool = False) -> FAISS:
        """Creates or loads existing vectorstore"""
        if os.path.exists(config.VECTORSTORE_DIR) and not force_rebuild:
            print(f"🔁 Loading existing vectorstore")
            self.vectorstore = FAISS.load_local(
                config.VECTORSTORE_DIR, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
        else:
            print(f"🔎 Creating new vectorstore")
            self.vectorstore = FAISS.from_documents(documents, self.embeddings)
            self._save_vectorstore()
        
        return self.vectorstore
    
    def _save_vectorstore(self):
        """Saves vectorstore to disk"""
        os.makedirs(os.path.dirname(config.VECTORSTORE_DIR), exist_ok=True)
        self.vectorstore.save_local(config.VECTORSTORE_DIR)
        print(f"💾 Vectorstore salvo em {config.VECTORSTORE_DIR}")
    
    def get_vectorstore(self) -> FAISS:
        """Returns current vectorstore"""
        if self.vectorstore is None:
            raise ValueError("Vectorstore not initialized")
        return self.vectorstore