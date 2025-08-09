from typing import Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from retriever.retriever_agent import RetrieverAgent
from utils.web_search import WebSearchAgent  # ADICIONAR ESTA LINHA
from config import config

class GeneratorAgent:
    def __init__(self, retriever: RetrieverAgent):
        self.retriever = retriever 
        self.web_search = WebSearchAgent()
        self.llm = ChatGoogleGenerativeAI(
            model=config.LLM_MODEL,
            temperature=config.LLM_TEMPERATURE
        )
        self.prompt = self._create_prompt()
    
    def _create_prompt(self) -> ChatPromptTemplate:
        """Creates optimized prompt for automotive RAG"""
        template = """You are a highly qualified automotive specialist.

RETRIEVED CONTEXT:
{context}

INSTRUCTIONS:
1. Answer ONLY based on the context provided above
2. If information is not in the context, clearly state you don't have that information
3. Be precise, technical and detailed when appropriate
4. Cite specific parts of the context when relevant
5. Keep focus strictly automotive
6. ALWAYS respond in Portuguese (Brazil)

QUESTION: {question}

RESPONSE:"""
        
        return ChatPromptTemplate.from_template(template)
    
    def generate_response(self, query: str) -> Dict[str, Any]:
        """Generates response based on retrieved context"""
        # Retrieve relevant context
        context = self.retriever.get_context_string(query)
        
        if not context.strip():
            print("🌐 Searching web for information...")
            web_results = self.web_search.search_automotive(query)
            web_context = self.web_search.format_web_results(web_results)
            
            if web_context:
                context = web_context
                sources = [f"Web {i+1}" for i in range(len(web_results))]
                
                response = self.llm.invoke(
                    self.prompt.format_messages(context=context, question=query)
                )
                
                return {
                    "answer": response.content,
                    "context_used": True,
                    "sources": sources,
                    "search_type": "web"
                }
        
        # Generate response using LLM
        print("🤖 Generating response...")
        response = self.llm.invoke(
            self.prompt.format_messages(context=context, question=query)
        )
        
        # Extract used sources
        sources = self._extract_sources(context)
        
        return {
            "answer": response.content,
            "context_used": True,
            "sources": sources,
            "context": context[:500] + "..." if len(context) > 500 else context
        }
    
    def _extract_sources(self, context: str) -> list:
        """Extracts information from used sources"""
        # Simples extração de fontes baseada nos documentos
        import re
        page_pattern = r'\[Página (\d+)\]'  # Mudança aqui
        matches = re.findall(page_pattern, context)
        return [f"Página {match}" for match in matches]