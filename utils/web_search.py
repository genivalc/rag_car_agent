from duckduckgo_search import DDGS
from typing import List, Dict
import time
import requests

class WebSearchAgent:
    def __init__(self):
        self.ddgs = DDGS()
    
    def search_automotive(self, query: str, max_results: int = 3) -> List[Dict]:
        """Searches automotive information on the web"""
        automotive_query = f"{query} manual car automotive maintenance"
        
        try:
            time.sleep(1)
            results = list(self.ddgs.text(
                automotive_query, 
                max_results=max_results,
                region='br-pt'
            ))
            return results
        except Exception as e:
            print(f"Web search unavailable: {e}")
            # Fallback: generic response
            return [{
                'title': 'Information not found',
                'body': 'Could not search for additional information on the web at the moment. Please try again later.'
            }]
    
    def format_web_results(self, results: List[Dict]) -> str:
        """Formats web results"""
        if not results:
            return ""
        
        formatted = "WEB INFORMATION:\n"
        for i, result in enumerate(results, 1):
            formatted += f"[Web {i}] {result.get('title', '')}\n"
            formatted += f"{result.get('body', '')}\n\n"
        
        return formatted
