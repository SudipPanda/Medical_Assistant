import requests
from langchain_community.tools.tavily_search import TavilySearchResults
import logging

class Travily_search:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        
    def search_travily(self , query):
         
        tavily_search = TavilySearchResults(max_results = 5)

        try:
            #strip of any quotes
            query = query.strip('"\'')
            search_docs = tavily_search.invoke(query)
            if len(search_docs):
                return "\n".join(["title: " + str(res["title"]) + " - " + 
                                  "url: " + str(res["url"]) + " - " + 
                                  "content: " + str(res["content"]) + " - " + 
                                  "score: " + str(res["score"]) for res in search_docs])
            return "No relevant results found."
        
        except Exception as E:
            self.logger.error(f"Travily search failed here with exception {E}")

         
