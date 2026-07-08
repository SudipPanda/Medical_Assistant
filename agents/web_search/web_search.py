import requests
from pubmed_search import pubmed_search
from travily_search import Travily_search

class web_search:
    def __init__(self):
        self.travily_search  = Travily_search()
        
    def web_search(self , query):
        travily_result = self.travily_search.search_travily(query)

        return f"Tavily Results:\n{travily_result}\n"

