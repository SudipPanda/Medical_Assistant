import requests
import logging

class pubmed_search:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def search_pubmed(self , pubmed_url , query:str):
        params = {
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": 5
        }
        
        try:
            response = requests.get(pubmed_url , params=params)
            data = response.json()
            article_ids = data.get("esearchresult", {}).get("idlist", [])
            if not article_ids:
                return "No relevant PubMed articles found."
            
            article_links = [f"https://pubmed.ncbi.nlm.nih.gov/{article_id}/" for article_id in article_ids]
            
            return "\n".join(article_links)
        
        except Exception as e:
            self.logger.error(f"problem in pubmed search here {e}")