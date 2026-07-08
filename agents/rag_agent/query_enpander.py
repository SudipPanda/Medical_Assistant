import logging
from typing import List , Dict , Any

class query_expand:
    def __init__(self , config):
        self.logger = logging.getLogger(f"{self.__module__}")
        self.config = config
        self.model = config.rag.llm
    
    def expand_query(self , original_query)->Dict[str , Any]:
        self.logger.info(f"Expanding query: {original_query}")

        expand_query = self._generate_expansion(original_query)

        return{
            'original_query':original_query , 
            'expand_query':expand_query
        }

    

    def _generate_expansion(self , query:str)->str:
        prompt = f"""
        As a medical expert, expand the following query with relevant medical terminology, 
        synonyms, and related concepts that would help in retrieving relevant medical information:
        
        User Query: {query}
        
        Expand the query only if you feel like it is required, otherwise keep the user query intact.
        Be specific to the medical or any other domain mentioned in the ueer query, do not add other medical domains.
        If the user query asks about answering in tabular format, include that in the expanded query and do not answer in tabular format yourself.
        Provide only the expanded query without explanations.
        """
        expand_query  = self.model.invoke(query)
        return expand_query
