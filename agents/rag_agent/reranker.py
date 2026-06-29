import os
import re
import logging
from pathlib import Path
from typing import List , Dict , Optional , Union , Any
from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self , config):
        self.logger = logging.getLogger(__name__)

        try:
            self.model_name = config.rag.reranker_model
            self.logger.info(f"Loading reranker model: {self.model_name}")
            self.model = CrossEncoder(self.model_name)
            self.top_k = config.rag.reranker_top_k
        except Exception as e:
            self.logger.error("an error has occure while loading the cofig in reranker")
    
    def re_ranking(self , query:str,  documents: Union[List[Dict[str, Any]], List[str]], parsed_content_dir: str):
        try:
            if not documents:
                return []
            
        
        except Exception as e:
            pass

