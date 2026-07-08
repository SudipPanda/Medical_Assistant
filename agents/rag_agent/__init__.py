import os
from typing import List , Dict , Optional , Any
from .doc_perser import MedicalDoc
from .query_enpander import query_expand
from .reranker import Reranker
from .response_generator import ResponseGenerator
from .vector_store import vector_store
import logging
from time import time

class Medical_Rag:
    def __init__(self , config):
        self.logger = logging.getLogger()
        self.config = config
        self.doc_perser = MedicalDoc()
        self.query_expander = query_expand(config)
        self.reranker = Reranker(config)
        self.response_gen = ResponseGenerator(config)
        self.vector_store = vector_store(config)
        self.parsed_content_dir = self.config.rag.parsed_content_dir
    
    def ingest_file(self, document_path:str):
        
        self.start_time = time()
        try:
            #parse document here
            self.logger.info("1.parsing the document and getting image here")
            convo , image = self.doc_perser.parse_document(document_path , self.parsed_content_dir)

            #summarized image here
            self.logger.info("2.summerized the image here")

            #create a vector store here 
            self.vector_store.create_vectorStore(
                document_chunk=convo , 
                document_path=document_path
            )

            return {
                "success":True ,
                "chunk_process":len(convo),
                "time":self.start_time-time()
            }

        except Exception as e:
            self.logger.error(f"getting error here {e}")
            return {
                "success":False , 
                "time":self.start_time-time()
            }
    
    def process_query(self ,query:str ,chat_history: Optional[List[Dict[str, str]]] = None):
        start_time = time()
        try:
            #expand query
            self.logger.info(f"1>expanding query here {query}")
            expand_query = self.query_expander.expand_query(query)
            expand_query = expand_query['expand_query']
            self.logger.info(f"original {query}")
            self.logger.info(f"original {expand_query}")

            #retrieve query here
            self.logger.info(f"2>Retriever the query here")
            vector_store = self.vector_store._load_document()
            relevant_docs = self.vector_store.retrieve_relevant_document(
                query = query , 
                vectorstore=vector_store
            )
            self.logger.info(f"the length of retrieve document is {len(relevant_docs)}")

            #use reranker here
            self.logger.info(f"3>use Reranker here")
            if self.reranker and len(relevant_docs)>1:
                relevant_docs = self.reranker.re_ranking(query , relevant_docs)
                self.logger.info(f"the length of retrieve docs is {len(relevant_docs)}")
    
            else:
                reranked_docs = relevant_docs
            
            #generate response here
            self.logger.info(f"4>Generate Response Here")
            response = self.response_gen.generate_response(
                query=query,
                retrieved_docs=relevant_docs,
                #picture_paths=
                #chat_distory=
            )

            response["process_time"] = time() - start_time()
            return response

        except Exception as e:
            self.logger.error(f"an error has occure here {e}")
            return {
                "success":False , 
                "time":start_time-time()
            }

