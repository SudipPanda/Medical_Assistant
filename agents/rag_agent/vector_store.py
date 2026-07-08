import os
import re
import logging
from typing import List , Dict , Any
from langchain_core.documents import Document
from langchain.storage import InMemoryStore, LocalFileStore
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, SparseVectorParams, VectorParams, OptimizersConfigDiff
from uuid4 import uuid4

class vector_store:
    def __init__(self , config):
        self.logger = logging.getLogger(__name__)
        self.collection_name = config.rag.collection_name
        self.embedding_dim = config.rag.embedding_dim
        self.distance_metric = config.rag.distance_metric
        self.embedding_model = config.rag.embedding_model
        self.retrieval_top_k = config.rag.top_k
        self.vector_search_type = config.rag.vector_search_type
        self.vectorstore_local_path = config.rag.vector_local_path
        self.docstore_local_path = config.rag.doc_local_path

        # Use the singleton client instead of creating a new one
        # self.client = QdrantClientManager.get_client(config)
        self.client = QdrantClient(path=self.vectorstore_local_path)
    
    def _does_collection_exist(self)->bool:
        try:
            collection_info = self.client.get_collections()
            collection_names = [collection.name for collection in collection_info.collections]
            return self.collection_name in collection_names

        except Exception as e:
            self.logger.error(f"Error checking for collection existence: {e}")
            return False


    def _create_collection(self):
        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={"dense": VectorParams(size=self.embedding_dim, distance=Distance.COSINE)},
                sparse_vectors_config={
                    "sparse": SparseVectorParams(index=models.SparseIndexParams(on_disk=False))
                },
            )

            self.logger.info(f"new collection created {self.collection_name}")
        except Exception as e:
            self.logger.error(f"an error has occure while creating collection {e}")
            raise e
        
    def _load_document(self):
        sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

        vector_store = QdrantVectorStore(
            client = self.client ,
            collection_name= self.collection_name ,
            embedding= self.embedding_model , 
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name="dense",
            sparse_vector_name="sparse",
        )

        return vector_store

    def create_vectorStore(
        self , 
        document_chunk:List[str] , 
        document_path
    ):

        docs_id = [str(uuid4) for _ in range(len(document_chunk))]

        langchain_document =[]

        for id_idx , chunk in enumerate(document_chunk):
            langchain_document.append(
                Document(
                    page_content = chunk , 
                    metadata={
                            "source": os.path.basename(document_path),
                            "doc_id": docs_id[id_idx],
                            # "source_path": Path(os.path.abspath(document_path)).as_uri()
                            #"source_path": os.path.join("http://localhost:8000/", document_path)
                        }
                )
            )
            
            #setting up the sparse embedding here
            sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
            
            #checking if collections exist or not here
            collection_exist = self._does_collection_exist()

            if not collection_exist:
                self._create_collection()
                self.logger.info("collection succesfull created here")
            else:
                self.logger.info("collection already is here")
            
            qdrant_vectorstore = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                embedding=self.embedding_model,
                sparse_embedding=sparse_embeddings,
                retrieval_mode=RetrievalMode.HYBRID,
                vector_name="dense",
                sparse_vector_name="sparse",
            )

            qdrant_vectorstore.add_documents(documents=langchain_document, ids=docs_id)
            
    
    def retrieve_relevant_document(self , query: str,vectorstore: QdrantVectorStore,
            ):
        
        result = vectorstore.similarity_search_with_score(
            query=query,
            k=self.retrieval_top_k
           )
        retrieve_docs = []

        for chunk , score in result:
            doc_dict = {
                "id": chunk.metadata['doc_id'],
                "content": chunk,
                "score": score,  # Use the actual similarity score
                "source": chunk.metadata['source'],
               # "source_path": chunk.metadata['source_path'],
            }

            retrieve_docs.append(doc_dict)
        
        return retrieve_docs
        

        





    
    
