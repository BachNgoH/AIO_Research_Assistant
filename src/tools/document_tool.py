import torch
import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.core.schema import MetadataMode

from llama_index.core import QueryBundle
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore
from typing import List, Optional
from llama_index.core.tools import FunctionTool

simple_content_template = """
Document: {paper_link}
Paper: {paper_content}
"""

def load_document_search_tool():
    device_type = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5", cache_folder="./models", device=device_type) # must be the same as the previous stage

    chroma_client = chromadb.PersistentClient(path="./DB/docs")
    chroma_collection = chroma_client.get_or_create_collection("gemma_assistant_aio_docs")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)    
    # load the vectorstore
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    paper_index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context, embed_model=embed_model)

    paper_retriever = paper_index.as_retriever(
        similarity_top_k=5,
    )
    
    def retrieve_ai_concepts(query_str: str):
        
        retriver_response =  paper_retriever.retrieve(query_str)
        retriever_result = []
        for n in retriver_response:
            print([n.node.metadata])
            file_name = n.node.metadata["file_name"]
            # paper_id = list(n.node.relationships.items())[0][1].node_id
            paper_content = n.node.get_content(metadata_mode=MetadataMode.LLM)
            
            document_link = f"https://github.com/BachNgoH/AIO_Documents/tree/main/Documents/{file_name}"
            retriever_result.append(
                simple_content_template.format(
                    paper_link=document_link, 
                    paper_content=paper_content
                )
            )
        return retriever_result
            
        
    # paper_search_tool = QueryEngineTool.from_defaults(
    #     query_engine=paper_query_engine,
    #     description="Useful for answering questions related to scientific papers",
    # )
    return FunctionTool.from_defaults(retrieve_ai_concepts, description="Useful for answering about AI and Python concepts")