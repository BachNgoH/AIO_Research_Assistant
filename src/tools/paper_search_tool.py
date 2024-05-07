import os
import torch
import chromadb
from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.core.schema import MetadataMode
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.tools import FunctionTool
from llama_index.core import QueryBundle
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore
from llama_index.core import Settings
from typing import List, Optional
from src.constants import EMBEDDING_MODEL_NAME, EMBEDDING_SERVICE


class PaperYearNodePostprocessor(BaseNodePostprocessor):
    def _postprocess_nodes(
        self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle]
    ) -> List[NodeWithScore]:
        # subtracts 1 from the score
        paper_year = query_bundle.query_str.split("\n")[0]
        if paper_year == "None":
            return nodes
        else:
            for n in nodes:
                print([n])

            return nodes


simple_content_template = """
Paper link: {paper_link}
Paper: {paper_content}
"""

def load_paper_search_tool():
    device_type = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    if EMBEDDING_SERVICE == "ollama":
        embed_model = OllamaEmbedding(model_name=EMBEDDING_MODEL_NAME)
    elif EMBEDDING_SERVICE == "hf":
        embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME, cache_folder="./models", device=device_type, embed_batch_size=64)
    elif EMBEDDING_SERVICE == "openai":
        embed_model = OpenAIEmbedding(model=EMBEDDING_MODEL_NAME, api_key=os.environ["OPENAI_API_KEY"])
    else:
        raise NotImplementedError()   


    chroma_client = chromadb.PersistentClient(path="./DB/arxiv")
    chroma_collection = chroma_client.get_or_create_collection("gemma_assistant_arxiv_papers")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)    
    # load the vectorstore
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    paper_index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context, embed_model=embed_model)
    Settings.llm = None
    paper_retriever = paper_index.as_retriever(
        similarity_top_k=5,    
    )
    node_postporcessor = PaperYearNodePostprocessor()
    
    def retrieve_paper(query_str: str, year: int = "None"):
        query_str = f"{year}\n{query_str}"
        retriever_response =  node_postporcessor.postprocess_nodes(
            paper_retriever.retrieve(query_str), 
            QueryBundle(query_str=query_str))
        
        retriever_result = []
        for n in retriever_response:
            paper_id = n.metadata["paper_id"]
            paper_content = n.node.get_content(metadata_mode=MetadataMode.LLM)
            
            paper_link = f"https://arxiv.org/abs/{paper_id}"
            retriever_result.append(
                simple_content_template.format(
                    paper_link=paper_link, 
                    paper_content=paper_content
                )
            )
        return retriever_result
            
        
    # paper_search_tool = QueryEngineTool.from_defaults(
    #     query_engine=paper_query_engine,
    #     description="Useful for answering questions related to scientific papers",
    # )
    return FunctionTool.from_defaults(retrieve_paper, description="Useful for answering questions related to scientific papers, add paper year if needed")