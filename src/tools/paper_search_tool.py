import torch
import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.core.tools import QueryEngineTool

from llama_index.core import QueryBundle
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore
from typing import List, Optional

class LinkNodePostprocessor(BaseNodePostprocessor):
    def _postprocess_nodes(
        self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle]
    ) -> List[NodeWithScore]:
        # subtracts 1 from the score

        for n in nodes:
            paper_id = list(n.node.relationships.items())[0][1].node_id
            n.node.metadata = {"link": f"https://arxiv.org/abs/{paper_id}"} 

        return nodes

def load_paper_search_tool(llm):
    device_type = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5", cache_folder="./models", device=device_type) # must be the same as the previous stage

    chroma_client = chromadb.PersistentClient(path="./DB/arxiv")
    chroma_collection = chroma_client.get_or_create_collection("gemma_assistant_arxiv_papers")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)    
    # load the vectorstore
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    paper_index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context, embed_model=embed_model)

    paper_query_engine = paper_index.as_query_engine(
        streaming=True,
        similarity_top_k=5,
        llm=None,
        node_postprocessors=[LinkNodePostprocessor()]
    )
    
    paper_search_tool = QueryEngineTool.from_defaults(
        query_engine=paper_query_engine,
        description="Useful for answering questions related to scientific papers",
    )
    return paper_search_tool