import torch
import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.core.vector_stores.types import MetadataFilters, ExactMatchFilter

def load_graph_search_tool():

    device_type = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5", cache_folder="./models", device=device_type) # must be the same as the previous stage

    chroma_client = chromadb.PersistentClient(path="./DB/graph")
    chroma_collection = chroma_client.get_or_create_collection("gemma_assistant_graph")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    # load the vectorstore
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    rel_index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context, embed_model=embed_model)


    filters = MetadataFilters(filters=[
        ExactMatchFilter(
            key="title", 
            value="active learning with statistical models"
        )
    ])
    return rel_index.as_query_engine()