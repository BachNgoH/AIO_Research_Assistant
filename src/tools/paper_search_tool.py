from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
import torch



def load_paper_search_tool():
    device_type = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5", cache_folder="../models", device=device_type) # must be the same as the previous stage

    chroma_client = chromadb.PersistentClient(path="./gemma-assistant-db/arxiv")
    chroma_collection = chroma_client.get_or_create_collection("gemma_assistant_arxiv_papers")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    # load the vectorstore
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    paper_index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context, embed_model=embed_model)

    paper_query_engine = paper_index.as_query_engine(
        similarity_top_k=5,
    )

    return paper_query_engine