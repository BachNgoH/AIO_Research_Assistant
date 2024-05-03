import torch
import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

device_type = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def ingest_paper():
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5", cache_folder="../models", device=device_type)
    chroma_client = chromadb.PersistentClient(path="../DB/arxiv")
    chroma_collection = chroma_client.get_or_create_collection("gemma_assistant_arxiv_papers")


    # Create vector store
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        arxiv_documents, storage_context=storage_context, embed_model=embed_model, show_progress=True
    )