from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
import torch
from llama_index.core.postprocessor import SentenceTransformerRerank


def load_ds_tool(llm):
    device_type = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5", cache_folder="../models", device=device_type) # must be the same as the previous stage

    chroma_client = chromadb.PersistentClient(path="./gemma-assistant-db/wiki")
    chroma_collection = chroma_client.get_or_create_collection("gemma_assistant_wiki")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    # load the vectorstore
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    ds_index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context, embed_model=embed_model)

    rerank_postprocessor = SentenceTransformerRerank(
        model='mixedbread-ai/mxbai-rerank-xsmall-v1',
        top_n=3, # number of nodes after re-ranking, 
        keep_retrieval_score=True
    )   

    # re-define our query engine
    data_science_query_engine = ds_index.as_query_engine(
        similarity_top_k=10,
        llm=llm,
        node_postprocessors=[rerank_postprocessor],
    )

    return data_science_query_engine
