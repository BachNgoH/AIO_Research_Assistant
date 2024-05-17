import os
import chromadb
import sys
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from dotenv import load_dotenv
from constants import EMBEDDING_SERVICE, EMBEDDING_MODEL_NAME
from llama_index.core.tools import FunctionTool
from llama_index.core.schema import MetadataMode
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core import VectorStoreIndex, StorageContext

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_KEY")

simple_content_template = """
Facebook Link: https://www.facebook.com/aivietnam.edu.vn
Web: https://aivietnam.edu.vn/

{content}
"""

def load_info_aio_tool():
    device_type = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    if EMBEDDING_SERVICE == "ollama":
        embed_model = OllamaEmbedding(model_name=EMBEDDING_MODEL_NAME)
    elif EMBEDDING_SERVICE == "hf":
        embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME, cache_folder="./models", device=device_type, embed_batch_size=10)
    elif EMBEDDING_SERVICE == "openai":
        embed_model = OpenAIEmbedding(model=EMBEDDING_MODEL_NAME, api_key=os.environ["OPENAI_API_KEY"])
    else:
        raise NotImplementedError()   

    chroma_client = chromadb.PersistentClient(path="./DB/general_info")
    chroma_collection = chroma_client.get_or_create_collection("general_info")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    general_info_index = VectorStoreIndex.from_vector_store(
        vector_store, storage_context=storage_context, embed_model=embed_model
    )

    retriever = general_info_index.as_retriever(similarity_top_k=2)

    rerank_postprocessor = SentenceTransformerRerank(
        model="mixedbread-ai/mxbai-rerank-xsmall-v1",
        top_n=2,  # number of nodes after re-ranking,
        keep_retrieval_score=True,
    )

    def retrieve_aio_info(query: str):
        retriever_response = retriever.retrieve(query)
        retriever_result = rerank_postprocessor.postprocess_nodes(
            retriever_response, query_str=query
        )
        retriever_result = [simple_content_template.format(
            content=n.get_content(MetadataMode.LLM)) for n in retriever_result]
        
        return "\n================\n".join(retriever_result)

    return FunctionTool.from_defaults(
        retrieve_aio_info,
        description="Useful for answering about general information of AIO course.",
    )
