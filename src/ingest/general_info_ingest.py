import os
import chromadb
import sys
import torch
from llama_parse import LlamaParse
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

from dotenv import load_dotenv
from constants import EMBEDDING_SERVICE, EMBEDDING_MODEL_NAME
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext

load_dotenv(override=True)

device_type = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_general_info():
    if EMBEDDING_SERVICE == "ollama":
        embed_model = OllamaEmbedding(model_name=EMBEDDING_MODEL_NAME)
    elif EMBEDDING_SERVICE == "hf":
        embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME, cache_folder="./models", device=device_type, embed_batch_size=10)
    elif EMBEDDING_SERVICE == "openai":
        embed_model = OpenAIEmbedding(model=EMBEDDING_MODEL_NAME, api_key=os.environ["OPENAI_API_KEY"])
    else:
        raise NotImplementedError()   

    chroma_client = chromadb.PersistentClient("./DB/general_info")
    general_info_collection = chroma_client.create_collection("general_info")

    parser_gpt4o = LlamaParse(
        result_type="markdown",
        # base_url="https://api.staging.llamaindex.ai",
        api_key=os.environ["LLAMA_PARSE_API_KEY"],
        gpt4o_mode=True,
        gpt4o_api_key=os.environ["OPENAI_API_KEY"]
    )
    
    documents = SimpleDirectoryReader(
        "./data/General_Documents",
        file_extractor={".pdf": parser_gpt4o}
    ).load_data()

    vector_store = ChromaVectorStore(chroma_collection=general_info_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, embed_model=embed_model
    )


if __name__ == "__main__":
    load_general_info()
