import os
import torch
import chromadb
import sys
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import SimpleDirectoryReader, Settings
from llama_parse import LlamaParse  # pip install llama-parse
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from constants import DOCUMENT_EMBEDDING_MODEL_NAME, DOCUMENT_EMBEDDING_SERVICE
from dotenv import load_dotenv

load_dotenv(override=True)
device_type = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_data(source_dir):
    parser = LlamaParse(
        api_key=os.environ.get("LLAMA_PARSE_API_KEY"),  # can also be set in your env as LLAMA_CLOUD_API_KEY
        result_type="markdown"  # "markdown" and "text" are available
    )

    file_extractor = {'.docx': parser}
    return SimpleDirectoryReader(source_dir, file_extractor=file_extractor).load_data()


def ingest_paper():
    Settings.chunk_size = 2048
    Settings.chunk_overlap = 64
    documents = load_data("./data/FPT_Docs")
    
    if DOCUMENT_EMBEDDING_SERVICE == "ollama":
        embed_model = OllamaEmbedding(model_name=DOCUMENT_EMBEDDING_MODEL_NAME)
    elif DOCUMENT_EMBEDDING_SERVICE == "hf":
        embed_model = HuggingFaceEmbedding(model_name=DOCUMENT_EMBEDDING_MODEL_NAME, cache_folder="./models", device=device_type)
    elif DOCUMENT_EMBEDDING_SERVICE == "openai":
        embed_model = OpenAIEmbedding(model=DOCUMENT_EMBEDDING_MODEL_NAME, api_key=os.environ["OPENAI_API_KEY"])
    else:
        raise NotImplementedError()   
       
    chroma_client = chromadb.PersistentClient(path="./DB/docs-fpt")
    chroma_collection = chroma_client.get_or_create_collection("gemma_assistant_fpt_docs")


    # Create vector store
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, embed_model=embed_model, show_progress=True
    )
    
if __name__ == "__main__":
    ingest_paper()