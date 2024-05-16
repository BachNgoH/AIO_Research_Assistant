import os
from llama_index.llms.groq import Groq
from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama
from llama_index.llms.gemini import Gemini
from llama_index.core.agent import AgentRunner
from llama_index.core import Settings
from src.tools.paper_search_tool import load_paper_search_tool, load_daily_paper_tool
from src.tools.document_tool import load_document_search_tool
from src.constants import SYSTEM_PROMPT
from starlette.responses import StreamingResponse, Response
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv
import logging
from src.constants import (
    SERVICE,
    TEMPERATURE,
    MODEL_ID,
    STREAM
)
load_dotenv(override=True)

MODE = "normal"

class AssistantService:
    query_engine: AgentRunner
    tools_dict: dict
    
    def __init__(self):

        self.tools_dict = {
            "paper_search_tool": load_paper_search_tool,
            "daily_paper_tool": load_daily_paper_tool,
            "document_search_tool": load_document_search_tool
        }
        self.query_engine = self.create_query_engine()
        

    
    def create_query_engine(self):
        """
        Creates and configures a query engine for routing queries to the appropriate tools.
        
        This method initializes and configures a query engine for routing queries to specialized tools based on the query type.
        It loads a language model, along with specific tools for tasks such as code search and paper search.
        
        Returns:
            AgentRunner: An instance of AgentRunner configured with the necessary tools and settings.
        """
        if MODE == "fpt":
            from llama_index.core import VectorStoreIndex
            from llama_index.embeddings.ollama import OllamaEmbedding
            from llama_index.embeddings.openai import OpenAIEmbedding
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
            from llama_index.vector_stores.chroma import ChromaVectorStore
            from llama_index.core import StorageContext
            import chromadb

            from src.constants import DOCUMENT_EMBEDDING_MODEL_NAME, DOCUMENT_EMBEDDING_SERVICE
            if DOCUMENT_EMBEDDING_SERVICE == "ollama":
                embed_model = OllamaEmbedding(model_name=DOCUMENT_EMBEDDING_MODEL_NAME)
            elif DOCUMENT_EMBEDDING_SERVICE == "hf":
                embed_model = HuggingFaceEmbedding(
                    model_name=DOCUMENT_EMBEDDING_MODEL_NAME, 
                    cache_folder="./models", 
                    embed_batch_size=64)
            elif DOCUMENT_EMBEDDING_SERVICE == "openai":
                embed_model = OpenAIEmbedding(
                    model=DOCUMENT_EMBEDDING_MODEL_NAME, 
                    api_key=os.environ["OPENAI_API_KEY"])
            else:
                raise NotImplementedError() 
            chroma_client = chromadb.PersistentClient(path="./DB/docs-fpt")
            chroma_collection = chroma_client.get_or_create_collection("gemma_assistant_fpt_docs")
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)    
            # load the vectorstore
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            paper_index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context, embed_model=embed_model)
            query_engine = paper_index.as_chat_engine(streaming=True)
            return query_engine
        
        llm = self.load_model(SERVICE, MODEL_ID)
        Settings.llm = llm
        paper_search_tool = self.tools_dict["paper_search_tool"]()
        document_search_tool = self.tools_dict["document_search_tool"]()
        daily_paper_tool = self.tools_dict["daily_paper_tool"]()
        
        query_engine = AgentRunner.from_llm(
            tools=[
                document_search_tool,
                paper_search_tool,
                daily_paper_tool
            ],
            verbose=True,
            llm=llm,
            system_prompt = SYSTEM_PROMPT
        )
        return query_engine
    
    def load_model(self, service, model_id):
        """
        Select a model for text generation using multiple services.
        Args:
            service (str): Service name indicating the type of model to load.
            model_id (str): Identifier of the model to load from HuggingFace's model hub.
        Returns:
            LLM: llama-index LLM for text generation
        Raises:
            ValueError: If an unsupported model or device type is provided.
        """
        logging.info(f"Loading Model: {model_id}")
        logging.info("This action can take a few minutes!")

        if service == "ollama":
            logging.info(f"Loading Ollama Model: {model_id}")
            return Ollama(model=model_id, temperature=TEMPERATURE)
        elif service == "openai":
            logging.info(f"Loading OpenAI Model: {model_id}")
            return OpenAI(model=model_id, temperature=TEMPERATURE, api_key=os.getenv("OPENAI_API_KEY"))
        elif service == "groq":
            logging.info(f"Loading Groq Model: {model_id}")    
            return Groq(model=model_id, temperature=TEMPERATURE, api_key=os.getenv("GROQ_API_KEY"))
        elif service == "gemini":
            logging.info(f"Loading Gemini Model: {model_id}")
            return Gemini(model=model_id, temperature=TEMPERATURE, api_key=os.getenv("GOOGLE_API_KEY"))
        else:
            raise NotImplementedError("The implementation for other types of LLMs are not ready yet!")
    
    def predict(self, prompt):
        """
        Predicts the next sequence of text given a prompt using the loaded language model.

        Args:
            prompt (str): The input prompt for text generation.

        Returns:
            str: The generated text based on the prompt.
        """
        # Assuming query_engine is already created or accessible
        if STREAM:

            streaming_response = self.query_engine.stream_chat(prompt)
            return StreamingResponse(streaming_response.response_gen, media_type="application/text; charset=utf-8")
        else:
            return Response(self.query_engine.chat(prompt).response, media_type="application/text; charset=utf-8")
        