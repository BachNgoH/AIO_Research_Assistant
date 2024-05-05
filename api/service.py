from llama_index.llms.groq import Groq
from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama
from llama_index.llms.gemini import Gemini
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector

import os
from src.tools import load_paper_search_tool, load_ds_tool, load_code_tool, load_graph_search_tool
from dotenv import load_dotenv
import logging
from constants import (
    SERVICE,
    TEMPERATURE,
    MODEL_ID
)
load_dotenv()

class AssistantService:
    query_engine: RouterQueryEngine
    
    def __init__(self):
        self.query_engine = self.create_query_engine()
    
    def create_query_engine(self):
        """
        Creates and configures a query engine for routing queries to the appropriate tools.
        
        This method initializes and configures a query engine for routing queries to specialized tools based on the query type.
        It loads a language model, along with specific tools for tasks such as code search and paper search.
        
        Returns:
            RouterQueryEngine: An instance of RouterQueryEngine configured with the necessary tools and settings.
        """
        llm = self.load_model(SERVICE, MODEL_ID)
        paper_search_tool = load_paper_search_tool(llm=llm)
        code_tool = load_code_tool(llm=llm)
        
        
        query_engine = RouterQueryEngine(
            selector=LLMSingleSelector,
            query_engine_tools=[
                code_tool,
                paper_search_tool
            ],
            verbose=True,
            llm=llm
        )
        return query_engine
    
    def load_model(service, model_id):
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
            return OpenAI(model=model_id, temperature=TEMPERATURE, api_key=os.environ["OPENAI_API_KEY"])
        elif service == "groq":
            logging.info(f"Loading Groq Model: {model_id}")    
            return Groq(model=model_id, temperature=TEMPERATURE, api_key=os.environ["GROQ_API_KEY"])
        elif service == "gemini":
            logging.info(f"Loading Gemini Model: {model_id}")
            return Gemini(model=model_id, temperature=TEMPERATURE, api_key=os.environ["GOOGLE_API_KEY"])
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
        response = self.query_engine.query(prompt)
        return response