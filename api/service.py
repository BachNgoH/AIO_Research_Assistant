from llama_index.llms.groq import Groq
from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama
from llama_index.llms.gemini import Gemini
import os
from src.tools import load_paper_search_tool, load_ds_tool, load_code_tool, load_graph_search_tool
from dotenv import load_dotenv
import logging

load_dotenv()

class AssistantService:

    def __init__(self) -> None:
        self.llm = self.load_model()

    def create_query_engine(self):
        paper_search_tool = load_paper_search_tool(llm=self.llm)
        code_tool = load_code_tool(llm=self.llm)
    
    @staticmethod
    def load_model(service="ollama", device_type="cpu", model_id="", model_basename=None):
        """
        Select a model for text generation using the HuggingFace library.
        If you are running this for the first time, it will download a model for you.
        subsequent runs will use the model from the disk.

        Args:
            service (str): Service name indicating the type of model to load.
            device_type (str): Type of device to use, e.g., "cuda" for GPU or "cpu" for CPU.
            model_id (str): Identifier of the model to load from HuggingFace's model hub.
            model_basename (str, optional): Basename of the model if using quantized models.
                Defaults to None.

        Returns:
            HuggingFacePipeline: A pipeline object for text generation using the loaded model.

        Raises:
            ValueError: If an unsupported model or device type is provided.
        """
        logging.info(f"Loading Model: {model_id}, on: {device_type}")
        logging.info("This action can take a few minutes!")

        if service == "ollama":
            logging.info(f"Loading Ollama Model: {model_id}")
            return Ollama(model=model_id, temperature=cfg.MODEL.TEMPERATURE)
        elif service == "openai":
            logging.info(f"Loading OpenAI Model: {model_id}")
            return OpenAI(model=model_id, temperature=cfg.MODEL.TEMPERATURE, api_key=os.environ["OPENAI_API_KEY"])
        elif service == "groq":
            logging.info(f"Loading Groq Model: {model_id}")    
            return Groq(model=model_id, temperature=cfg.MODEL.TEMPERATURE, api_key=os.environ["GROQ_API_KEY"])
        elif service == "gemini":
            logging.info(f"Loading Gemini Model: {model_id}")
            return Gemini(model=model_id, temperature=cfg.MODEL.TEMPERATURE, api_key=os.environ["GOOGLE_API_KEY"])
        else:
            raise NotImplementedError("The implementation for other types of LLMs are not ready yet!")
    
    @staticmethod
    def predict():
        pass