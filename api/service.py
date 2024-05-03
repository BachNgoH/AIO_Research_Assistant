from llama_index.llms.groq import Groq
import os
from src.tools import load_paper_search_tool, load_ds_tool, load_code_tool, load_graph_search_tool
from dotenv import load_dotenv

load_dotenv()

class AssistantService:

    def __init__(self) -> None:
        self.llm = self.load_model()

    def create_query_engine(self):
        paper_search_tool = load_paper_search_tool(llm=self.llm)
        code_tool = load_code_tool(llm=self.llm)
        
    def load_model(self):
        api_key = os.getenv("GROQ_API_KEY")
        llm = Groq(model="llama3-70b-8192")
        return llm
    
    @staticmethod
    def predict():
        pass