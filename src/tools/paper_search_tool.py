import os
import torch
import chromadb
import networkx as nx
from pyvis.network import Network
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.core.schema import MetadataMode
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.tools import FunctionTool
from llama_index.core import QueryBundle
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore
from llama_index.core import Settings
from typing import List, Optional
from src.constants import EMBEDDING_MODEL_NAME, EMBEDDING_SERVICE
from src.tools.graph_search_tool import create_ego_graph
import requests
import feedparser
from datetime import datetime

simple_content_template = """
Paper link: {paper_link}
Paper: {paper_content}
"""


class PaperYearNodePostprocessor(BaseNodePostprocessor):
    def _postprocess_nodes(
        self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle]
    ) -> List[NodeWithScore]:
        paper_year = query_bundle.query_str.split("\n")[0]
        if paper_year == "None":
            return nodes
        filtered_nodes = []
        for node in nodes:
            date = node.metadata.get('date', '')  # Get the date or default to empty string if not present
            if date:  # Check if date is not empty
                date_year = date.split('-')[0]  # Extract the year from the 'YYYY-MM-DD' format
                if date_year == str(paper_year):  # Compare the extracted year with the target year
                    filtered_nodes.append(node)
        return filtered_nodes


def load_paper_search_tool():
    device_type = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    if EMBEDDING_SERVICE == "ollama":
        embed_model = OllamaEmbedding(model_name=EMBEDDING_MODEL_NAME)
    elif EMBEDDING_SERVICE == "hf":
        embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME, cache_folder="./models", device=device_type, embed_batch_size=64)
    elif EMBEDDING_SERVICE == "openai":
        embed_model = OpenAIEmbedding(model=EMBEDDING_MODEL_NAME, api_key=os.environ["OPENAI_API_KEY"])
    else:
        raise NotImplementedError()   


    chroma_client = chromadb.PersistentClient(path="./DB/arxiv")
    chroma_collection = chroma_client.get_or_create_collection("gemma_assistant_arxiv_papers")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)    
    # load the vectorstore
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    paper_index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context, embed_model=embed_model)
    Settings.llm = None
    paper_retriever = paper_index.as_retriever(
        similarity_top_k=5,    
    )
    node_postporcessor = PaperYearNodePostprocessor()
    
    # graph = load_graph_data()
    graph = None
    
    def retrieve_paper(query_str: str, year: str = "None"):
        """
        Retrieves papers based on the given query string and optional year.

        Args:
            query_str (str): The query string used to search for papers.
            year (str, optional): The year of the papers to retrieve. Defaults to "None".

        Returns:
            list: A list of retrieved papers, each containing the paper link and content.
        """
        query_str = f"{year}\n{query_str}"
        retriever_response =  node_postporcessor.postprocess_nodes(
            paper_retriever.retrieve(query_str), 
            QueryBundle(query_str=query_str))
        
        retriever_result = []
        for n in retriever_response:
            paper_id = n.metadata["paper_id"]
            paper_content = n.node.get_content(metadata_mode=MetadataMode.LLM)
            
            paper_link = f"https://arxiv.org/abs/{paper_id}"
            retriever_result.append(
                simple_content_template.format(
                    paper_link=paper_link, 
                    paper_content=paper_content
                )
            )
            
        combined_ego_graph = create_ego_graph(retriever_response, service="ss", graph=graph)
        nt = Network(notebook=True)#, font_color='#10000000')
        nt.from_nx(combined_ego_graph)
        for node in nt.nodes:
            node['value'] = combined_ego_graph.nodes[node['id']]['size']

        nt.save_graph("./outputs/nx_graph.html")
        
        return retriever_result
            
        
    # paper_search_tool = QueryEngineTool.from_defaults(
    #     query_engine=paper_query_engine,
    #     description="Useful for answering questions related to scientific papers",
    # )
    return FunctionTool.from_defaults(retrieve_paper, description="Useful for answering questions related to scientific papers, add paper year if needed")


def load_daily_paper_tool():
    def get_latest_arxiv_papers():
        max_results = 25
        categories = ['cs.AI', 'cs.CV', 'cs.IR', 'cs.LG', 'cs.CL']
        base_url = 'http://export.arxiv.org/api/query?'
        all_categories = [f'cat:{category}' for category in categories]
        search_query = '+OR+'.join(all_categories)
        
        paper_list = []
        start = 0
        today = datetime.utcnow().date()
        
        while True:
            query = f'search_query={search_query}&start={start}&max_results={max_results}&sortBy=submittedDate&sortOrder=descending'
            response = requests.get(base_url + query)
            feed = feedparser.parse(response.content)
            
            new_papers_found = True
            for r in feed.entries:
                paper_date = datetime.strptime(r['published'][:10], '%Y-%m-%d').date()
                if paper_date != today:
                    new_papers_found = False
                    paper_list.append(f"""
    Title: {r['title']}
    Link: {r['link']}
    Summary: {r['summary']}
                    """)
            
            if not new_papers_found:
                break
            
            start += max_results
        
        return "\n==============\n".join(paper_list)
    
    return FunctionTool.from_defaults(get_latest_arxiv_papers, description="Useful for getting latest daily papers")


def load_get_time_tool():

    def get_current_time():
        """
        Returns the current time in the format: "YYYY-MM-DD HH:MM:SS".
        """
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    return FunctionTool.from_defaults(get_current_time)