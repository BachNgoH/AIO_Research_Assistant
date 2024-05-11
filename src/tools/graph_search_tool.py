from tqdm import tqdm
import json
import torch
import networkx as nx
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.core.vector_stores.types import MetadataFilters, ExactMatchFilter

relationships_dict = {
    "Supporting Evidence": "Is Evidence For",
    "Methodological Basis": "Is Methodological Basis For",
    "Theoretical Foundation": "Is Theoretical Foundation For", 
    "Data Source": "Is Data Source For",
    "Extension or Continuation": "Is Extension or Continuation Of",
}

class PaperNode:
    title: str
    arxiv_id: str
    
    def __init__(self, title, arxiv_id):
        self.title = title
        self.arxiv_id = arxiv_id

    def __str__(self) -> str:
        return f"Title: {self.title},\n Arxiv ID: {self.arxiv_id}"

class PaperEdge:
    category: str
    explanation: str
    verbose = True

    def __init__(self, category, explanation):
        self.category = category
        self.explanation = explanation

    def __str__(self) -> str:
        if self.verbose:
            return f"Category: {self.category},\n Explanation: {self.explanation}"
        else:
            return f"Category: {self.category}"


def find_connected_nodes(graph, node, relationship=None):
    """
    Find nodes connected to the given node with an optional filter on the type of relationship.
    """
    connected_nodes = []
    for n, nbrs in graph.adj.items():
        if n == node:
            for nbr, eattr in nbrs.items():
                if relationship is None or eattr['label'] == relationship:
                    connected_nodes.append(nbr)
    return connected_nodes

# Function to search for a node by arxiv_id and return its details
def find_nodes_by_arxiv_id(graph, arxiv_id):
    for node, data in graph.nodes(data=True):
        if data.get('arxiv_id') == arxiv_id:
            return data  # or return data['paper_node'] to return the PaperNode object itself
    return "Paper not found in the graph."


def find_shortest_path(graph, source, target):
    """
    Find the shortest path between two nodes.
    """
    try:
        path = nx.shortest_path(graph, source=source, target=target)
        return path
    except nx.NetworkXNoPath:
        return None


def find_nodes_by_keyword(graph, keyword):
    """
    Find nodes that contain the given keyword in their name and retrieve their connected nodes and relationships.
    """
    keyword = keyword.lower()  # Convert keyword to lowercase for case-insensitive matching
    matching_nodes = [node for node in graph.nodes if keyword in node.lower()]

    # related_nodes = {}
    # for node in matching_nodes:
    #     connections = []
    #     for neighbor, details in graph[node].items():
    #         connections.append((neighbor, details['title'].split('\n')[0]))
    #     related_nodes[node] = connections

    return matching_nodes

def find_graph_nodes_from_retriever(graph, retrieved_nodes):
    all_nodes = []
    for r in retrieved_nodes:
        title = r.text.split("\n")[0]
        print(title)
        nodes = find_nodes_by_keyword(graph, title)
        if len(nodes) > 0:
            all_nodes += nodes
            
    return all_nodes

def load_graph_data() -> nx.DiGraph:
    paper_dict = {}
    with open("./outputs/parsed_arxiv_papers.json") as f:
        annotated_article = json.load(f)

    for article_dict in tqdm(annotated_article, total=len(annotated_article)):
        paper_dict[article_dict['title'].lower()] = PaperNode(title=article_dict['title'], arxiv_id=article_dict['arxiv_id'])

        if "mapped_citation" in article_dict.keys():
            for key,val in article_dict['mapped_citation'].items():
                title = val['title']
                if title not in paper_dict.keys():
                    paper_node = PaperNode(title=val['title'], arxiv_id=val['arxiv_id'])
                    paper_dict[title] = paper_node

    triplets = []
    error_count = 0

    for article_dict in annotated_article:
        if "mapped_citation" not in article_dict.keys():
            print(article_dict['title'])
            continue
        for key, val in article_dict['mapped_citation'].items():
            title = val['title']
            citation = val['citation']
            
            # Use a dictionary to group explanations by category
            category_explanations = {}
            for rel in citation:
                # try:
                if 'Category' in rel.keys() and 'Explanation' in rel.keys():
                    category = rel['Category']
                    explanation = rel['Explanation']
                    if category not in category_explanations:
                        category_explanations[category] = []
                    category_explanations[category].append(explanation)
                else:
                    error_count += 1


            source_node = paper_dict[title]
            target_node = paper_dict[article_dict['title'].lower()]

            # Construct triplets with aggregated explanations for each category
            if len(category_explanations.items()) > 0:
                for category, explanations in category_explanations.items():
                    if category not in relationships_dict.keys():
                        relationships_dict[category] = f"Is {category} Of"

                    aggregated_explanation = "; ".join(set(explanations))  # Remove duplicates and join explanations
                    rel = PaperEdge(category=category, explanation=aggregated_explanation)
                    reverse_rel = PaperEdge(category=relationships_dict[category], explanation=aggregated_explanation)

                    # Add the relationship in both directions
                    triplets.append((source_node, rel, target_node))
                    triplets.append((target_node, reverse_rel, source_node))
            else:
                rel = PaperEdge(category="Unk", explanation="Unk")
                triplets.append((source_node, rel, target_node))
                
    print("Number of triplets: ", len(triplets))
    
    G = nx.DiGraph()

    # Add nodes and edges
    for source_node, relationship, target_node in triplets:
        # Add nodes if they are not already in the graph
        if source_node.arxiv_id not in G:
            G.add_node(source_node.title, title=str(source_node), arxiv_id=source_node.arxiv_id)
        if target_node.arxiv_id not in G:
            G.add_node(target_node.title, title=str(target_node), arxiv_id=target_node.arxiv_id)
        
        # Add edge with relationship details
        G.add_edge(source_node.title, target_node.title, title=str(relationship), category=relationship.category, explanation=relationship.explanation)
        
    return G