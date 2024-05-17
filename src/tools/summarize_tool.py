from llama_index.core.tools import FunctionTool

def load_summarize_tool():
    
    def summarize(arxiv_id: str):
        """Summarize the paper with the given arXiv ID."""
        return arxiv_id
    
    return FunctionTool.from_defaults(summarize)