import os
import requests
import fitz  # PyMuPDF
import time
from llama_index.core.tools import FunctionTool
from llama_index.llms.gemini import Gemini


summarize_template = """
You are an expert researcher, summarize the key points of the given paper. The summary should include the following sections:

Introduction:

- Briefly describe the context and motivation for the study.
- What problem or question does the paper address?

Methods:

- Outline the methodology used in the study.
- What approach or techniques were used to address the problem?

Results:

- Summarize the main findings or results of the study.
- What are the key outcomes?

Discussion:

- Interpret the significance of the results.
- How do the findings contribute to the field?
- Are there any limitations mentioned?

Conclusion:

- Summarize the main conclusions of the paper.
- What future directions or questions do the authors suggest?

####
Here is the paper content

{content}

"""

def download_arxiv_paper(arxiv_id, save_path="./outputs/temp_paper.pdf"):
    """
    Downloads a paper from arXiv given its ID and saves the PDF file.

    Parameters:
    arxiv_id (str): The arXiv ID of the paper to download.
    save_path (str): The path where the PDF file will be saved.

    Returns:
    bool: True if the download was successful, False otherwise.
    """
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    
    try:
        response = requests.get(pdf_url)
        if response.status_code == 200:
            with open(save_path, 'wb') as pdf_file:
                pdf_file.write(response.content)
            print(f"Paper {arxiv_id} has been downloaded successfully and saved to {save_path}.")
            return True
        else:
            print(f"Failed to download paper {arxiv_id}. HTTP status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"An error occurred while downloading the paper: {e}")
        return False

def extract_text_from_pdf(pdf_path):
    """
    Extracts text content from a PDF file.

    Parameters:
    pdf_path (str): The path to the PDF file.

    Returns:
    str: The extracted text content.
    """
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text()
        return text
    except Exception as e:
        print(f"An error occurred while extracting text from the PDF: {e}")
        return ""


def load_summarize_tool():
    
    summarize_llm = Gemini(model_name="models/gemini-1.5-flash-latest", api_key=os.environ["GOOGLE_API_KEY"])
    print("Load summarize LLM successfully.")
    
    def summarize(arxiv_id: str):
        """Summarize the paper with the given arXiv ID."""
        file_path = "./outputs/temp_paper.pdf"
        if download_arxiv_paper(arxiv_id, file_path):
            text_content = extract_text_from_pdf(file_path)
            print("Extracted text content:")
                
            prompt = summarize_template.format(content=text_content)
            try:
                response = summarize_llm.complete(prompt)
                
            except Exception as _:
                time.sleep(120)
                response = summarize_llm.complete(prompt)
    
            return response
        else:
            return f"Cannot find paper with id {arxiv_id}."
    
    return FunctionTool.from_defaults(summarize)