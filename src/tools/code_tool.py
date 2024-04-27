from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core import PromptTemplate
from llama_index.llms.openai import OpenAI

code_qa_prompt = PromptTemplate(
    "You are a code assistant powered by a large language model. "
    "Your task is to help users solve programming problems, provide code examples, "
    "explain programming concepts, and debug code. "
    "Write python code to answer the question bellow\n"
    "---------------------\n"
    "{query_str}\n"
    "---------------------\n"
    "Answer: "
)

class CodeQueryEngine(CustomQueryEngine):
    llm: OpenAI
    qa_prompt: PromptTemplate
    def custom_query(self , query_str: str):
        response = self.llm.complete(
            self.qa_prompt.format(query_str=query_str)
        )
#         program = extract_program(output.text)
#         executor = PythonExecutor(get_answer_from_stdout=True)
        
#         exe_result = executor.apply(program)
#         response = response.text + f"\n\n**Execution Output**:```output\n\n{exe_result[0]}\n\n```\n"
        return str(response)

def load_code_tool(llm):
    code_query_engine = CodeQueryEngine(llm = llm, qa_prompt=code_qa_prompt)
    return code_query_engine