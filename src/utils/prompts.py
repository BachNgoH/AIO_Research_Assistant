DEFAULT_CITATION_EXTRACTION_PROMPT_TEMPLATE = """
You are about to analyze segments of academic texts where authors cite other papers. 

### Instruction
Your task is to identify and articulate the nature of the relationship between the citing paper and the cited works. Focus on understanding why the author has chosen to reference these works and in what context. Use the following guidelines to structure your response:

- Identify the Citation: Start by locating the citation within the text. Citations might appear in various formats, such as numbers in brackets, author names, or footnotes.

- Contextual Clues: Read the sentences or paragraphs surrounding the citation to gather context. Pay attention to verbs and phrases that might indicate the relationship, such as "builds upon," "contradicts," "extends," "employs," or "reviews."

- Classify the Relationship: Based on the context and the way the cited work is discussed, classify the relationship into one of the following categories:

    - Supporting Evidence: The cited work provides foundational data, theories, or methodologies that support the claims or hypotheses of the citing paper.
    - Methodological Basis: The citing paper adopts or adapts the methods, techniques, or procedures from the cited work.
    - Theoretical Foundation: The citing paper leverages theories, models, or frameworks established in the cited work to underpin its own research.
    - Data Source: The citation is used to acknowledge the origin of a dataset, model, or specific information that the citing paper utilizes in its research or analysis. This category highlights the reliance on external data or pre-existing models as a foundational element for the study conducted in the citing paper.
    - Extension or Continuation: The citing paper expands upon the research of the cited work, exploring new dimensions, contexts, or variables.

For each citation, provide a brief summary in the format of JSON that includes:

- The citation number (as presented in the text).
- The category of the relationship (from the classifications provided above).
- A short explanation of how the cited work contributes to the citing paper.
The format of the output should be strictly in JSON format in the form: 
    
{{
    "Citation": "[citation number]",
    "Category": "[relationship category]",
    "Explanation": "[brief explanation of the relationship]"
}}

Example1 :
{{
    "Citation": "[1]",
    "Category": "Supporting Evidence",
    "Explanation": "The cited work provides a foundational algorithm that the citing paper leverages to improve computational efficiency in data analysis."
}}
Example 2:
{{
    "Citation": "(Bai et al. 2018)",
    "Category": "Supporting Evidence",
    "Explanation": "The cited work presents a comprehensive review of the literature on the topic, providing a strong theoretical foundation for the claims made in the citing paper."
}}

No explanation is needed
### Input
{input}
### Response
"""

DEFAULT_CITATION_INFER_PROMPT_TEMPLATE = """You are about to analyze segments of academic texts where authors cite other papers. 

### Instruction
Your task is to identify and articulate the nature of the relationship between the citing paper and the cited works. Focus on understanding why the author has chosen to reference these works and in what context. Use the following guidelines to structure your response:

- Identify the Citation: Start by locating the citation within the text. Citations might appear in various formats, such as numbers in brackets, author names, or footnotes.

- Contextual Clues: Read the sentences or paragraphs surrounding the citation to gather context. Pay attention to verbs and phrases that might indicate the relationship, such as "builds upon," "contradicts," "extends," "employs," or "reviews."

- Classify the Relationship: Based on the context and the way the cited work is discussed, classify the relationship into one of the following categories:

    - Supporting Evidence: The cited work provides foundational data, theories, or methodologies that support the claims or hypotheses of the citing paper.
    - Methodological Basis: The citing paper adopts or adapts the methods, techniques, or procedures from the cited work.
    - Theoretical Foundation: The citing paper leverages theories, models, or frameworks established in the cited work to underpin its own research.
    - Data Source: The citation is used to acknowledge the origin of a dataset, model, or specific information that the citing paper utilizes in its research or analysis. This category highlights the reliance on external data or pre-existing models as a foundational element for the study conducted in the citing paper.
    - Extension or Continuation: The citing paper expands upon the research of the cited work, exploring new dimensions, contexts, or variables.

For each citation, provide a brief summary in the format of JSON that includes:

- The citation number (as presented in the text).
- The category of the relationship (from the classifications provided above).
- A short explanation of how the cited work contributes to the citing paper.

### Input:
{input}

### Response:
"""

DEFAULT_SYSTEM_PROMPT = """
Bạn được đưa một nội dung từ một văn bản và công việc của bạn là trả lời một câu hỏi của user về nội dung đã được cung cấp
    
Một số quy luật cần tuân theo
    1. Trích dẫn trực tiếp ngữ cảnh vào trong câu trả lời qua các đường link được cung cấp
    2. Tránh những câu như 'Dựa vào ngữ cảnh được cung cấp,...' hay 'Trong bối cảnh ...' hay tất cả những câu tương tự

"""

DEFAULT_USER_PROMPT = """
Ngữ cảnh được cung cấp như sau
---------------------
{context_str}
---------------------
Dựa trên nội dung được cung cấp. Hãy trả lời câu hỏi từ người dùng. Nếu nội dung được cung cấp không hề liên quan hoặc không đủ để bạn đưa ra câu trả lời. Hãy nói rằng bạn "Tôi không có đủ thông tin để trả lời".
Hãy trả lời và kết thúc câu trả lời một cách đầy đủ.

Nếu bạn ghi nhớ và làm đúng những gì tôi đã dặn dò, tôi sẽ tip cho bạn $1000 vào cuối ngày
User question: {query_str}
AI Response:

"""

DEFAULT_REFINE_PROMPT = """
Câu hỏi gốc như sau: {query_str}
Chung ta có một câu trả lời có sẵn: {existing_answer}
Chúng tôi có cơ hội tinh chỉnh câu trả lời hiện có (chỉ khi cần) với một số ngữ cảnh khác bên dưới.
------------
{context_msg}
------------
Với bối cảnh mới, hãy tinh chỉnh câu trả lời ban đầu để trả lời truy vấn tốt hơn. Nếu ngữ cảnh không hữu ích, hãy trả lại câu trả lời ban đầu.
Refined Answer: 

"""