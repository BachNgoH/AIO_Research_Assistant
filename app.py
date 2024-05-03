import os 
from fastapi import FastAPI, Request
from llama_index.llms.groq import Groq
app = FastAPI()

@app.post("/complete")
async def complete_text(request: Request):
    data = await request.json()
    message = data.get("message")
    api_key = data.get("api_key")

    if not message:
        return {"error": "Please provide 'text' in the request body"}
    llm = Groq(model="llama3-70b-8192", api_key=api_key)
    completion = llm.complete(message)
    
    return {"completion": completion.text}