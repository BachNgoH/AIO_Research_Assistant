import os 
from fastapi import FastAPI, Request
from llama_index.llms.groq import Groq
from dotenv import load_dotenv
from api import controller

load_dotenv()
# run on port 8001
app = FastAPI()

@app.get("/")
async def root():
    return {
        "message": "Hello World!",
    }


app.include_router(controller.router, prefix="/v1")