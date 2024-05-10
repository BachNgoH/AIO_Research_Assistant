import sys
from fastapi import FastAPI
from dotenv import load_dotenv
from api.controller import router

load_dotenv(override=True)
app = FastAPI()

@app.get("/")
async def root():
    return {
        "message": "Hello World!",
    }


app.include_router(router, prefix="/v1")