from fastapi import APIRouter, Request
from .service import AssistantService

router = APIRouter()
assistant = AssistantService()
# logger = logging.getLogger(__name__)

@router.post("/complete")
async def complete_text(request: Request):
    data = await request.json()
    message = data.get("message")
    response = assistant.predict(message)
    source_links = []
    for node in response.source_nodes:
        source_links.append(node.node.metadata["link"])
    print(source_links)
    return {"completion": response.response, "sources": source_links}
