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
    return response