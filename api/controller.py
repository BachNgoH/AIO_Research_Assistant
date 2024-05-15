from fastapi import APIRouter, Request
from .service import AssistantService
import asyncio
import schedule
from datetime import time
from src.tasks.paper_task import daily_ingest_analyze
import logging
import re

router = APIRouter()
assistant = AssistantService()

# --- Scheduled Task Logic ---

execution_time = "06:55"  # Default: 6:55 AM UTC 

def run_scheduled_task():
    print("Scheduled ingestion started!")
    daily_ingest_analyze()
    print("Scheduled ingestion executed!")

async def schedule_task():
    schedule.every().day.at(execution_time).do(run_scheduled_task)

    while True:
        schedule.run_pending()
        await asyncio.sleep(1)


# --- API Endpoints ---

@router.post("/complete")
async def complete_text(request: Request):
    data = await request.json()
    message = data.get("message")
    response = assistant.predict(message)
    return response

@router.post("/schedule/update")
async def update_schedule(request: Request):
    data = await request.json()
    new_time_str = data.get("new_time")
    global execution_time  # Access global variable

    # Time Format Validation
    time_pattern = re.compile(r"^(0[0-9]|1[0-9]|2[0-3]):[0-5][0-9]$")
    if not time_pattern.match(new_time_str):
        return {"error": "Invalid time format. Use HH:MM (e.g., 09:30)"}

    try:
        schedule.clear()  # Clear existing schedule
        execution_time = new_time_str
        asyncio.create_task(schedule_task())  # Restart scheduling
        return {"message": f"Schedule updated to {new_time_str}"}
    except ValueError:  # Catch any errors from invalid time values
        return {"error": "Invalid time format. Use HH:MM (e.g., 09:30)"}

@router.get("/schedule")
def get_schedule():
    return {"execution_time": execution_time}

# --- Startup ---
@router.on_event("startup")
async def start_schedule():
    asyncio.create_task(schedule_task())  # Initiate scheduling on app start
