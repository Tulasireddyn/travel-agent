from fastapi import FastAPI, HTTPException
from backend.schemas.travel import TravelRequest, TravelResponse, ChatRequest, ChatResponse
from backend.orchestrator import create_travel_graph
from backend.agents.recommend_activities import recommend_activities
from backend.agents.fetch_useful_links import fetch_useful_links
from backend.agents.weather_forecaster import weather_forecaster
from backend.agents.packing_list_generator import packing_list_generator
from backend.agents.food_culture_recommender import food_culture_recommender
from backend.agents.chat_agent import chat_node
from dotenv import load_dotenv
import os
import asyncio

from backend.models.database import init_db

load_dotenv()

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="AI Travel Planner API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def on_startup():
    init_db()

graph = create_travel_graph()

@app.get("/")
async def root():
    return {"message": "AI Travel Planner API is running"}

@app.post("/generate-itinerary", response_model=TravelResponse)
async def generate_itinerary_endpoint(request: TravelRequest):
    try:
        initial_state = {
            "preferences": request.dict(),
            "itinerary": "",
            "weather_forecast": "",
            "packing_list": "",
            "useful_links": [],
            "activity_suggestions": "",
            "food_culture_info": "",
            "warning": ""
        }
        # Use ainvoke for async graph execution
        result = await graph.ainvoke(initial_state)
        return TravelResponse(**result)
    except Exception as e:
        print(f"Error in /generate-itinerary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/agent/activities", response_model=TravelResponse)
async def get_activities(state: dict):
    try:
        # recommend_activities might still be sync, but ainvoke handles both
        # If we make it async later, this remains compatible
        res = await asyncio.to_thread(recommend_activities, state) if not asyncio.iscoroutinefunction(recommend_activities) else await recommend_activities(state)
        return TravelResponse(**res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/agent/weather", response_model=TravelResponse)
async def get_weather(state: dict):
    try:
        res = await asyncio.to_thread(weather_forecaster, state) if not asyncio.iscoroutinefunction(weather_forecaster) else await weather_forecaster(state)
        return TravelResponse(**res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/agent/packing", response_model=TravelResponse)
async def get_packing(state: dict):
    try:
        res = await asyncio.to_thread(packing_list_generator, state) if not asyncio.iscoroutinefunction(packing_list_generator) else await packing_list_generator(state)
        return TravelResponse(**res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/agent/food-culture", response_model=TravelResponse)
async def get_food_culture(state: dict):
    try:
        res = await asyncio.to_thread(food_culture_recommender, state) if not asyncio.iscoroutinefunction(food_culture_recommender) else await food_culture_recommender(state)
        return TravelResponse(**res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/agent/links", response_model=TravelResponse)
async def get_links(state: dict):
    try:
        res = await asyncio.to_thread(fetch_useful_links, state) if not asyncio.iscoroutinefunction(fetch_useful_links) else await fetch_useful_links(state)
        return TravelResponse(**res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        state = {
            "preferences": request.context.get("preferences", {}),
            "itinerary": request.context.get("itinerary", ""),
            "chat_history": request.chat_history,
            "user_question": request.user_question,
            "chat_response": ""
        }
        res = await asyncio.to_thread(chat_node, state) if not asyncio.iscoroutinefunction(chat_node) else await chat_node(state)
        return ChatResponse(
            response=res["chat_response"],
            updated_history=res["chat_history"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
