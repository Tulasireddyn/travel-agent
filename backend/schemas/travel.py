from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class TravelRequest(BaseModel):
    origin: str = Field(..., example="Mumbai")
    destination: str = Field(..., example="Paris")
    month: str = Field(..., example="May")
    duration: int = Field(..., ge=1, le=30, example=7)
    num_people: str = Field(..., example="2")
    holiday_type: str = Field(..., example="Romantic")
    budget_type: str = Field(..., example="Luxury")
    comments: Optional[str] = Field(None, example="I love architecture and good food.")

class UsefulLink(BaseModel):
    title: str
    link: str

class TravelResponse(BaseModel):
    itinerary: Optional[str] = None
    activity_suggestions: Optional[str] = None
    useful_links: Optional[List[UsefulLink]] = []
    weather_forecast: Optional[str] = None
    packing_list: Optional[str] = None
    food_culture_info: Optional[str] = None
    warning: Optional[str] = None

class ChatRequest(BaseModel):
    user_question: str
    context: Dict
    chat_history: List[Dict[str, str]] = []

class ChatResponse(BaseModel):
    response: str
    updated_history: List[Dict[str, str]]
