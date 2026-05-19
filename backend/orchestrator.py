from typing import TypedDict, Annotated, List, Dict
from langgraph.graph import StateGraph, END
from backend.agents.generate_itinerary import generate_itinerary
from backend.agents.recommend_activities import recommend_activities
from backend.agents.fetch_useful_links import fetch_useful_links
from backend.agents.weather_forecaster import weather_forecaster
from backend.agents.packing_list_generator import packing_list_generator
from backend.agents.food_culture_recommender import food_culture_recommender
from backend.agents.chat_agent import chat_node

class GraphState(TypedDict):
    preferences: dict
    itinerary: str
    weather_forecast: str
    packing_list: str
    useful_links: List[dict]
    activity_suggestions: str
    food_culture_info: str
    chat_history: List[dict]
    user_question: str
    chat_response: str
    warning: str

def create_travel_graph():
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("generate_itinerary", generate_itinerary)
    workflow.add_node("recommend_activities", recommend_activities)
    workflow.add_node("fetch_useful_links", fetch_useful_links)
    workflow.add_node("weather_forecaster", weather_forecaster)
    workflow.add_node("packing_list_generator", packing_list_generator)
    workflow.add_node("food_culture_recommender", food_culture_recommender)
    
    # Define execution flow
    workflow.set_entry_point("generate_itinerary")
    
    # After itinerary is generated, run all other agents in parallel
    workflow.add_edge("generate_itinerary", "recommend_activities")
    workflow.add_edge("generate_itinerary", "fetch_useful_links")
    workflow.add_edge("generate_itinerary", "weather_forecaster")
    workflow.add_edge("generate_itinerary", "packing_list_generator")
    workflow.add_edge("generate_itinerary", "food_culture_recommender")
    
    # All nodes lead to END (implicit for parallel branches in simple graphs like this)
    workflow.add_edge("recommend_activities", END)
    workflow.add_edge("fetch_useful_links", END)
    workflow.add_edge("weather_forecaster", END)
    workflow.add_edge("packing_list_generator", END)
    workflow.add_edge("food_culture_recommender", END)
    
    return workflow.compile()
