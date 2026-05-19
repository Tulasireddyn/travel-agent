from langchain_core.messages import HumanMessage
from langchain_community.utilities import GoogleSerperAPIWrapper
from backend.core.llm import get_llm
import json 

async def recommend_activities(state):
    llm = get_llm()
    search = GoogleSerperAPIWrapper()
    
    preferences = state.get('preferences', {})
    destination = preferences.get('destination', '')
    month = preferences.get('month', '')
    itinerary_summary = state.get('itinerary', '')[:500] 
    
    print(f"DEBUG: Suggesting activities for {destination}...")
    
    prompt = f"""
    Suggest 3-5 unique local activities and hidden gems in {destination} for the month of {month}.
    Consider these preferences: {preferences.get('holiday_type', 'Any')}
    Context itinerary: {itinerary_summary}
    """
    
    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        return {"activity_suggestions": response.content}
    except Exception as e:
        print(f"DEBUG: Error in recommend_activities: {e}")
        return {"activity_suggestions": "Could not fetch activities."}