from langchain_core.messages import HumanMessage
from langchain_community.chat_models import ChatOllama
from langchain_community.utilities import GoogleSerperAPIWrapper
import json 

def recommend_activities(state):
    llm = ChatOllama(model="llama3.2", base_url="http://localhost:11434")
    search = GoogleSerperAPIWrapper()
    
    destination = state['preferences']['destination']
    month = state['preferences']['month']

    print(f"DEBUG: Searching for activities in {destination}...")
    try:
        activity_search = search.run(f"top unique things to do in {destination} {month} hidden gems")
    except Exception as e:
        activity_search = "Could not fetch activity data."

    prompt = f"""
    Based on the following preferences, itinerary, and VERIFIED SEARCH RESULTS, suggest unique local activities.
    
    Preferences: {json.dumps(state['preferences'], indent=2)}
    Itinerary Summary: {state['itinerary'][:500]}...
    
    VERIFIED SEARCH RESULTS (Use these to avoid hallucinating locations):
    {activity_search}

    INSTRUCTIONS:
    - Only suggest activities that actually exist and are mentioned in the search results or are well-known.
    - Provide suggestions in bullet points for each day.
    """
    try:
        result = llm.invoke([HumanMessage(content=prompt)]).content
        return {"activity_suggestions": result.strip()}
    except Exception as e:
        return {"activity_suggestions": "", "warning": str(e)}