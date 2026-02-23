from langchain_core.messages import HumanMessage
from langchain_community.chat_models import ChatOllama
from langchain_community.utilities import GoogleSerperAPIWrapper
import json 

def generate_itinerary(state):
    llm = ChatOllama(model="llama3.2", base_url="http://localhost:11434")
    search = GoogleSerperAPIWrapper()
    
    # 1. Fetch real-time data for flights and budget
    origin = state['preferences'].get('origin', 'Mumbai') # Default if not provided
    destination = state['preferences']['destination']
    month = state['preferences']['month']
    budget_type = state['preferences']['budget_type']
    
    print(f"DEBUG: Searching for travel options from {origin} to {destination}...")
    try:
        # 1. Check for nearest airport first
        nearest_airport_search = search.run(f"nearest airport to {destination}")
        
        # 2. General flight search
        flight_search = search.run(f"flights from {origin} to {destination} in {month} typical price")
        
        # 3. Train Search
        train_search = search.run(f"trains from {origin} to {destination} price duration irctc")
        
        # 4. Bus Search
        bus_search = search.run(f"bus from {origin} to {destination} redbus abhibus price duration")
        
        budget_search = search.run(f"average daily travel cost {destination} {budget_type} budget")
    except Exception as e:
        print(f"DEBUG: Search failed: {e}")
        nearest_airport_search = "Could not fetch nearest airport."
        flight_search = "Could not fetch flight data."
        train_search = "Could not fetch train data."
        bus_search = "Could not fetch bus data."
        budget_search = "Could not fetch budget data."

    # 2. Inject into prompt
    prompt = f"""
    Using the following preferences and REAL-TIME RESEARCH DATA, create a detailed travel itinerary.
    
    User Preferences:
    {json.dumps(state['preferences'], indent=2)}

    REAL-TIME RESEARCH DATA (Use this for accuracy):
    - Nearest Airport Info: {nearest_airport_search}
    - Flight Info: {flight_search}
    - Train Info: {train_search}
    - Bus Info: {bus_search}
    - Budget Estimates: {budget_search}

    INSTRUCTIONS:
    1. TRANSPORTATION OPTIONS (CRITICAL):
       - You MUST present a "Transportation Options" section at the very beginning.
       - OPTION 1: Flights (Use 'Flight Info' & 'Nearest Airport Info'). Check if the destination has an airport. If not, explain that they must fly to the verified nearest airport and take ground transport.
       - OPTION 2: Train (Use 'Train Info'). Suggest trains if available, mentioning approximate duration and cost.
       - OPTION 3: Bus (Use 'Bus Info'). Suggest buses if available (mention Redbus/AbhiBus as sources).
       - Let the user know they can choose the best option based on their budget and convenience.
    2. BUDGET: Use the provided budget estimates.
    3. ITINERARY: Include sections for each day, dining options, and downtime.
    """
    
    try:
        print("DEBUG: Sending request to Ollama...")
        result = llm.invoke([HumanMessage(content=prompt)]).content
        print(f"DEBUG: Ollama response length: {len(result)}")
        if not result or not result.strip():
            return {"itinerary": "", "warning": "LLM returned empty response. Check if Ollama is running correctly and model is loaded."}
        return {"itinerary": result.strip(), "warning": ""}
    except Exception as e:
        print(f"DEBUG: Exception in generate_itinerary: {e}")
        return {"itinerary": "", "warning": str(e)}