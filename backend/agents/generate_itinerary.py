import asyncio
import json 
import time
from langchain_core.messages import HumanMessage
from langchain_community.utilities import GoogleSerperAPIWrapper
from backend.core.llm import get_llm

async def generate_itinerary(state):
    llm = get_llm()
    search = GoogleSerperAPIWrapper()
    
    preferences = state.get('preferences', {})
    origin = preferences.get('origin', 'Mumbai')
    destination = preferences.get('destination')
    month = preferences.get('month')
    budget_type = preferences.get('budget_type')
    
    if not destination:
        return {"warning": "Destination is required."}

    print(f"\n[BACKEND] --- Start Generate Itinerary for {destination} ---")
    
    try:
        start_time = time.time()
        print(f"[BACKEND] Step 1: Running parallel searches...")
        
        # Parallel search with safety timeout
        async def safe_search(query):
            try:
                # 10s timeout for each search query
                return await asyncio.wait_for(asyncio.to_thread(search.run, query), timeout=10.0)
            except asyncio.TimeoutError:
                print(f"[BACKEND] WARNING: Search timeout for query: {query}")
                return "Search timed out."
            except Exception as e:
                print(f"[BACKEND] ERROR: Search failed for {query}: {e}")
                return "Search failed."

        tasks = [
            safe_search(f"nearest airport to {destination}"),
            safe_search(f"flights from {origin} to {destination} in {month} typical price"),
            safe_search(f"trains from {origin} to {destination} price duration irctc"),
            safe_search(f"bus from {origin} to {destination} redbus abhibus price duration"),
            safe_search(f"average daily travel cost {destination} {budget_type} budget")
        ]
        
        results = await asyncio.gather(*tasks)
        
        nearest_airport_search = results[0]
        flight_search = results[1]
        train_search = results[2]
        bus_search = results[3]
        budget_search = results[4]
        
        print(f"[BACKEND] Searches completed in {time.time() - start_time:.2f}s")
        
    except Exception as e:
        print(f"[BACKEND] FATAL ERROR in search orchestration: {e}")
        nearest_airport_search = flight_search = train_search = bus_search = budget_search = "Data unavailable."

    # 2. Inject into prompt
    # Drastic truncation to avoid stalling the LLM
    def truncate(text, limit=300):
        if not text: return "N/A"
        return text[:limit] + "..." if len(text) > limit else text

    prompt = f"Create a short 7-day travel itinerary for {destination} in {month}. Budget: {budget_type}. Use these notes: Transport: {truncate(flight_search, 100)}, Cost: {truncate(budget_search, 100)}. Keep it under 500 words."
    
    try:
        print("[BACKEND] Step 2: Sending prompt to LLM (Timeout: 60s)...")
        llm_start = time.time()
        
        # Wrap LLM call in a timeout
        try:
            response = await asyncio.wait_for(llm.ainvoke([HumanMessage(content=prompt)]), timeout=60.0)
            result = response.content
            print(f"[BACKEND] LLM response received in {time.time() - llm_start:.2f}s")
        except asyncio.TimeoutError:
            print("[BACKEND] WARNING: LLM timed out. Using high-quality fallback.")
            return {
                "itinerary": f"## Quick Itinerary: {destination} ({month})\n\n"
                             f"Your local AI model is taking a bit longer than expected, but here is a curated high-speed itinerary for your {budget_type} trip!\n\n"
                             f"### 🛄 Transport Concept\n"
                             f"- Suggested: {truncate(flight_search, 150)}\n\n"
                             f"### 📅 7-Day Snapshot\n"
                             f"**Day 1-2: Arrival & City Orientation** - Explore the central landmarks and local markets. {truncate(budget_search, 50)}\n"
                             f"**Day 3-4: Culture & Heritage** - Visit primary museums, heritage sites, and historical monuments.\n"
                             f"**Day 5-6: Local Hidden Gems** - Escape the crowds to nearby parks or smaller districts.\n"
                             f"**Day 7: Relaxation & Departure** - Leisurely breakfast followed by souvenir shopping.\n\n"
                             f"--- \n"
                             f"*Tip: Use the 'Activities' and 'Weather' buttons above for more live data once your system cools down!*",
                "warning": "Model generation timed out. Providing curated quick-start itinerary."
            }

        if not result or not result.strip():
            return {"itinerary": "Itinerary generation failed.", "warning": "Empty LLM response."}
        return {"itinerary": result.strip(), "warning": ""}
    except Exception as e:
        print(f"[BACKEND] ERROR in LLM call: {e}")
        return {"itinerary": "An error occurred during generation.", "warning": str(e)}
