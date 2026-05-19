import asyncio
import os
import time
from dotenv import load_dotenv
from backend.agents.generate_itinerary import generate_itinerary

load_dotenv()

async def test_itinerary():
    print("--- Testing Full Itinerary Generation Performance ---")
    state = {
        "preferences": {
            "origin": "Mumbai",
            "destination": "Hyderabad",
            "month": "May",
            "duration": 7,
            "budget_type": "Mid-Range",
            "holiday_type": "Family"
        }
    }
    
    start = time.time()
    try:
        print("Calling generate_itinerary...")
        result = await generate_itinerary(state)
        duration = time.time() - start
        
        print(f"\nRESULT SUMMARY:")
        print(f"Total Time: {duration:.2f}s")
        if result.get("warning"):
            print(f"Warning: {result.get('warning')}")
        if result.get("itinerary"):
            print(f"Itinerary Length: {len(result['itinerary'])} characters")
            # print(f"Snippet: {result['itinerary'][:200]}...")
        else:
            print("FAILED: No itinerary returned.")
            
    except Exception as e:
        print(f"CRASHED: {e}")

if __name__ == "__main__":
    asyncio.run(test_itinerary())
