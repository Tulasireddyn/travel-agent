import asyncio
import os
import time
from dotenv import load_dotenv
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.messages import HumanMessage
from backend.core.llm import get_llm

load_dotenv()

async def diagnose():
    print("--- Starting Backend Diagnosis ---")
    
    # 1. Check Serper API
    print("\n1. Testing Google Serper API...")
    serper_key = os.getenv("SERPER_API_KEY")
    if not serper_key:
        print("ERROR: SERPER_API_KEY not found in .env")
    else:
        try:
            search = GoogleSerperAPIWrapper()
            start = time.time()
            res = await asyncio.to_thread(search.run, "test search")
            print(f"SUCCESS: Serper API responded in {time.time() - start:.2f}s")
            # print(f"Preview: {res[:100]}...")
        except Exception as e:
            print(f"ERROR: Serper API failed: {e}")

    # 2. Check Ollama
    print("\n2. Testing Ollama (Llama 3.2)...")
    try:
        llm = get_llm()
        print(f"Initialized LLM: {llm}")
        start = time.time()
        # Simple prompt to test responsiveness
        response = await llm.ainvoke([HumanMessage(content="Hello, say 'ready'")])
        print(f"SUCCESS: Ollama responded in {time.time() - start:.2f}s")
        print(f"Response: {response.content}")
    except Exception as e:
        print(f"ERROR: Ollama failed: {e}")

    print("\n--- Diagnosis Complete ---")

if __name__ == "__main__":
    asyncio.run(diagnose())
