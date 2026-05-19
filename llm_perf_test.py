import asyncio
import time
from backend.core.llm import get_llm
from langchain_core.messages import HumanMessage

async def call_llm(name, prompt):
    llm = get_llm()
    start = time.time()
    print(f"[{name}] Starting...")
    try:
        res = await llm.ainvoke([HumanMessage(content=prompt)])
        print(f"[{name}] Finished in {time.time() - start:.2f}s")
        return res.content
    except Exception as e:
        print(f"[{name}] Error: {e}")
        return None

async def test_parallel():
    print("--- Testing Parallel LLM Calls ---")
    start = time.time()
    prompts = [
        "What is the capital of France? Just the name.",
        "What is 2+2? Just the number.",
        "Tell me a 1-sentence joke."
    ]
    tasks = [call_llm(f"Call {i}", p) for i, p in enumerate(prompts)]
    await asyncio.gather(*tasks)
    print(f"Total Parallel Time: {time.time() - start:.2f}s")

    print("\n--- Testing Serial LLM Calls ---")
    start = time.time()
    for i, p in enumerate(prompts):
        await call_llm(f"Call {i}", p)
    print(f"Total Serial Time: {time.time() - start:.2f}s")

if __name__ == "__main__":
    asyncio.run(test_parallel())
