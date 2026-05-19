from langchain_core.messages import HumanMessage
from backend.core.llm import get_llm

async def food_culture_recommender(state):
    llm = get_llm()
    prompt = f"""
    For a trip to {state['preferences'].get('destination', '')} with a {state['preferences'].get('budget_type', 'mid-range')} budget:
    1. Suggest popular local dishes and recommended dining options.
    2. Provide key cultural etiquette or tips for travelers.
    """
    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        return {"food_culture_info": response.content}
    except Exception:
        return {"food_culture_info": "Food and culture info unavailable."}