from langchain_core.messages import HumanMessage
from backend.core.llm import get_llm

async def weather_forecaster(state):
    llm = get_llm()
    prompt = f"Provide a brief weather forecast for {state['preferences'].get('destination', '')} in {state['preferences'].get('month', '')}. Mention typical temperature and conditions."
    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        return {"weather_forecast": response.content}
    except Exception:
        return {"weather_forecast": "Weather info unavailable."}