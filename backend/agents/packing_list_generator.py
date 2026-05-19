from langchain_core.messages import HumanMessage
from backend.core.llm import get_llm

async def packing_list_generator(state):
    llm = get_llm()
    prompt = f"""
    Generate a comprehensive packing list for a {state['preferences'].get('holiday_type', 'general')} holiday in {state['preferences'].get('destination', '')} during {state['preferences'].get('month', '')} for {state['preferences'].get('duration', 0)} days.
    Include essentials based on expected weather and trip type.
    """
    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        return {"packing_list": response.content}
    except Exception:
        return {"packing_list": "Packing list could not be generated."}