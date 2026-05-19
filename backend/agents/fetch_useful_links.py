from langchain_community.utilities import GoogleSerperAPIWrapper

import asyncio

async def fetch_useful_links(state):
    search = GoogleSerperAPIWrapper()
    destination = state['preferences'].get('destination', '')
    month = state['preferences'].get('month', '')
    
    query = f"top travel tips and guide for {destination} in {month}"
    try:
        # GoogleSerperAPIWrapper.results is synchronous, we run in thread
        results = await asyncio.to_thread(search.results, query)
        organic = results.get('organic', [])[:5]
        links = [{"title": res.get('title'), "link": res.get('link')} for res in organic]
        return {"useful_links": links}
    except Exception as e:
        return {"useful_links": [], "warning": f"Failed to fetch links: {str(e)}"}