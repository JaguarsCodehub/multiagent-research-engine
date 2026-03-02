import os
import sys
import aiohttp
from datetime import datetime, timezone
from typing import List

# Determine the absolute path to the project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from config import OPENNINJA_API_KEY
from schemas.outputs import Source

async def openninja_search(query: str, num_results: int = 3) -> List[Source]:
    """
    Search the web using OpenNinja Direct REST API.
    Note: Endpoint and response structure might need adjustment based on official OpenNinja API docs.
    """
    if not OPENNINJA_API_KEY or OPENNINJA_API_KEY == "your_openninja_api_key_here":
        print("Warning: OPENNINJA_API_KEY is not set. Returning empty results.")
        return []

    # Endpoint from documentation screenshots: https://api.openwebninja.com/realtime-web-search/search
    url = "https://api.openwebninja.com/realtime-web-search/search"
    
    headers = {
        "x-api-key": OPENNINJA_API_KEY,
        "Content-Type": "application/json"
    }
    
    params = {
        "q": query,
        "max_results": num_results
    }
    
    sources = []
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Logic to handle the specific response structure from screenshots
                    # The screenshot shows "text_parts" in the response schema
                    results = data.get("organic_results", [])
                    if not results and "results" in data:
                        results = data.get("results", [])
                    
                    # If we get "text_parts" but no organic results, it might be an AI Mode or direct answer
                    if not results and "text_parts" in data:
                        snippet = " ".join([p.get("text", "") for p in data.get("text_parts", []) if isinstance(p, dict)])
                        if snippet:
                            sources.append(Source(
                                url=f"https://www.google.com/search?q={query.replace(' ', '+')}",
                                title="OpenWeb Ninja AI / Direct Answer",
                                snippet=snippet,
                                retrieved_at=datetime.now(timezone.utc),
                                embedding_id=""
                            ))
                    
                    for i, res in enumerate(results[:num_results]):
                        sources.append(Source(
                            url=res.get("link", res.get("url", "")),
                            title=res.get("title", f"Result {i+1}"),
                            snippet=res.get("snippet", res.get("description", "")),
                            retrieved_at=datetime.now(timezone.utc),
                            embedding_id=""
                        ))
                elif response.status == 401:
                    print(f"OpenNinja Authentication Error (401): Please verify that your API key is correct and that you have 'Subscribed' or clicked 'Try Free' for the 'Real-Time Web Search' API in your OpenWeb Ninja dashboard.")
                else:
                    text = await response.text()
                    print(f"OpenNinja API error: {response.status} - {text}")
        except Exception as e:
            print(f"OpenNinja Search Request failed: {e}")
            
    return sources
