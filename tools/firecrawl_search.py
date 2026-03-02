import aiohttp
from datetime import datetime, timezone
from typing import List

from config import FIRECRAWL_API_KEY
from schemas.outputs import Source
from utils.logger import AgentLogger

async def firecrawl_search(query: str, limit: int = 3) -> List[Source]:
    """
    Searches the web using FireCrawl and returns LLM-ready markdown content for each result.
    """
    if not FIRECRAWL_API_KEY:
        AgentLogger.error("Missing FIRECRAWL_API_KEY in environment variables.")
        return []

    url = "https://api.firecrawl.dev/v1/search"
    headers = {
        "Authorization": f"Bearer {FIRECRAWL_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "query": query,
        "limit": limit,
        "scrapeOptions": {
            "formats": ["markdown"]
        }
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload, timeout=25) as response:
                if response.status == 401:
                    text_resp = await response.text()
                    AgentLogger.error(f"FireCrawl API error: 401 Unauthorized. {text_resp}")
                    return []
                elif response.status != 200:
                    text_resp = await response.text()
                    AgentLogger.error(f"FireCrawl API error: {response.status} - {text_resp}")
                    return []

                resp_json = await response.json()
                results = []
                data = resp_json.get("data", [])
                
                for item in data:
                    # Extract the raw markdown content
                    markdown_content = item.get("markdown", "")
                    
                    # Truncate to prevent context window explosion for the LLM
                    if len(markdown_content) > 3000:
                        markdown_content = markdown_content[:3000] + "\n\n...[Content Truncated]..."
                        
                    metadata = item.get("metadata", {})
                    title = metadata.get("title", item.get("url", "Untitled Document"))
                    
                    # Fallback to description if markdown is empty
                    snippet = markdown_content if markdown_content else metadata.get("description", "No content available.")
                    
                    results.append(Source(
                        url=item.get("url", ""),
                        title=title,
                        snippet=snippet,
                        retrieved_at=datetime.now(timezone.utc),
                        embedding_id=""
                    ))
                    
                return results

    except Exception as e:
        AgentLogger.error(f"FireCrawl Search Request failed: {e}")
        return []

