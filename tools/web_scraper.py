import aiohttp
from bs4 import BeautifulSoup

async def scrape_url(url: str) -> str:
    """
    Scrape the main text content from a given URL.
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Remove scripts, styles, and boilerplate
                    for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
                        element.extract()
                        
                    text = soup.get_text(separator=' ', strip=True)
                    # Simple heuristic: take the first 4000 characters to avoid huge contexts
                    return text[:4000]
                else:
                    return f"Failed to fetch {url}: HTTP {response.status}"
    except Exception as e:
        return f"Error scraping {url}: {e}"
