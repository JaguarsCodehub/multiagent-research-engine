import json
import logging
from typing import List, Dict
from openai import AsyncOpenAI

from config import WORKER_MODEL, OPENAI_API_KEY
from schemas.inputs import WebTask
from schemas.outputs import WebResult, Source
from tools.firecrawl_search import firecrawl_search
from tools.pinecone_tools import store_to_pinecone
from tools.cost_tracker import global_tracker
from utils.logger import AgentLogger

client = AsyncOpenAI(api_key=OPENAI_API_KEY)

WEB_AGENT_PROMPT = """
You are an expert Wall Street Financial Analyst & Web Researcher.
Your task: Use your FireCrawl tool to search the live web for the user's queries regarding markets, stocks, M&A, SEC filings, or competitive research.
1. Use `firecrawl_search` to find URLs and immediately retrieve the full markdown text of the result.
2. Use `store_to_pinecone` to save critical financial data, analyst quotes, or news paragraphs into our shared memory so the Fact Checker can find them later.
3. Finally, summarize the findings and extract the main factual financial claims you found. You MUST return your final response as a pure JSON object matching the WebResult schema:
{"sources": [{"url": "", "title": "", "snippet": "", "retrieved_at": "", "embedding_id": ""}], "claims": ["financial fact 1", "industry claim 2"], "embeddings_stored": int}
"""

AVAILABLE_TOOLS = {
    "firecrawl_search": firecrawl_search,
    "store_to_pinecone": store_to_pinecone
}

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "firecrawl_search",
            "description": "Searches the web and retrieves the full markdown content of the pages for financial due diligence.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "store_to_pinecone",
            "description": "Store a text chunk into Pinecone vector memory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "chunk_id": {"type": "string"},
                    "text": {"type": "string"},
                    "source_url": {"type": "string"}
                },
                "required": ["chunk_id", "text"]
            }
        }
    }
]

class WebAgent:
    async def run(self, task: WebTask) -> WebResult:
        AgentLogger.web(f"Starting web search on {len(task.queries)} queries...")
        messages = [
            {"role": "system", "content": WEB_AGENT_PROMPT},
            {"role": "user", "content": f"Please research these queries: {task.queries}"}
        ]
        
        num_embeddings_stored = 0
        
        for step in range(10): # Max 10 agent loop iterations to prevent infinite loops
            # Exclude response_format when calling tools, only use in final validation if needed
            response = await client.chat.completions.create(
                model=WORKER_MODEL,
                messages=messages,
                tools=TOOLS_SCHEMA,
                tool_choice="auto"
            )
            
            # Track cost
            usage = response.usage
            global_tracker.add_cost("worker", usage.prompt_tokens, usage.completion_tokens)
            
            message = response.choices[0].message
            messages.append(message)
            
            if not message.tool_calls:
                # Finished, should be JSON
                try:
                    content = message.content or ""
                    
                    # Try to find a JSON block using regex if there's conversational text
                    import re
                    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
                    if json_match:
                        content_to_parse = json_match.group(1)
                    else:
                        content_to_parse = content.strip()
                        
                    if not content_to_parse:
                        AgentLogger.web("Web Agent returned an empty response. Returning empty result.")
                        return WebResult(sources=[], claims=[], embeddings_stored=num_embeddings_stored)
                        
                    res_dict = json.loads(content_to_parse)
                    sources = [Source(**s) for s in res_dict.get("sources", [])]
                    
                    return WebResult(
                        sources=sources,
                        claims=res_dict.get("claims", []),
                        embeddings_stored=num_embeddings_stored
                    )
                except json.JSONDecodeError:
                    AgentLogger.web("Web Agent provided conversational text instead of JSON data (likely due to API search failures).")
                    return WebResult(sources=[], claims=[], embeddings_stored=num_embeddings_stored)
                except Exception as e:
                    AgentLogger.error(f"Web Agent encountered an unexpected error parsing results: {e}")
                    return WebResult(sources=[], claims=[], embeddings_stored=num_embeddings_stored)
                    
            # Handle tool calls
            for tool_call in message.tool_calls:
                func_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                
                func = AVAILABLE_TOOLS.get(func_name)
                if func:
                    AgentLogger.web(f"Tool call: {func_name}")
                    if func_name == "store_to_pinecone":
                        num_embeddings_stored += 1
                        
                    try:
                        res = await func(**args)
                        # JSON serialize Pydantic models from search
                        if func_name == "firecrawl_search":
                            res = [s.model_dump(mode="json") for s in res]
                        tool_res = json.dumps(res)
                    except Exception as e:
                        tool_res = f"Error executing tool: {e}"
                        
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": func_name,
                        "content": tool_res
                    })

        return WebResult(sources=[], claims=[], embeddings_stored=num_embeddings_stored)
