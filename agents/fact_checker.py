import json
import logging
import re
from datetime import datetime, timezone
from openai import AsyncOpenAI

from config import WORKER_MODEL, OPENAI_API_KEY
from schemas.inputs import FactTask
from schemas.outputs import FactResult, ClaimVerification, Source
from tools.firecrawl_search import firecrawl_search
from tools.pinecone_tools import rag_query_pinecone
from tools.cost_tracker import global_tracker
from utils.logger import AgentLogger

client = AsyncOpenAI(api_key=OPENAI_API_KEY)

FACT_AGENT_PROMPT = """
You are an expert Fact Checker Agent.
Your task: Take a list of claims and rigorously cross-verify them using the live web and our internal vector memory. Act as a highly skeptical hedge fund auditor.
1. Use `rag_query_pinecone` to check if our internal SEC filings/documents already verify or dispute the claim.
2. Use `firecrawl_search` for external verification to check if live market realities contradict management claims.
3. For each claim, determine a verdict: "VERIFIED", "DISPUTED", "UNVERIFIABLE", or "NUANCED".
Return pure JSON matching the FactResult schema:
{"verifications": [{"claim": "", "verdict": "", "supporting_sources": [{"url": "", "title": "", "snippet": "", "retrieved_at": "", "embedding_id": ""}], "dispute_note": "Optional explanation of conflict"}]}
"""

AVAILABLE_TOOLS = {
    "firecrawl_search": firecrawl_search,
    "rag_query_pinecone": rag_query_pinecone
}

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "firecrawl_search",
            "description": "Searches the web for live financial data, news, and counter-evidence.",
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
            "name": "rag_query_pinecone",
            "description": "Query internal verified chunks regarding a claim.",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
        }
    }
]

class FactCheckerAgent:
    async def run(self, task: FactTask) -> FactResult:
        AgentLogger.fact(f"Verifying {len(task.claims)} claims against the context...")
        messages = [
            {"role": "system", "content": FACT_AGENT_PROMPT},
            {"role": "user", "content": f"Context: {task.query_context}\nClaims to Verify: {json.dumps(task.claims)}"}
        ]
        
        for step in range(10):
            response = await client.chat.completions.create(
                model=WORKER_MODEL,
                messages=messages,
                tools=TOOLS_SCHEMA,
                tool_choice="auto"
            )
            
            usage = response.usage
            global_tracker.add_cost("worker", usage.prompt_tokens, usage.completion_tokens)
            
            message = response.choices[0].message
            messages.append(message)
            
            if not message.tool_calls:
                try:
                    content = message.content or ""
                    
                    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
                    if json_match:
                        content_to_parse = json_match.group(1)
                    else:
                        content_to_parse = content.strip()
                        
                    if not content_to_parse:
                        raise ValueError("Empty response content from Fact Checker LLM")
                    
                    res_dict = json.loads(content_to_parse)
                    verifications = []
                    for v in res_dict.get("verifications", []):
                        sources = []
                        for s in v.get("supporting_sources", []):
                            if isinstance(s, dict):
                                # Ensure datetime is set if missing
                                if "retrieved_at" not in s or not s["retrieved_at"]:
                                    s["retrieved_at"] = datetime.now(timezone.utc).isoformat()
                                sources.append(Source(**s))
                            elif isinstance(s, str):
                                sources.append(Source(
                                    url=s, title="Unknown Source", snippet="Source URL provided without details",
                                    retrieved_at=datetime.now(timezone.utc), embedding_id=""
                                ))
                        v["supporting_sources"] = sources
                        verifications.append(ClaimVerification(**v))
                    return FactResult(verifications=verifications)
                except Exception as e:
                    AgentLogger.error(f"Fact Checker failed to parse JSON. Error: {e}")
                    AgentLogger.error(f"Raw content was: {message.content}")
                    return FactResult(verifications=[])
                    
            for tool_call in message.tool_calls:
                func_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                func = AVAILABLE_TOOLS.get(func_name)
                
                if func:
                    AgentLogger.fact(f"Tool call: {func_name}")
                    try:
                        res = await func(**args)
                        if func_name == "firecrawl_search":
                            res = [s.model_dump(mode="json") for s in res]
                        tool_res = json.dumps(res)
                    except Exception as e:
                        tool_res = f"Error: {e}"
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": func_name,
                        "content": tool_res
                    })

        return FactResult(verifications=[])
