import json
import logging
import uuid
from typing import List, Dict
from openai import AsyncOpenAI

from config import WORKER_MODEL, OPENAI_API_KEY
from schemas.inputs import DocTask
from schemas.outputs import DocResult, DocAnalysis
from tools.pdf_parser import parse_document, chunk_document
from tools.pinecone_tools import store_to_pinecone, rag_query_pinecone
from tools.cost_tracker import global_tracker
from utils.logger import AgentLogger

client = AsyncOpenAI(api_key=OPENAI_API_KEY)

DOC_AGENT_PROMPT = """
You are an expert Financial Document Analysis Agent.
Your task: Analyze local PDFs/documents (like 10-Ks, M&A filings, or earnings transcripts), extract text, chunk them into the shared vector memory, and extract major financial guidance, growth metrics, and risk factors.
1. Use `parse_document` to extract raw text from files provided by the user.
2. Use `chunk_document` to slice the text into manageable chunks.
3. Use `store_to_pinecone` to save these chunks to Pinecone RAG for yourself and other agents.
4. After storing, analyze the document relative to the user's query and format your final findings.
Return pure JSON matching the DocResult schema:
{"analysis": [{"chunk_id": "", "content": "", "source_doc": ""}], "citations": ["doc1.pdf", "doc2.pdf"], "claims": ["claim 1"]}
"""

# Defining synchronous wrappers for simple tools to be called easily natively or wrapped async
async def async_parse_document(file_path: str): return parse_document(file_path)
async def async_chunk_document(text: str): return chunk_document(text)

AVAILABLE_TOOLS = {
    "parse_document": async_parse_document,
    "chunk_document": async_chunk_document,
    "store_to_pinecone": store_to_pinecone,
    "rag_query_pinecone": rag_query_pinecone
}

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "parse_document",
            "description": "Extract raw text from a local file path (PDF, TXT, MD, etc.).",
            "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}}, "required": ["file_path"]}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "chunk_document",
            "description": "Split long text into chunks.",
            "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "store_to_pinecone",
            "description": "Store a text chunk into Pinecone.",
            "parameters": {
                "type": "object", 
                "properties": {"chunk_id": {"type": "string"}, "text": {"type": "string"}, "source_url": {"type": "string"}}, 
                "required": ["chunk_id", "text"]
            }
        }
    }
]

class DocAgent:
    async def run(self, task: DocTask) -> DocResult:
        AgentLogger.doc(f"Analyzing {len(task.documents)} local document(s)...")
        messages = [
            {"role": "system", "content": DOC_AGENT_PROMPT},
            {"role": "user", "content": f"Query: {task.query}\nDocuments to analyze: {task.documents}"}
        ]
        
        for step in range(12):
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
                    
                    # Try to find a JSON block using regex if there's conversational text
                    import re
                    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
                    if json_match:
                        content_to_parse = json_match.group(1)
                    else:
                        content_to_parse = content.strip()
                        
                    if not content_to_parse:
                        raise ValueError("Empty response content from Doc Agent LLM")
                        
                    res_dict = json.loads(content_to_parse)
                    analysis = [DocAnalysis(**a) for a in res_dict.get("analysis", [])]
                    return DocResult(
                        analysis=analysis,
                        citations=res_dict.get("citations", []),
                        claims=res_dict.get("claims", [])
                    )
                except Exception as e:
                    AgentLogger.error(f"Doc Agent failed to parse JSON. Error: {e}")
                    AgentLogger.error(f"Raw content was: {message.content}")
                    return DocResult(analysis=[], citations=[], claims=[])
                    
            for tool_call in message.tool_calls:
                func_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                func = AVAILABLE_TOOLS.get(func_name)
                
                if func:
                    AgentLogger.doc(f"Tool call: {func_name}")
                    try:
                        res = await func(**args)
                        tool_res = json.dumps(res) if isinstance(res, (list, dict)) else str(res)
                    except Exception as e:
                        tool_res = f"Error: {e}"
                        
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": func_name,
                        "content": tool_res[:2000] # truncating to prevent context bloat from raw pdf texts!
                    })

        return DocResult(analysis=[], citations=[], claims=[])
