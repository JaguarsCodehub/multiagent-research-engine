from openai import AsyncOpenAI
import json

from config import ORCHESTRATOR_MODEL, OPENAI_API_KEY
from tools.cost_tracker import global_tracker
from utils.logger import AgentLogger

client = AsyncOpenAI(api_key=OPENAI_API_KEY)

WRITER_PROMPT = """
You are an expert Wall Street Research Analyst.
Your task: Synthesize all verified claims, financial metrics, and dispute notes into a highly structured, professional Investment Due Diligence Memo.
The memo should include:
- Executive Summary
- ASCII Visual Comparisons (Use ASCII tables or simple bar/flow diagrams to compare KPIs like Revenue, Margin, or Market Share)
- Market Dynamics & Web Sentiment (The live realities)
- Internal Claims & Guidance (What the local documents said)
- Discrepancies & Risk Factors (Highlight conflicts found by Fact Checker)
- Conclusion / Bull vs. Bear Case
- You must use inline citations (e.g., [1], [2]) corresponding to the actual sources provided.
- Include a 'References' section at the bottom mapping the citation numbers to the URLs or document names.
- Output ONLY the raw Markdown text. Do not wrap in ```markdown blocks if possible.
"""

class WriterAgent:
    async def run(self, topic: str, write_context: dict) -> str:
        """
        Takes the aggregated context and drafts the final report.
        """
        AgentLogger.writer(f"Drafting final structured report for '{topic}'...")
        # We use a larger/smarter model for the final writing pass
        messages = [
            {"role": "system", "content": WRITER_PROMPT},
            {"role": "user", "content": f"Topic: {topic}\n\nContext & Verified Data:\n{json.dumps(write_context, default=str)}"}
        ]
        
        response = await client.chat.completions.create(
            model=ORCHESTRATOR_MODEL,
            messages=messages
        )
        
        usage = response.usage
        global_tracker.add_cost("orchestrator", usage.prompt_tokens, usage.completion_tokens)
        
        return response.choices[0].message.content
