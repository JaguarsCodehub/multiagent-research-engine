import asyncio
import argparse
import sys
import os
import re
from datetime import datetime

from dotenv import load_dotenv

# Optional: Add current dir to python paths
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Load env safely
load_dotenv()

from agents.orchestrator import Orchestrator
from schemas.inputs import ResearchQuery

from utils.logger import console, AgentLogger
from rich.markdown import Markdown
from rich.panel import Panel

async def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Research Content Engine")
    parser.add_argument("--query", type=str, required=True, help="The research topic to investigate.")
    parser.add_argument("--depth", type=str, default="medium", choices=["shallow", "medium", "deep"])
    parser.add_argument("--docs", type=str, nargs="*", default=[], help="Paths to local documents (PDF, TXT, MD, etc.).")
    args = parser.parse_args()
    
    if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "your_openai_api_key_here":
        AgentLogger.error("OPENAI_API_KEY environment variable is missing or placeholder. Execution may fail.")
        
    AgentLogger.system(f"Igniting Engine on Topic: [bold]'{args.query}'[/bold] (Depth: {args.depth})")
    if args.docs:
        AgentLogger.system(f"Loading {len(args.docs)} local document(s)...")
    
    orchestrator = Orchestrator()
    req = ResearchQuery(topic=args.query, depth=args.depth, documents=args.docs)
    
    try:
        report = await orchestrator.run(req)
        
        # Print actual markdown rendered using Rich
        console.print("\n")
        console.print(Panel(Markdown(report.markdown), title=f"📑 FINAL RESEARCH REPORT: {report.topic.upper()}", border_style="cyan"))
        
        metrics_text = (
            f"- [bold]Total Sources Read:[/bold] {report.total_sources}\n"
            f"- [bold]Verified Claims:[/bold]    {report.verified_claims}\n"
            f"- [bold]Disputed Claims:[/bold]    {report.disputed_claims}\n"
            f"- [bold]Total Cost (USD):[/bold]   ${report.cost_usd:.5f}\n"
        )
        metrics_text += "\n[bold]Agent Token Costs:[/bold]\n"
        for agent, cost in report.agent_costs.items():
            metrics_text += f"  - {agent}: ${cost:.5f}\n"
            
        console.print(Panel(metrics_text, title="📊 METRICS", border_style="green"))

        # Preserve the report to a file
        reports_dir = os.path.join(project_root, "reports")
        os.makedirs(reports_dir, exist_ok=True)
        
        # Create a safe filename from the topic
        safe_topic = re.sub(r'[^\w\s-]', '', report.topic).strip().replace(' ', '_')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"Report_{safe_topic}_{timestamp}.md"
        filepath = os.path.join(reports_dir, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(report.markdown)
            
        AgentLogger.system(f"Report preserved to: [bold][blue]{filepath}[/blue][/bold]")
        
    except Exception as e:
        AgentLogger.error(f"Execution failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
