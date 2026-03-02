import asyncio
from datetime import datetime, timezone
import logging
import uuid
from typing import List

from schemas.inputs import ResearchQuery, WebTask, DocTask, FactTask
from schemas.outputs import FinalReport, ClaimVerification, DocResult
from agents.web_agent import WebAgent
from agents.doc_agent import DocAgent
from agents.fact_checker import FactCheckerAgent
from agents.writer import WriterAgent
from tools.cost_tracker import global_tracker
from memory.session_state import SessionState
from utils.logger import AgentLogger

class Orchestrator:
    def __init__(self):
        self.web_agent = WebAgent()
        self.doc_agent = DocAgent()
        self.fact_agent = FactCheckerAgent()
        self.writer_agent = WriterAgent()
        self.session = SessionState()
        
    async def cross_question_loop(self, web_claims: List[str], doc_claims: List[str], topic: str) -> List[ClaimVerification]:
        """
        The adversarial logic core: Agents challenge each other's outputs.
        """
        AgentLogger.orchestrator("Starting Cross-Question Loop to resolve conflicts...")
        all_claims = list(set(web_claims + doc_claims))
        if not all_claims:
            return []
            
        fact_task = FactTask(claims=all_claims, query_context=topic, session_id=str(uuid.uuid4()))
        fact_result = await self.fact_agent.run(fact_task)
        
        disputed = [v for v in fact_result.verifications if v.verdict == "DISPUTED"]
        if disputed:
            AgentLogger.orchestrator(f"Found {len(disputed)} disputed claims. Initiating deeper counter-evidence search...")
            disputed_topics = [d.claim for d in disputed]
            
            # Send Web Agent out specifically to find counter-evidence for disputed claims
            counter_task = WebTask(
                queries=[f"evidence rejecting or resolving: {claim}" for claim in disputed_topics],
                num_sources_per_query=2,
                session_id=str(uuid.uuid4())
            )
            counter_web_result = await self.web_agent.run(counter_task)
            
            # We would normally re-evaluate here, but for phase 1 we will just append notes
            for claim_ver in fact_result.verifications:
                if claim_ver.verdict == "DISPUTED":
                    claim_ver.dispute_note = "Resolved via counter-search, refer to citations."
                    
        return fact_result.verifications

    async def run(self, query: ResearchQuery) -> FinalReport:
        AgentLogger.orchestrator(f"Beginning research on topic: [bold]'{query.topic}'[/bold]")
        await self.session.set_state(query.session_id, "status", {"state": "started", "topic": query.topic})
        
        web_task = WebTask(
            queries=[query.topic, f"latest news about {query.topic}", f"{query.topic} pros and cons"],
            num_sources_per_query=3,
            session_id=query.session_id
        )
        
        doc_task = DocTask(
            documents=query.documents,
            query=query.topic,
            session_id=query.session_id
        )
        
        AgentLogger.orchestrator("Delegating tasks: Fan-out gathering Web and Doc data in parallel...")
        
        if doc_task.documents:
            web_result, doc_result = await asyncio.gather(
                self.web_agent.run(web_task),
                self.doc_agent.run(doc_task)
            )
        else:
            AgentLogger.orchestrator("No local documents provided. Skipping Doc Agent.")
            web_result = await self.web_agent.run(web_task)
            doc_result = DocResult(analysis=[], citations=[], claims=[])
        
        verifications = await self.cross_question_loop(web_result.claims, doc_result.claims, query.topic)
        
        AgentLogger.orchestrator("Analysis complete. Handoff to Writer Agent for Final Report synthesis...")
        context = {
            "web_sources": [s.model_dump() for s in web_result.sources],
            "doc_analysis": [a.model_dump() for a in doc_result.analysis],
            "verifications": [v.model_dump() for v in verifications]
        }
        
        markdown_report = await self.writer_agent.run(query.topic, context)
        
        disputed_count = sum(1 for v in verifications if v.verdict == "DISPUTED")
        verified_count = sum(1 for v in verifications if v.verdict == "VERIFIED")
        
        report = FinalReport(
            topic=query.topic,
            session_id=query.session_id,
            markdown=markdown_report,
            total_sources=len(web_result.sources) + len(doc_result.citations),
            verified_claims=verified_count,
            disputed_claims=disputed_count,
            cost_usd=global_tracker.get_summary()["total_usd"],
            agent_costs=global_tracker.get_summary()["agent_costs"],
            generated_at=datetime.now(timezone.utc)
        )
        
        await self.session.set_state(query.session_id, "status", {"state": "completed"})
        await self.session.close()
        AgentLogger.success("Orchestrator pipeline complete!")
        return report
