import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from config import WORKER_MODEL, ORCHESTRATOR_MODEL, EMBEDDING_MODEL

# Pricing per 1k tokens (Update based on OpenAI current pricing)
PRICING = {
    "gpt-4o-mini": {"prompt": 0.00015, "completion": 0.00060},
    "gpt-4o": {"prompt": 0.00500, "completion": 0.01500},
    "text-embedding-3-small": {"prompt": 0.00002, "completion": 0.00000}
}

class CostTracker:
    def __init__(self):
        self.costs = {"worker": 0.0, "orchestrator": 0.0, "embedding": 0.0}
        self.tokens = {"worker": 0, "orchestrator": 0, "embedding": 0}
        
    def add_cost(self, agent_type: str, prompt_tokens: int, completion_tokens: int):
        model = WORKER_MODEL if agent_type == "worker" else ORCHESTRATOR_MODEL
        if agent_type == "embedding":
            model = EMBEDDING_MODEL
            
        p_cost = (prompt_tokens / 1000.0) * PRICING[model]["prompt"]
        c_cost = (completion_tokens / 1000.0) * PRICING[model]["completion"]
        
        self.costs[agent_type] += (p_cost + c_cost)
        self.tokens[agent_type] += (prompt_tokens + completion_tokens)
        
    def get_summary(self) -> dict:
        total_cost = sum(self.costs.values())
        return {
            "total_usd": round(total_cost, 5),
            "agent_costs": {k: round(v, 5) for k, v in self.costs.items()},
            "tokens": self.tokens
        }

# Global tracker for simple usage across modules
global_tracker = CostTracker()
