"""Race Research Agent: runs exactly the analyses requested in the plan."""
from __future__ import annotations

from typing import Dict, List

from ..config import DEFAULT_CONFIG, ModelConfig
from ..models import RaceState
from ..research import RESEARCH_TOOLS
from ..workflow_state import WorkflowState

AGENT = "Race Research"


class ResearchAgent:
    def __init__(self, cfg: ModelConfig = DEFAULT_CONFIG):
        self.cfg = cfg

    def run(self, state: RaceState, analyses: List[str], ws: WorkflowState,
            already_done: Dict = None) -> Dict:
        findings = dict(already_done or {})
        for name in analyses:
            if name in findings:
                continue
            tool = RESEARCH_TOOLS.get(name)
            if tool is None:
                ws.record(AGENT, "tool", name, "unknown analysis requested; skipped")
                continue
            findings[name] = tool(state, self.cfg)
            ws.record(AGENT, "tool", name, " ".join(findings[name]["summary"]))
        skipped = sorted(set(RESEARCH_TOOLS) - set(analyses))
        ran = [a.replace("_", " ") for a in analyses if a in RESEARCH_TOOLS]
        ws.record(AGENT, "agent", "research_complete",
                  f"Ran {', '.join(ran)}"
                  + (f". Skipped (not in the plan): {', '.join(s.replace('_', ' ') for s in skipped)}" if skipped else ""))
        return findings
