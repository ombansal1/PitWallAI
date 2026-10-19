"""Shared workflow state passed between agents by the orchestrator."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .models import RaceState, SimulationResult, Strategy


@dataclass
class WorkflowState:
    analysis_id: str
    question: str
    input_snapshot: Dict[str, Any]
    trigger_event: Optional[Dict[str, Any]] = None
    model_mode: str = "mock"
    race_state: Optional[RaceState] = None
    input_issues: List[Dict[str, str]] = field(default_factory=list)
    plan: Optional[Dict[str, Any]] = None
    plan_history: List[Dict[str, Any]] = field(default_factory=list)
    research: Dict[str, Any] = field(default_factory=dict)
    candidates: Dict[str, Strategy] = field(default_factory=dict)
    simulations: Dict[str, SimulationResult] = field(default_factory=dict)
    rivals: Dict[str, Any] = field(default_factory=dict)
    positions: Dict[str, Any] = field(default_factory=dict)
    attempts: Dict[str, int] = field(default_factory=dict)
    critic_reports: List[Dict[str, Any]] = field(default_factory=list)
    revisions: List[Dict[str, Any]] = field(default_factory=list)
    retry_count: int = 0
    status: str = "running"
    selected_id: Optional[str] = None
    recommendation: Optional[Dict[str, Any]] = None
    log: List[Dict[str, Any]] = field(default_factory=list)
    listener: Optional[Any] = field(default=None, repr=False, compare=False)   # live progress callback

    def record(self, agent: str, kind: str, action: str, detail: str = "", **data) -> None:
        """Append an entry describing something that actually happened."""
        entry = {"step": len(self.log) + 1, "agent": agent, "kind": kind,
                 "action": action, "detail": detail, **({"data": data} if data else {})}
        self.log.append(entry)
        if self.listener is not None:
            self.listener(entry)
