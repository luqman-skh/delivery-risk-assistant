"""
Data Schemas and Pydantic Contracts for Multi-Agent Workflow.
"""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


ActionType = Literal[
    "CONTINUE_MONITORING",
    "CONTACT_LOGISTICS",
    "PRIORITIZE_SHIPMENT",
    "NOTIFY_CUSTOMER",
    "HUMAN_ESCALATION",
    "EXPEDITE_CARRIER",
    "RE_ROUTE_WAREHOUSE",
    "PROACTIVE_CUSTOMER_CREDIT",
]


class RiskPredictionOutput(BaseModel):
    order_id: int
    risk_score: float = Field(..., ge=0.0, le=1.0)
    risk_level: Literal["LOW", "MEDIUM", "HIGH"]
    is_delayed_risk: bool
    latency_ms: float
    top_factors: List[str]


class DeliveryRiskAssessment(BaseModel):
    order_id: int
    risk_score: float = Field(..., ge=0.0, le=1.0)
    risk_level: Literal["LOW", "MEDIUM", "HIGH"]
    recommended_action: ActionType
    supporting_evidence: List[str]
    policy_citation: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    escalate_to_human: bool
    tool_calls_executed: List[str] = Field(default_factory=list)
    processing_time_ms: float = 0.0
    model_used: str = "general-slm"
    human_override: Optional[dict] = None
