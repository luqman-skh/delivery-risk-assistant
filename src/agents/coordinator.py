"""
Multi-Agent Coordinator (WP 1.5).
Orchestrates specialized subagents and tools:
- RiskAgent (Tabular ML / SLM Risk Assessment)
- Operational Tools (Order, Customer, Corridor, Policy)
- SLM Decision Engine (General-Purpose SLM in Phase 1)
- Human-in-the-Loop Override Interface (FR6)
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import jinja2

from src.agents.risk_agent import RiskAgent
from src.agents.schemas import DeliveryRiskAssessment
from src.models.decision_model import BaseDecisionModel, GeneralPurposeSLM
from src.tools.operational_tools import (
    get_customer_profile,
    get_order_details,
    get_recovery_policy,
    get_shipping_corridor_history,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class DeliveryRiskCoordinator:
    """Central multi-agent coordinator for the Delivery Risk Assistant."""

    def __init__(
        self,
        risk_agent: Optional[RiskAgent] = None,
        decision_model: Optional[BaseDecisionModel] = None,
        policy_version: str = "v1.0",
        prompt_template_path: str = "configs/prompts/coordinator_v1.jinja2",
    ):
        self.risk_agent = risk_agent or RiskAgent()
        self.decision_model = decision_model or GeneralPurposeSLM()
        self.policy_version = policy_version
        self.prompt_template_path = prompt_template_path
        
        # Load Jinja template
        template_str = Path(prompt_template_path).read_text(encoding="utf-8")
        self.jinja_template = jinja2.Template(template_str)

    def evaluate_order(
        self,
        order_id: int,
        raw_order_features: Optional[Dict[str, Any]] = None,
    ) -> DeliveryRiskAssessment:
        """Executes full multi-agent risk assessment and recovery decision workflow."""
        start_time = time.perf_counter()
        tool_calls: List[str] = []

        try:
            # Step 1: Tool Call - Order Details
            tool_calls.append("get_order_details")
            order_info = get_order_details(order_id)
            if order_info.get("status") == "error":
                return self._create_error_fallback(
                    order_id=order_id,
                    error_msg=order_info.get("message", "Order details retrieval failed."),
                    tool_calls=tool_calls,
                    start_time=start_time,
                )

            # Step 2: Tool Call - Customer Profile
            customer_id = order_info.get("customer_id", 0)
            tool_calls.append("get_customer_profile")
            customer_info = get_customer_profile(customer_id)

            # Step 3: Tool Call - Shipping Corridor Statistics
            market = order_info.get("market", "USCA")
            shipping_mode = order_info.get("shipping_mode", "Standard Class")
            tool_calls.append("get_shipping_corridor_history")
            corridor_info = get_shipping_corridor_history(market, shipping_mode)

            # Step 4: Tool Call - Active Recovery Policy
            tool_calls.append("get_recovery_policy")
            policy_info = get_recovery_policy(self.policy_version)

            # Step 5: Risk Agent Inference
            # Merge raw features with retrieved operational details
            eval_features = dict(order_info)
            if raw_order_features:
                eval_features.update(raw_order_features)
            eval_features["corridor_delay_risk"] = corridor_info.get("corridor_delay_risk", 0.54)

            risk_out = self.risk_agent.assess_risk(order_id=order_id, feature_dict=eval_features)

            # Step 6: Format Context & Prompt for SLM Decision
            context = {
                "order_id": order_id,
                "risk_score": risk_out.risk_score,
                "risk_level": risk_out.risk_level,
                "order_value": order_info.get("order_value", 0.0),
                "shipping_mode": shipping_mode,
                "customer_id": customer_id,
                "customer_segment": customer_info.get("customer_segment", "Consumer"),
                "customer_impact_severe": customer_info.get("customer_impact_severe", False),
                "corridor_delay_rate": corridor_info.get("corridor_delay_risk", 0.54),
                "tool_findings": f"- Top risk factors: {', '.join(risk_out.top_factors)}\n"
                                 f"- Corridor reliability: {corridor_info.get('reliability_rating', 'MODERATE')}\n"
                                 f"- Customer severe impact: {customer_info.get('customer_impact_severe', False)}",
                "tool_status": "success",
            }

            rendered_prompt = self.jinja_template.render(**context)

            # Step 7: Invoke SLM Decision Engine
            raw_decision = self.decision_model.decide(
                prompt=rendered_prompt,
                context=context,
                policy_version=self.policy_version,
            )

            total_elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            # Step 8: Validate and emit structured output contract
            return DeliveryRiskAssessment(
                order_id=order_id,
                risk_score=risk_out.risk_score,
                risk_level=risk_out.risk_level,
                recommended_action=raw_decision.get("recommended_action", "HUMAN_ESCALATION"),
                supporting_evidence=raw_decision.get("supporting_evidence", []),
                policy_citation=raw_decision.get("policy_citation", f"Policy {self.policy_version}"),
                confidence=float(raw_decision.get("confidence", 0.90)),
                escalate_to_human=bool(raw_decision.get("escalate_to_human", False)),
                tool_calls_executed=tool_calls,
                processing_time_ms=round(total_elapsed_ms, 2),
                model_used=self.decision_model.model_name,
            )

        except Exception as exc:
            logger.exception("Unexpected exception in coordinator workflow for order %d", order_id)
            return self._create_error_fallback(
                order_id=order_id,
                error_msg=f"Unexpected agent failure: {str(exc)}",
                tool_calls=tool_calls,
                start_time=start_time,
            )

    def _create_error_fallback(
        self,
        order_id: int,
        error_msg: str,
        tool_calls: List[str],
        start_time: float,
    ) -> DeliveryRiskAssessment:
        """Deterministic safety fallback ensuring NFR2 Reliability compliance."""
        total_elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return DeliveryRiskAssessment(
            order_id=order_id,
            risk_score=0.50,
            risk_level="HIGH",
            recommended_action="HUMAN_ESCALATION",
            supporting_evidence=[
                f"Agent system exception: {error_msg}",
                "Safety protocol triggered: order escalated to human operations queue.",
            ],
            policy_citation="Policy NFR2: Failures in risk prediction or tool calls clearly reported.",
            confidence=1.0,
            escalate_to_human=True,
            tool_calls_executed=tool_calls,
            processing_time_ms=round(total_elapsed_ms, 2),
            model_used="safety_fallback",
        )

    def apply_human_override(
        self,
        assessment: DeliveryRiskAssessment,
        override_action: str,
        reason: str,
        reviewer_id: str,
    ) -> DeliveryRiskAssessment:
        """FR6: Allows human operations staff to review and override proposed recovery action."""
        assessment_copy = assessment.model_copy(deep=True)
        assessment_copy.human_override = {
            "original_action": assessment.recommended_action,
            "overridden_action": override_action,
            "reason": reason,
            "reviewer_id": reviewer_id,
            "timestamp": time.time(),
        }
        assessment_copy.recommended_action = override_action
        assessment_copy.escalate_to_human = False
        return assessment_copy
