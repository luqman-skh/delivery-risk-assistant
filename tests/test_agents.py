"""
Unit tests for multi-agent coordinator and subagents.
"""

import pytest
import pandas as pd
from src.agents.coordinator import DeliveryRiskCoordinator
from src.agents.risk_agent import RiskAgent
from src.agents.schemas import DeliveryRiskAssessment


@pytest.fixture(scope="module")
def sample_order():
    test_df = pd.read_parquet("data/processed/test.parquet")
    row = test_df.iloc[0]
    return int(row["Order Id"]), row.to_dict()


def test_coordinator_end_to_end(sample_order):
    order_id, row_dict = sample_order
    coordinator = DeliveryRiskCoordinator(policy_version="v1.0")
    
    assessment = coordinator.evaluate_order(order_id=order_id, raw_order_features=row_dict)
    
    assert isinstance(assessment, DeliveryRiskAssessment)
    assert assessment.order_id == order_id
    assert 0.0 <= assessment.risk_score <= 1.0
    assert assessment.risk_level in ["LOW", "MEDIUM", "HIGH"]
    assert assessment.recommended_action in [
        "CONTINUE_MONITORING",
        "CONTACT_LOGISTICS",
        "PRIORITIZE_SHIPMENT",
        "NOTIFY_CUSTOMER",
        "HUMAN_ESCALATION",
    ]
    assert len(assessment.supporting_evidence) > 0
    assert len(assessment.tool_calls_executed) >= 3
    assert assessment.processing_time_ms < 3500.0, "NFR1 End-to-end response time exceeded 3.5s"


def test_coordinator_error_handling_nfr2():
    coordinator = DeliveryRiskCoordinator(policy_version="v1.0")
    # Non-existent order ID
    assessment = coordinator.evaluate_order(order_id=99999999)
    
    assert assessment.recommended_action == "HUMAN_ESCALATION"
    assert assessment.escalate_to_human is True
    assert "could not be found" in assessment.supporting_evidence[0] or "failed" in assessment.supporting_evidence[0]


def test_human_override_fr6(sample_order):
    order_id, row_dict = sample_order
    coordinator = DeliveryRiskCoordinator(policy_version="v1.0")
    assessment = coordinator.evaluate_order(order_id=order_id, raw_order_features=row_dict)
    
    overridden = coordinator.apply_human_override(
        assessment=assessment,
        override_action="PRIORITIZE_SHIPMENT",
        reason="VIP customer contacted executive desk directly.",
        reviewer_id="agent_sarah_104",
    )
    
    assert overridden.recommended_action == "PRIORITIZE_SHIPMENT"
    assert overridden.human_override is not None
    assert overridden.human_override["reviewer_id"] == "agent_sarah_104"
    assert overridden.escalate_to_human is False
