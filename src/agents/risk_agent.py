"""
Risk Assessment Agent (WP 1.5).
Specialized agent that prepares feature payloads and queries the calibrated risk model.
"""

import logging
from typing import Any, Dict
from src.agents.schemas import RiskPredictionOutput
from src.models.risk_classifier import DeliveryRiskPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class RiskAgent:
    """Agent specialized in quantifying delivery delay risk."""

    def __init__(self, predictor: DeliveryRiskPredictor = None):
        self.predictor = predictor or DeliveryRiskPredictor()

    def assess_risk(self, order_id: int, feature_dict: Dict[str, Any]) -> RiskPredictionOutput:
        """Evaluates delivery delay probability for an order."""
        res = self.predictor.predict_one(feature_dict)
        return RiskPredictionOutput(
            order_id=order_id,
            risk_score=res["risk_score"],
            risk_level=res["risk_level"],
            is_delayed_risk=res["is_delayed_risk"],
            latency_ms=res["latency_ms"],
            top_factors=res["top_factors"],
        )
