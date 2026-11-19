"""
Unit tests for delivery risk prediction model.
"""

import os
import pytest
from src.models.risk_classifier import DeliveryRiskPredictor


@pytest.fixture
def predictor():
    pred = DeliveryRiskPredictor()
    assert pred.model is not None, "Model failed to load."
    return pred


def test_model_files_exist():
    assert os.path.exists("models/risk_xgb_v1.json")
    assert os.path.exists("models/artifacts/preprocessor.joblib")
    assert os.path.exists("models/artifacts/model_metadata.json")


def test_prediction_output_schema(predictor):
    sample_order = {
        "Days for shipment (scheduled)": 4,
        "Benefit per order": 50.0,
        "Sales per customer": 250.0,
        "Order Item Discount": 10.0,
        "Order Item Discount Rate": 0.05,
        "Order Item Product Price": 150.0,
        "Order Item Profit Ratio": 0.25,
        "Order Item Quantity": 2,
        "Sales": 300.0,
        "Order Item Total": 285.0,
        "Order Profit Per Order": 50.0,
        "Product Price": 150.0,
        "order_hour": 14,
        "order_day_of_week": 2,
        "order_month": 5,
        "corridor_delay_risk": 0.55,
        "Type": "DEBIT",
        "Shipping Mode": "Standard Class",
        "Customer Segment": "Consumer",
        "Market": "Pacific Asia",
        "Order Region": "Southeast Asia",
        "Department Name": "Fitness",
        "Category Name": "Sporting Goods",
    }

    res = predictor.predict_one(sample_order)
    assert "risk_score" in res
    assert 0.0 <= res["risk_score"] <= 1.0
    assert res["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
    assert isinstance(res["is_delayed_risk"], bool)
    assert res["latency_ms"] < 100.0, "NFR1 Latency violated (> 100ms)"
    assert len(res["top_factors"]) > 0


def test_missing_features_handled_gracefully(predictor):
    empty_order = {}
    res = predictor.predict_one(empty_order)
    assert "risk_score" in res
    assert 0.0 <= res["risk_score"] <= 1.0
