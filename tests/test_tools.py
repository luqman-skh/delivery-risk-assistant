"""
Unit tests for operational data tools suite.
"""

import pytest
import pandas as pd
from src.tools.operational_tools import (
    get_order_details,
    get_customer_profile,
    get_shipping_corridor_history,
    get_recovery_policy,
)


@pytest.fixture(scope="module")
def sample_ids():
    test_df = pd.read_parquet("data/processed/test.parquet")
    row = test_df.iloc[0]
    return int(row["Order Id"]), int(row["Customer Id"]), str(row["Market"]), str(row["Shipping Mode"])


def test_get_order_details_valid(sample_ids):
    order_id, _, _, _ = sample_ids
    res = get_order_details(order_id)
    assert res["status"] == "success"
    assert res["order_id"] == order_id
    assert "order_value" in res
    assert "shipping_mode" in res
    assert "scheduled_days" in res


def test_get_order_details_invalid():
    res = get_order_details(999999999)
    assert res["status"] == "error"
    assert res["error_code"] == "ORDER_NOT_FOUND"


def test_get_customer_profile_valid(sample_ids):
    _, customer_id, _, _ = sample_ids
    res = get_customer_profile(customer_id)
    assert res["status"] == "success"
    assert res["customer_id"] == customer_id
    assert res["customer_segment"] in ["Consumer", "Corporate", "Home Office"]
    assert "customer_impact_severe" in res


def test_get_customer_profile_invalid():
    res = get_customer_profile(999999999)
    assert res["status"] == "error"
    assert res["error_code"] == "CUSTOMER_NOT_FOUND"


def test_get_shipping_corridor_history(sample_ids):
    _, _, market, shipping_mode = sample_ids
    res = get_shipping_corridor_history(market, shipping_mode)
    assert res["status"] == "success"
    assert "corridor_delay_risk" in res
    assert res["reliability_rating"] in ["HIGH_RELIABILITY", "MODERATE", "ELEVATED_RISK"]


def test_get_recovery_policy():
    res = get_recovery_policy("v1.0")
    assert res["status"] == "success"
    assert "actions" in res
    assert "CONTINUE_MONITORING" in res["actions"]
    assert "PRIORITIZE_SHIPMENT" in res["actions"]
    assert "CONTACT_LOGISTICS" in res["actions"]
    assert "NOTIFY_CUSTOMER" in res["actions"]
    assert "HUMAN_ESCALATION" in res["actions"]
