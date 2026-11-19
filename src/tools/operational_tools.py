"""
Operational Data Tools Suite (WP 1.3).
Provides deterministic, audited microservice tools for the Multi-Agent Assistant:
- Order Details
- Customer Profile
- Shipping Corridor History
- Recovery Policy Retrieval
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Global cache for dataset indexing
_DATA_CACHE: Optional[pd.DataFrame] = None
_CUSTOMER_CACHE: Optional[pd.DataFrame] = None
_CORRIDOR_CACHE: Optional[pd.DataFrame] = None


def _load_data_cache() -> pd.DataFrame:
    global _DATA_CACHE
    if _DATA_CACHE is None:
        data_path = "data/processed/test.parquet"
        if not os.path.exists(data_path):
            data_path = "data/processed/train.parquet"
        _DATA_CACHE = pd.read_parquet(data_path)
    return _DATA_CACHE


def get_order_details(order_id: int) -> Dict[str, Any]:
    """Retrieves operational details for a specific order.
    
    Args:
        order_id: Numeric unique identifier for the order.
        
    Returns:
        Dictionary with order items, scheduled transit days, shipping mode, and order value.
    """
    df = _load_data_cache()
    matches = df[df["Order Id"] == order_id]
    
    if matches.empty:
        # Fallback to train set if not found in test set
        train_path = "data/processed/train.parquet"
        if os.path.exists(train_path):
            train_df = pd.read_parquet(train_path)
            matches = train_df[train_df["Order Id"] == order_id]
            
    if matches.empty:
        return {
            "status": "error",
            "error_code": "ORDER_NOT_FOUND",
            "message": f"Order ID {order_id} could not be found in active operational database.",
        }

    row = matches.iloc[0]
    total_order_value = float(row.get("Order Item Total", 0.0) or row.get("Sales", 0.0))
    scheduled_days = int(row.get("Days for shipment (scheduled)", 3))
    
    return {
        "status": "success",
        "order_id": int(order_id),
        "customer_id": int(row.get("Customer Id", 0)),
        "order_value": round(total_order_value, 2),
        "shipping_mode": str(row.get("Shipping Mode", "Standard Class")),
        "scheduled_days": scheduled_days,
        "product_name": str(row.get("Product Name", "General Item")),
        "department_name": str(row.get("Department Name", "Apparel")),
        "order_country": str(row.get("Order Country", "United States")),
        "order_region": str(row.get("Order Region", "North America")),
        "market": str(row.get("Market", "USCA")),
        "order_hour": int(row.get("order_hour", 12)),
        "order_day_of_week": int(row.get("order_day_of_week", 0)),
    }


def get_customer_profile(customer_id: int) -> Dict[str, Any]:
    """Retrieves customer tier, location, and historical fulfillment profile.
    
    Args:
        customer_id: Numeric unique customer ID.
        
    Returns:
        Dictionary with customer segment, geography, and risk indicators.
    """
    df = _load_data_cache()
    matches = df[df["Customer Id"] == customer_id]
    
    if matches.empty:
        train_path = "data/processed/train.parquet"
        if os.path.exists(train_path):
            train_df = pd.read_parquet(train_path)
            matches = train_df[train_df["Customer Id"] == customer_id]
            
    if matches.empty:
        return {
            "status": "error",
            "error_code": "CUSTOMER_NOT_FOUND",
            "message": f"Customer ID {customer_id} does not exist in CRM database.",
        }

    row = matches.iloc[0]
    customer_segment = str(row.get("Customer Segment", "Consumer"))
    
    # Calculate customer delay incidence across order history
    historical_orders_count = len(matches)
    delay_risk_history = float(matches["corridor_delay_risk"].mean() if "corridor_delay_risk" in matches.columns else 0.50)
    
    # High impact if Corporate segment or multiple delays
    severe_impact = customer_segment in ["Corporate", "Home Office"] and delay_risk_history > 0.55

    return {
        "status": "success",
        "customer_id": int(customer_id),
        "customer_segment": customer_segment,
        "customer_city": str(row.get("Customer City", "Unknown")),
        "customer_state": str(row.get("Customer State", "Unknown")),
        "customer_country": str(row.get("Customer Country", "Unknown")),
        "historical_orders_count": historical_orders_count,
        "historical_corridor_risk": round(delay_risk_history, 3),
        "customer_impact_severe": severe_impact,
    }


def get_shipping_corridor_history(
    market: str,
    shipping_mode: str,
    order_region: Optional[str] = None
) -> Dict[str, Any]:
    """Retrieves historical fulfillment reliability for a given market corridor and mode.
    
    Args:
        market: Market cluster (e.g. Pacific Asia, Europe, USCA).
        shipping_mode: Shipping mode (Standard Class, First Class, Second Class, Same Day).
        order_region: Regional delivery destination.
        
    Returns:
        Dictionary with historical delay rate and volume benchmarks.
    """
    df = _load_data_cache()
    filtered = df[(df["Market"] == market) & (df["Shipping Mode"] == shipping_mode)]
    
    if filtered.empty:
        return {
            "status": "success",
            "market": market,
            "shipping_mode": shipping_mode,
            "sample_size": 0,
            "corridor_delay_risk": 0.54,  # Global prior fallback
            "reliability_rating": "MODERATE",
        }

    corridor_delay_rate = float(filtered["corridor_delay_risk"].mean() if "corridor_delay_risk" in filtered.columns else 0.54)
    sample_size = len(filtered)
    
    if corridor_delay_rate < 0.40:
        rating = "HIGH_RELIABILITY"
    elif corridor_delay_rate <= 0.60:
        rating = "MODERATE"
    else:
        rating = "ELEVATED_RISK"

    return {
        "status": "success",
        "market": market,
        "shipping_mode": shipping_mode,
        "sample_size": sample_size,
        "corridor_delay_risk": round(corridor_delay_rate, 4),
        "reliability_rating": rating,
    }


def get_recovery_policy(policy_version: str = "v1.0") -> Dict[str, Any]:
    """Retrieves operational order recovery policy rules.
    
    Args:
        policy_version: Version of the policy ('v1.0' or 'v2.0').
        
    Returns:
        Dictionary with policy specifications and action conditions.
    """
    version_file_map = {
        "v1.0": "configs/policies/policy_v1.yaml",
        "v2.0": "configs/policies/policy_v2.yaml",
    }
    
    policy_path = version_file_map.get(policy_version, "configs/policies/policy_v1.yaml")
    if not os.path.exists(policy_path):
        return {
            "status": "error",
            "error_code": "POLICY_NOT_FOUND",
            "message": f"Policy specification '{policy_version}' not found at {policy_path}.",
        }

    with open(policy_path, "r", encoding="utf-8") as f:
        policy_data = yaml.safe_load(f)

    return {
        "status": "success",
        "policy_version": policy_data.get("version", policy_version),
        "name": policy_data.get("name"),
        "actions": policy_data.get("actions", {}),
    }
