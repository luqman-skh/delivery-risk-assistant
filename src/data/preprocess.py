"""
Data Preprocessing & Leakage Prevention Pipeline for Delivery Risk Assistant.
Enforces strict quarantine on target leakage features (e.g. Delivery Status, Days for shipping (real)).
"""

import json
import logging
import os
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Mandatory Quarantine: Columns that MUST NOT be used as inputs
LEAKAGE_COLUMNS = [
    "Delivery Status",
    "Days for shipping (real)",
    "shipping date (DateOrders)",
    "Order Status",
]

# Irrelevant / PII / Redundant columns
DROP_COLUMNS = [
    "Customer Email",
    "Customer Password",
    "Customer Fname",
    "Customer Lname",
    "Customer Street",
    "Product Description",
    "Product Image",
    "Order Zipcode",
    "Product Status",
]

NUMERIC_FEATURES = [
    "Days for shipment (scheduled)",
    "Benefit per order",
    "Sales per customer",
    "Order Item Discount",
    "Order Item Discount Rate",
    "Order Item Product Price",
    "Order Item Profit Ratio",
    "Order Item Quantity",
    "Sales",
    "Order Item Total",
    "Order Profit Per Order",
    "Product Price",
    "order_hour",
    "order_day_of_week",
    "order_month",
]

CATEGORICAL_FEATURES = [
    "Type",
    "Shipping Mode",
    "Customer Segment",
    "Market",
    "Order Region",
    "Department Name",
    "Category Name",
]

TARGET_COLUMN = "Late_delivery_risk"

METADATA_COLUMNS = [
    "Order Id",
    "Customer Id",
    "Customer City",
    "Customer Country",
    "Customer State",
    "Order City",
    "Order Country",
    "Order State",
    "Product Card Id",
    "Product Name",
    "order date (DateOrders)",
]


def load_raw_dataset(csv_path: str) -> pd.DataFrame:
    """Loads raw dataset handling encoding."""
    logger.info("Loading raw dataset from %s", csv_path)
    for encoding in ["utf-8", "latin-1", "ISO-8859-1", "cp1252"]:
        try:
            df = pd.read_csv(csv_path, encoding=encoding)
            logger.info("Successfully loaded dataset with %s encoding. Shape: %s", encoding, df.shape)
            return df
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Failed to decode {csv_path} with supported encodings.")


def audit_leakage(columns: List[str]) -> None:
    """Strictly asserts that none of the quarantined leakage columns are present."""
    leaked = [col for col in columns if col in LEAKAGE_COLUMNS]
    if leaked:
        raise ValueError(f"CRITICAL LEAKAGE DETECTED! Quarantined columns found in feature set: {leaked}")
    logger.info("Leakage audit passed: 0 forbidden columns detected.")


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts date components and derives leak-free operational indicators."""
    df = df.copy()
    
    # Parse order timestamp
    order_dt = pd.to_datetime(df["order date (DateOrders)"], errors="coerce")
    df["order_hour"] = order_dt.dt.hour.fillna(12).astype(int)
    df["order_day_of_week"] = order_dt.dt.day_of_week.fillna(0).astype(int)
    df["order_month"] = order_dt.dt.month.fillna(1).astype(int)

    # Impute missing values for numerics
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Clean categoricals
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna("Unknown").str.strip()

    return df


def compute_corridor_risk(
    train_df: pd.DataFrame,
    group_cols: List[str] = ["Shipping Mode", "Market"]
) -> pd.DataFrame:
    """Computes historical late delivery rate per shipping corridor on training data only."""
    risk_rates = (
        train_df.groupby(group_cols)[TARGET_COLUMN]
        .agg(["count", "mean"])
        .reset_index()
        .rename(columns={"mean": "corridor_delay_risk", "count": "corridor_order_count"})
    )
    # Bayesian smoothing: pull towards overall mean for small cohorts
    global_mean = train_df[TARGET_COLUMN].mean()
    smoothing_factor = 20
    risk_rates["corridor_delay_risk"] = (
        (risk_rates["corridor_delay_risk"] * risk_rates["corridor_order_count"] + global_mean * smoothing_factor)
        / (risk_rates["corridor_order_count"] + smoothing_factor)
    )
    return risk_rates


def run_pipeline(
    raw_path: str = "data/DataCoSupplyChainDataset.csv",
    output_dir: str = "data/processed",
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Runs complete end-to-end data preparation pipeline."""
    os.makedirs(output_dir, exist_ok=True)
    raw_df = load_raw_dataset(raw_path)

    # Initial check
    if TARGET_COLUMN not in raw_df.columns:
        raise KeyError(f"Target column '{TARGET_COLUMN}' not found in raw dataset.")

    # Target distribution
    target_dist = raw_df[TARGET_COLUMN].value_counts(normalize=True).to_dict()
    logger.info("Target distribution: %s", target_dist)

    # Feature extraction
    processed_df = extract_features(raw_df)

    # Split dataset stratifying by target
    train_val_df, test_df = train_test_split(
        processed_df,
        test_size=test_size,
        random_state=random_state,
        stratify=processed_df[TARGET_COLUMN],
    )

    relative_val_size = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val_size,
        random_state=random_state,
        stratify=train_val_df[TARGET_COLUMN],
    )

    # Compute corridor risk on TRAIN ONLY and map to all sets
    corridor_stats = compute_corridor_risk(train_df, group_cols=["Shipping Mode", "Market"])
    global_risk = float(train_df[TARGET_COLUMN].mean())

    def apply_corridor(df_in: pd.DataFrame) -> pd.DataFrame:
        df_out = df_in.merge(
            corridor_stats[["Shipping Mode", "Market", "corridor_delay_risk"]],
            on=["Shipping Mode", "Market"],
            how="left",
        )
        df_out["corridor_delay_risk"] = df_out["corridor_delay_risk"].fillna(global_risk)
        return df_out

    train_df = apply_corridor(train_df)
    val_df = apply_corridor(val_df)
    test_df = apply_corridor(test_df)

    # Prepare final feature lists
    final_features = NUMERIC_FEATURES + ["corridor_delay_risk"] + CATEGORICAL_FEATURES
    audit_leakage(final_features)

    # Save artifact files
    train_path = os.path.join(output_dir, "train.parquet")
    val_path = os.path.join(output_dir, "val.parquet")
    test_path = os.path.join(output_dir, "test.parquet")
    meta_path = os.path.join(output_dir, "feature_metadata.json")

    # Keep metadata columns for agent and tool lookups
    keep_columns = list(set(final_features + [TARGET_COLUMN] + METADATA_COLUMNS))
    train_df = train_df[keep_columns]
    val_df = val_df[keep_columns]
    test_df = test_df[keep_columns]

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)

    metadata = {
        "target": TARGET_COLUMN,
        "features": final_features,
        "numeric_features": NUMERIC_FEATURES + ["corridor_delay_risk"],
        "categorical_features": CATEGORICAL_FEATURES,
        "metadata_columns": METADATA_COLUMNS,
        "quarantined_leakage_columns": LEAKAGE_COLUMNS,
        "global_risk_prior": global_risk,
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "class_balance": target_dist,
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Saved train (%d), val (%d), test (%d) to %s", len(train_df), len(val_df), len(test_df), output_dir)
    return train_df, val_df, test_df


if __name__ == "__main__":
    run_pipeline()
