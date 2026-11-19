"""
Unit tests for data leakage prevention and feature schema integrity.
"""

import json
import pytest
import pandas as pd
from src.data.preprocess import LEAKAGE_COLUMNS, NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET_COLUMN


def test_processed_files_exist():
    train_df = pd.read_parquet("data/processed/train.parquet")
    val_df = pd.read_parquet("data/processed/val.parquet")
    test_df = pd.read_parquet("data/processed/test.parquet")
    
    assert len(train_df) > 100000
    assert len(val_df) > 20000
    assert len(test_df) > 20000


def test_no_quarantined_columns_in_splits():
    for split_name in ["train", "val", "test"]:
        df = pd.read_parquet(f"data/processed/{split_name}.parquet")
        for col in LEAKAGE_COLUMNS:
            assert col not in df.columns, f"CRITICAL LEAKAGE: '{col}' found in {split_name} split!"


def test_target_distribution_consistency():
    train_df = pd.read_parquet("data/processed/train.parquet")
    test_df = pd.read_parquet("data/processed/test.parquet")
    
    train_mean = train_df[TARGET_COLUMN].mean()
    test_mean = test_df[TARGET_COLUMN].mean()
    
    # Stratification check (should be within 1% of each other)
    assert abs(train_mean - test_mean) < 0.01


def test_metadata_consistency():
    with open("data/processed/feature_metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
        
    assert meta["target"] == TARGET_COLUMN
    for col in meta["features"]:
        assert col not in LEAKAGE_COLUMNS
