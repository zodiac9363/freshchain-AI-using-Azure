"""
FreshChain AI — Phase 7: Automated Tests for ETL Pipeline
Tests schema validation, domain bounds, label normalization, and feature calculations.
"""

import pytest
import pandas as pd
import numpy as np
from etl.validate import validate_sensor_dataset, validate_fao_dataset
from etl.transform import (
    calculate_vapor_pressure_deficit,
    calculate_temp_humidity_index,
    transform_sensor_data
)
from etl.load import create_stratified_splits


def test_sensor_validation_missing_columns():
    df_missing = pd.DataFrame({
        "Fruit": ["Tomato"],
        "Temp": [22],
        "Class": ["Good"]
    })
    is_ok, report, mask = validate_sensor_dataset(df_missing)
    assert not is_ok
    assert report["status"] == "FAILED_SCHEMA"
    assert "Humid (%)" in report["missing_columns"]


def test_sensor_validation_out_of_bounds():
    df_oob = pd.DataFrame({
        "Fruit": ["Tomato", "Banana"],
        "Temp": [100, 24],  # 100 C is physical violation
        "Humid (%)": [90, 85],
        "Light (Fux)": [10.0, 12.0],
        "CO2 (pmm)": [350, 300],
        "Class": ["Good", "Bad"]
    })
    is_ok, report, mask = validate_sensor_dataset(df_oob)
    assert not is_ok
    assert report["range_violations"]["Temp"]["violations_count"] == 1
    assert mask.iloc[0] is False or mask.iloc[0] == 0
    assert mask.iloc[1] is True or mask.iloc[1] == 1


def test_label_normalization_bad_and_bad():
    raw_df = pd.DataFrame({
        "Fruit": ["Tomato", "Orange", "Banana"],
        "Temp": [24.0, 22.0, 25.0],
        "Humid (%)": [90.0, 85.0, 92.0],
        "Light (Fux)": [12.0, 8.0, 15.0],
        "CO2 (pmm)": [360.0, 280.0, 410.0],
        "Class": ["Good", "Bad", "BAD"]
    })
    mask = pd.Series([True, True, True])
    clean_df, report = transform_sensor_data(raw_df, mask)
    
    # Good should be 0, both Bad and BAD should normalize to 1
    assert clean_df["is_spoilage"].tolist() == [0, 1, 1]
    assert report["class_balance"]["spoiled_count"] == 2
    assert report["class_balance"]["fresh_count"] == 1


def test_biophysical_feature_engineering():
    temp = pd.Series([25.0, 20.0])
    humid = pd.Series([90.0, 80.0])
    
    vpd = calculate_vapor_pressure_deficit(temp, humid)
    assert len(vpd) == 2
    assert (vpd >= 0.0).all()
    # At 80% humidity, VPD must be higher than at 90% humidity
    assert vpd.iloc[1] > vpd.iloc[0]
    
    thi = calculate_temp_humidity_index(temp, humid)
    assert len(thi) == 2
    assert not thi.isnull().any()


def test_stratified_split_ratios():
    # Synthetic dataset with 200 records
    fruits = ["Tomato", "Orange", "Pineapple", "Banana"] * 50
    classes = [0, 1] * 100
    df = pd.DataFrame({
        "fruit": fruits,
        "is_spoilage": classes,
        "temperature_c": [24.0] * 200,
        "humidity_pct": [90.0] * 200,
        "light_lux": [10.0] * 200,
        "co2_ppm": [320.0] * 200
    })
    
    splits = create_stratified_splits(df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_state=42)
    assert len(splits["train"]) == 140
    assert len(splits["validation"]) == 30
    assert len(splits["test"]) == 30
    
    # Check spoilage rate preserved in each split
    for name, split_df in splits.items():
        rate = split_df["is_spoilage"].mean()
        assert abs(rate - 0.50) < 0.05
