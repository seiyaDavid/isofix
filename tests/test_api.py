"""
API Testing Module

This module contains tests for the FastAPI endpoints including:
    - Input validation
    - Anomaly detection functionality
    - Error handling
    - Response format validation
"""

import pytest
from fastapi.testclient import TestClient
from api import app
import pandas as pd
import os
from datetime import datetime, timedelta

client = TestClient(app)


@pytest.fixture
def sample_data():
    """
    Create sample data for testing with a clear anomaly.
    """
    # Create dates
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
    df = pd.DataFrame({"Date": dates})

    # Create STOCK_1 with normal values and one extreme anomaly at the end
    normal_values = [10.0] * 98  # Start with stable values
    normal_values.extend([10.0, 500.0])  # Add extreme jump at the end (400% increase)
    df["STOCK_1"] = normal_values

    # Create STOCK_2 with small, normal variations
    df["STOCK_2"] = [10.0 + i * 0.0001 for i in range(100)]  # Very gradual increase

    return df


def test_detect_anomalies_endpoint(sample_data):
    """
    Test the detect_anomalies endpoint.

    Verifies:
        - Successful response (200)
        - Correct response format
        - Anomaly detection accuracy
        - Proper inclusion of anomalous stocks only
    """
    # Save sample data to temp file
    sample_data.to_csv("test_data.csv", index=False)

    with open("test_data.csv", "rb") as f:
        response = client.post("/detect_anomalies/", files={"file": f})

    os.remove("test_data.csv")

    assert response.status_code == 200
    data = response.json()

    # Print response for debugging
    print(f"API Response: {data}")

    assert "date" in data
    assert "anomalous_stocks" in data
    assert (
        "STOCK_1" in data["anomalous_stocks"]
    ), "STOCK_1 should be detected as anomalous"
    assert (
        "STOCK_2" not in data["anomalous_stocks"]
    ), "STOCK_2 should not be detected as anomalous"
    assert data["anomalous_stocks"]["STOCK_1"]["is_anomaly"] == True


def test_force_retrain_endpoint(sample_data):
    """Test the force_retrain endpoint"""
    sample_data.to_csv("test_data.csv", index=False)

    with open("test_data.csv", "rb") as f:
        response = client.post("/force_retrain/", files={"file": f})

    os.remove("test_data.csv")

    assert response.status_code == 200
    data = response.json()
    assert "date" in data
    assert "anomalous_stocks" in data


def test_invalid_data_format():
    """
    Test handling of invalid data format.

    Verifies:
        - 400 response for missing Date column
        - Appropriate error message
    """
    df = pd.DataFrame({"Wrong_Column": [1, 2, 3]})
    df.to_csv("test_data.csv", index=False)

    with open("test_data.csv", "rb") as f:
        response = client.post("/detect_anomalies/", files={"file": f})

    os.remove("test_data.csv")
    assert response.status_code == 400
    assert "must contain a 'Date' column" in response.json()["detail"]


def test_empty_data():
    """Test handling of empty data"""
    df = pd.DataFrame({"Date": [], "STOCK_1": []})
    df.to_csv("test_data.csv", index=False)

    with open("test_data.csv", "rb") as f:
        response = client.post("/detect_anomalies/", files={"file": f})

    os.remove("test_data.csv")
    assert response.status_code == 400
    assert "Empty data provided" in response.json()["detail"]


def test_no_stock_columns():
    """Test handling of data with no stock columns"""
    df = pd.DataFrame({"Date": [1, 2, 3]})
    df.to_csv("test_data.csv", index=False)

    with open("test_data.csv", "rb") as f:
        response = client.post("/detect_anomalies/", files={"file": f})

    os.remove("test_data.csv")
    assert response.status_code == 400
    assert "No stock columns found" in response.json()["detail"]
