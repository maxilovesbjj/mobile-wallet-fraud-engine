import pytest
import pandas as pd
from src.features.engineering import compute_financial_sensors

def test_flag_vaciado_detection():
    data = pd.DataFrame([{
        "type": "CASH_OUT",
        "amount": 5000.0,
        "oldbalanceOrg": 5000.0,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 5000.0
    }])
    transformed = compute_financial_sensors(data)
    assert transformed["flag_vaciado"].iloc[0] == 1
    assert transformed["error_balance_origen"].iloc[0] == 0.0

def test_error_balance_calculation():
    # Saldo previo 10,000, transfiere 3,000, saldo nuevo deberia ser 7,000 pero dice 8,000
    data = pd.DataFrame([{
        "type": "TRANSFER",
        "amount": 3000.0,
        "oldbalanceOrg": 10000.0,
        "newbalanceOrig": 8000.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 3000.0
    }])
    transformed = compute_financial_sensors(data)
    assert transformed["error_balance_origen"].iloc[0] == 1000.0
    assert transformed["flag_vaciado"].iloc[0] == 0
