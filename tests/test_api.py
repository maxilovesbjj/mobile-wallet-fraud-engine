import pytest
from fastapi.testclient import TestClient
from src.api.main import app

@pytest.fixture
def client():
    # El bloque 'with' asegura que el lifespan (carga del modelo) se ejecute
    with TestClient(app) as c:
        yield c

def test_predict_mule_account_cashout(client):
    """Tipologia AML: Vaciado total de cuenta mula (Layering -> Integration)"""
    payload = {
        "type": "CASH_OUT",
        "amount": 500000.0,
        "oldbalanceOrg": 500000.0,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 10000.0,
        "newbalanceDest": 510000.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["flag_vaciado"] == 1
    assert "is_sar_alert" in data
    assert "fraud_probability" in data

def test_predict_legitimate_transfer(client):
    """Comportamiento transaccional normal: Transferencia con saldo remanente"""
    payload = {
        "type": "TRANSFER",
        "amount": 1500.0,
        "oldbalanceOrg": 5000.0,
        "newbalanceOrig": 3500.0,
        "oldbalanceDest": 2000.0,
        "newbalanceDest": 3500.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["flag_vaciado"] == 0

def test_reject_non_vulnerable_channels(client):
    """Regulacion AML: El motor solo debe auditar canales de alto riesgo"""
    payload = {
        "type": "PAYMENT",
        "amount": 100.0,
        "oldbalanceOrg": 100.0,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 100.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    assert "Solo se evaluan canales vulnerables" in response.json()["detail"]
