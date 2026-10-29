from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import xgboost as xgb
import pandas as pd
import os
from src.features.engineering import compute_financial_sensors

app = FastAPI(
    title="Mobile Wallet AML Engine",
    description="Inferencia en tiempo real para deteccion de layering y cuentas mula.",
    version="1.0.0"
)

MODEL_PATH = "models/xgb_aml_engine.json"
model = None

@app.on_event("startup")
def load_engine():
    global model
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Artefacto {MODEL_PATH} no encontrado. Ejecuta src/models/train.py primero.")
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)

class TransactionPayload(BaseModel):
    type: str = Field(..., description="TRANSFER o CASH_OUT")
    amount: float = Field(..., gt=0)
    oldbalanceOrg: float = Field(..., ge=0)
    newbalanceOrig: float = Field(..., ge=0)
    oldbalanceDest: float = Field(..., ge=0)
    newbalanceDest: float = Field(..., ge=0)

    class Config:
        json_schema_extra = {
            "example": {
                "type": "CASH_OUT",
                "amount": 250000.0,
                "oldbalanceOrg": 250000.0,
                "newbalanceOrig": 0.0,
                "oldbalanceDest": 10000.0,
                "newbalanceDest": 260000.0
            }
        }

class AMLDecision(BaseModel):
    is_sar_alert: bool
    fraud_probability: float
    flag_vaciado: int
    error_balance_origen: float
    error_balance_destino: float

@app.get("/health")
def health():
    return {"status": "ok", "model_ready": model is not None}

@app.post("/predict", response_model=AMLDecision)
def score_transaction(tx: TransactionPayload):
    if tx.type not in ["TRANSFER", "CASH_OUT"]:
        raise HTTPException(
            status_code=400, 
            detail="Solo se evaluan canales vulnerables: TRANSFER o CASH_OUT"
        )

    df_in = pd.DataFrame([tx.dict()])
    df_transformed = compute_financial_sensors(df_in)

    feature_cols = [
        'amount', 'oldbalanceOrg', 'newbalanceOrig', 
        'oldbalanceDest', 'newbalanceDest', 
        'error_balance_origen', 'error_balance_destino', 
        'flag_vaciado', 'type_encoded'
    ]
    
    prob = float(model.predict_proba(df_transformed[feature_cols])[0][1])
    is_alert = bool(prob >= 0.5)

    return AMLDecision(
        is_sar_alert=is_alert,
        fraud_probability=round(prob, 4),
        flag_vaciado=int(df_transformed['flag_vaciado'].iloc[0]),
        error_balance_origen=round(float(df_transformed['error_balance_origen'].iloc[0]), 2),
        error_balance_destino=round(float(df_transformed['error_balance_destino'].iloc[0]), 2)
    )
