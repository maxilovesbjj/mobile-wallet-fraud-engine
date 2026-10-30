from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
import xgboost as xgb
import pandas as pd
import os
import logging
from src.features.engineering import compute_financial_sensors
from src.core.config import settings

# Configuracion de logging de auditoria AML
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("aml_engine")

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.path.exists(settings.MODEL_PATH):
        logger.error(f"Artefacto {settings.MODEL_PATH} no encontrado.")
        raise RuntimeError(f"Artefacto {settings.MODEL_PATH} no encontrado. Ejecuta el pipeline de entrenamiento.")
    
    model = xgb.XGBClassifier()
    model.load_model(settings.MODEL_PATH)
    ml_models["engine"] = model
    logger.info("Modelo AML cargado exitosamente en memoria.")
    yield
    ml_models.clear()

app = FastAPI(
    title="Mobile Wallet AML Engine",
    description="Inferencia en tiempo real para deteccion de layering y cuentas mula.",
    version="1.0.0",
    lifespan=lifespan
)

class TransactionPayload(BaseModel):
    type: str = Field(..., description="TRANSFER o CASH_OUT")
    amount: float = Field(..., gt=0)
    oldbalanceOrg: float = Field(..., ge=0)
    newbalanceOrig: float = Field(..., ge=0)
    oldbalanceDest: float = Field(..., ge=0)
    newbalanceDest: float = Field(..., ge=0)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "type": "CASH_OUT",
                "amount": 250000.0,
                "oldbalanceOrg": 250000.0,
                "newbalanceOrig": 0.0,
                "oldbalanceDest": 10000.0,
                "newbalanceDest": 260000.0
            }
        }
    )

class AMLDecision(BaseModel):
    is_sar_alert: bool
    fraud_probability: float
    flag_vaciado: int
    error_balance_origen: float
    error_balance_destino: float

@app.get("/health")
def health():
    return {"status": "ok", "model_ready": "engine" in ml_models}

@app.post("/predict", response_model=AMLDecision)
def score_transaction(tx: TransactionPayload):
    if tx.type not in ["TRANSFER", "CASH_OUT"]:
        raise HTTPException(
            status_code=400, 
            detail="Solo se evaluan canales vulnerables: TRANSFER o CASH_OUT"
        )

    model = ml_models.get("engine")
    if not model:
        raise HTTPException(status_code=500, detail="Modelo no inicializado en memoria")

    df_in = pd.DataFrame([tx.model_dump()])
    df_transformed = compute_financial_sensors(df_in)

    feature_cols = [
        'amount', 'oldbalanceOrg', 'newbalanceOrig', 
        'oldbalanceDest', 'newbalanceDest', 
        'error_balance_origen', 'error_balance_destino', 
        'flag_vaciado', 'type_encoded'
    ]
    
    prob = float(model.predict_proba(df_transformed[feature_cols])[0][1])
    is_alert = bool(prob >= settings.SAR_ALERT_THRESHOLD)

    # Rastro de auditoria para operaciones sospechosas
    if is_alert:
        logger.warning(
            f"ALERTA SAR DETECTADA | Probabilidad: {prob:.4f} | "
            f"Monto: {tx.amount} | Tipo: {tx.type} | "
            f"Vaciado: {df_transformed['flag_vaciado'].iloc[0]}"
        )

    return AMLDecision(
        is_sar_alert=is_alert,
        fraud_probability=round(prob, 4),
        flag_vaciado=int(df_transformed['flag_vaciado'].iloc[0]),
        error_balance_origen=round(float(df_transformed['error_balance_origen'].iloc[0]), 2),
        error_balance_destino=round(float(df_transformed['error_balance_destino'].iloc[0]), 2)
    )
