import os
import duckdb
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

def train_and_export():
    parquet_path = "data/processed/paysim_filtered.parquet"
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"No se encontro {parquet_path}. Ejecuta primero el pipeline de datos.")

    print("--- 1. Extrayendo datos procesados y calculando sensores ---")
    con = duckdb.connect()
    df_features = con.execute(f"""
        SELECT 
            step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig, 
            nameDest, oldbalanceDest, newbalanceDest, isFraud,
            ABS((oldbalanceOrg - newbalanceOrig) - amount) AS error_balance_origen,
            ABS((newbalanceDest - oldbalanceDest) - amount) AS error_balance_destino,
            CASE WHEN amount = oldbalanceOrg AND amount > 0 THEN 1 ELSE 0 END AS flag_vaciado
        FROM '{parquet_path}'
    """).df()

    print("--- 2. Preparando tensores ---")
    df_features['type_encoded'] = df_features['type'].map({'TRANSFER': 0, 'CASH_OUT': 1})
    X = df_features.drop(columns=['step', 'type', 'nameOrig', 'nameDest', 'isFraud'])
    y = df_features['isFraud']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    peso_fraude = y_train.value_counts()[0] / y_train.value_counts()[1]
    print(f"Multiplicador scale_pos_weight: {peso_fraude:.2f}")

    print("--- 3. Entrenando motor XGBoost ---")
    motor_xgb = xgb.XGBClassifier(
        scale_pos_weight=peso_fraude,
        random_state=42,
        n_jobs=-1
    )
    motor_xgb.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    model_output = "models/xgb_aml_engine.json"
    motor_xgb.save_model(model_output)
    print(f"--- 4. Motor exportado exitosamente en: {model_output} ---")

if __name__ == "__main__":
    train_and_export()
