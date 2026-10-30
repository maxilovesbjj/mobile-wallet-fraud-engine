import os
import json
import duckdb
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from src.features.engineering import compute_financial_sensors
from src.core.config import settings

def run_training_pipeline():
    parquet_path = settings.DATA_PATH
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Archivo no encontrado: {parquet_path}")

    print(f"1. Cargando datos desde {parquet_path}...")
    con = duckdb.connect()
    df_raw = con.execute(f"SELECT * FROM '{parquet_path}'").df()

    print("2. Calculando sensores a traves de src.features...")
    df_transformed = compute_financial_sensors(df_raw)

    feature_cols = [
        'amount', 'oldbalanceOrg', 'newbalanceOrig', 
        'oldbalanceDest', 'newbalanceDest', 
        'error_balance_origen', 'error_balance_destino', 
        'flag_vaciado', 'type_encoded'
    ]
    
    X = df_transformed[feature_cols]
    y = df_transformed['isFraud']

    print("3. Split estratificado (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    peso_fraude = float(y_train.value_counts()[0] / y_train.value_counts()[1])
    print(f"   scale_pos_weight: {peso_fraude:.2f}")

    print("4. Entrenando XGBoost...")
    model = xgb.XGBClassifier(
        scale_pos_weight=peso_fraude,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss"
    )
    model.fit(X_train, y_train)

    print("5. Evaluando modelo sobre conjunto de prueba...")
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    matrix = confusion_matrix(y_test, y_pred).tolist()

    # Asegurar que el directorio del modelo exista
    os.makedirs(os.path.dirname(settings.MODEL_PATH), exist_ok=True)
    
    model.save_model(settings.MODEL_PATH)

    metrics_payload = {
        "scale_pos_weight": peso_fraude,
        "confusion_matrix": {
            "tn": matrix[0][0], "fp": matrix[0][1],
            "fn": matrix[1][0], "tp": matrix[1][1]
        },
        "classification_report": report
    }

    with open(settings.METRICS_PATH, "w") as f:
        json.dump(metrics_payload, f, indent=4)

    print(f"Entrenamiento finalizado. Artefactos guardados en {os.path.dirname(settings.MODEL_PATH)}/")

if __name__ == "__main__":
    run_training_pipeline()