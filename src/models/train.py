import os
import json
import shutil
from datetime import datetime
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

    os.makedirs(os.path.dirname(settings.MODEL_PATH), exist_ok=True)
    
    # 6. Versionado del artefacto (guarda version historica y actualiza 'latest')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    versioned_model_path = settings.MODEL_PATH.replace(".json", f"_{timestamp}.json")
    
    model.save_model(versioned_model_path)
    shutil.copy(versioned_model_path, settings.MODEL_PATH)

    metrics_payload = {
        "timestamp": timestamp,
        "scale_pos_weight": peso_fraude,
        "confusion_matrix": {
            "tn": matrix[0][0], "fp": matrix[0][1],
            "fn": matrix[1][0], "tp": matrix[1][1]
        },
        "classification_report": report
    }

    versioned_metrics_path = settings.METRICS_PATH.replace(".json", f"_{timestamp}.json")
    with open(versioned_metrics_path, "w") as f:
        json.dump(metrics_payload, f, indent=4)
    shutil.copy(versioned_metrics_path, settings.METRICS_PATH)

    print(f"Entrenamiento finalizado. Modelo versionado en: {versioned_model_path}")

if __name__ == "__main__":
    run_training_pipeline()
