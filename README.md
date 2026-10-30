# Mobile Wallet AML Engine: Detección de Layering y Cuentas Mula

Motor de detección en tiempo real para identificar transacciones de billeteras móviles que exhiben el patrón de *layering* típico de lavado de activos — específicamente el ciclo `TRANSFER` → `CASH_OUT` usado para mover e integrar fondos ilícitos fuera del sistema financiero —, priorizando la captura de cuentas mula (usadas para vaciar saldos) sobre el fraude transaccional genérico.

No modela relaciones topológicas entre cuentas (la componente de red se aborda en análisis de grafos posteriores); se enfoca estrictamente en la señal a nivel de transacción individual y dinámica de balance. El uso de validadores lógicos y árboles de decisión (XGBoost) garantiza un bajo costo computacional, optimizado para *scoring* en tiempo real con baja latencia.

## Arquitectura y Stack Tecnológico
* **Data Processing & Pipelines:** DuckDB, Parquet
* **Machine Learning & Explainability:** XGBoost, Scikit-Learn, SHAP
* **Serving & API:** FastAPI, Pydantic v2, Uvicorn
* **Análisis Exploratorio:** Pandas, Jupyter
* **Calidad:** Pytest

## Enfoque de Ingeniería

1. **Out-of-Core Processing (DuckDB):**
   El dataset crudo contiene +6.3 millones de transacciones. En lugar de saturar la memoria RAM cargando todo en Pandas, se implementó DuckDB como motor SQL en memoria para diagnosticar la fuga. Se demostró que el 100% del fraude ocurre en solo 2 canales (`TRANSFER` y `CASH_OUT`), permitiendo reducir el caudal de entrenamiento a 2.7M de registros en formato columnar Parquet.

2. **Feature Engineering (Física de Balances):**
   En lugar de depender de montos brutos, se construyeron sensores matemáticos que miden la discrepancia contable entre saldos iniciales, finales y el monto transferido, delatando los vaciados de saldos típicos de cuentas mula. Esta lógica vive en `src/features/engineering.py` y es compartida tanto por el pipeline de entrenamiento como por la API de inferencia, para garantizar que el modelo vea exactamente los mismos features en ambos contextos (evita *train/serve skew*).

3. **Cost-Sensitive Learning (Cero SMOTE):**
   El dataset presenta un desbalance extremo (1:774). En lugar de inyectar ruido estadístico con datos sintéticos (SMOTE), se calculó la penalización matemática exacta de la clase minoritaria y se inyectó nativamente en el motor XGBoost (`scale_pos_weight = 336.34`).

4. **Explicabilidad Regulatoria (SHAP TreeExplainer):**
   Auditoría de caja blanca para generación de Reportes de Operación Sospechosa (ROS/SAR). La atribución mediante valores de Shapley confirma cuantitativamente que la activación de `flag_vaciado` es el principal vector de empuje (+8 a +14 SHAP value) hacia la clasificación de cuenta mula.

## Resultados del Prototipo
Validación sobre un caudal de prueba de +550,000 transacciones (20% de la data unseen):
* **Recall (Captura):** 99.6% (1,637 de 1,643 fraudes detectados)
* **Precision (Pureza):** 98.9%
* **Falsos Positivos:** Apenas 17 alertas sobre 552,439 transacciones legítimas.

Cada ejecución de `src/models/train.py` guarda estas métricas (matriz de confusión + classification report completo) en `models/metrics.json`, para trazabilidad y auditoría del modelo entrenado.

## Limitaciones Conocidas
* **Dataset sintético (PaySim):** los resultados anteriores provienen de datos simulados. El altísimo recall/precision puede reflejar parcialmente patrones propios del simulador (p. ej. balances en cero asociados a cuentas `M` de comercios) y no generalizar igual sobre tráfico real; se recomienda re-validar sobre datos productivos antes de usar el modelo para decisiones operativas.
* **Sin componente de grafos:** no se modelan relaciones entre cuentas (redes de mulas, reincidencia), solo la señal transaccional individual.

## Instalación

Requiere Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Uso

### 1. Preparar los datos
Coloca el dataset crudo de PaySim en `data/raw/paysim.csv` y ejecuta las celdas del notebook de EDA (`notebooks/01_eda_and_prototyping.ipynb`) para generar el Parquet filtrado en `data/processed/paysim_filtered.parquet`.

### 2. Entrenar el modelo
```bash
python -m src.models.train
```
Esto calcula los sensores, entrena el motor XGBoost, y exporta `models/xgb_aml_engine.json` junto con `models/metrics.json`.

### 3. Levantar la API de scoring
```bash
uvicorn src.api.main:app --reload
```
Endpoints disponibles:
* `GET /health` — estado del servicio y si el modelo está cargado en memoria.
* `POST /predict` — recibe una transacción (`type`, `amount`, `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`) y devuelve la probabilidad de fraude, si constituye una alerta SAR, y el detalle de los sensores calculados. Solo acepta transacciones de tipo `TRANSFER` o `CASH_OUT`.

Documentación interactiva disponible en `http://localhost:8000/docs` una vez levantado el servicio.

### 4. Correr los tests
```bash
pytest
```
Cubre los sensores de ingeniería de features (`tests/test_sensors.py`).

## Estructura del Proyecto
```text
├── data/
│   ├── raw/                  # Dataset crudo (ignorado en git)
│   └── processed/            # Parquet filtrado (ignorado en git)
├── models/                   # Artefactos del modelo y métricas (ignorado en git)
├── notebooks/
│   └── 01_eda_and_prototyping.ipynb
├── src/
│   ├── features/              # Cálculo de sensores y balance (compartido train/serve)
│   │   └── engineering.py
│   ├── models/                # Pipeline de entrenamiento y exportación
│   │   └── train.py
│   └── api/                   # Endpoints FastAPI para scoring en tiempo real
│       └── main.py
├── tests/
│   └── test_sensors.py
├── requirements.txt
└── README.md
```
