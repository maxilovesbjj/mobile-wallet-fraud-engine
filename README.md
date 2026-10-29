# Mobile Wallet Fraud Detection Engine

Motor de detección de fraude de alto rendimiento para transacciones de billeteras móviles (basado en el dataset PaySim). Diseñado con un enfoque en ingeniería de datos para producción y *Cost-Sensitive Learning*.

## Arquitectura y Stack Tecnológico
* **Data Processing & Pipelines:** DuckDB, Parquet
* **Machine Learning:** XGBoost, Scikit-Learn
* **Análisis Exploratorio:** Pandas, Jupyter

## Enfoque de Ingeniería

1. **Out-of-Core Processing (DuckDB):** 
   El dataset crudo contiene +6.3 millones de transacciones. En lugar de saturar la memoria RAM cargando todo en Pandas, se implementó DuckDB como motor SQL en memoria para diagnosticar y aislar la fuga. Se demostró que el 100% del fraude ocurre en solo 2 canales (`TRANSFER` y `CASH_OUT`), permitiendo reducir el caudal de entrenamiento a 2.7M de registros en formato Parquet.

2. **Feature Engineering (Física de Balances):**
   En lugar de depender de los montos brutos, se construyeron sensores matemáticos que miden la discrepancia contable entre saldos iniciales, finales y el monto transferido, delatando los vaciados de cuentas típicos de redes de lavado.

3. **Cost-Sensitive Learning (Cero SMOTE):**
   El dataset presenta un desbalance extremo (1:774). En lugar de inyectar ruido estadístico con datos sintéticos (SMOTE), se calculó la penalización matemática exacta de la clase minoritaria y se inyectó nativamente en el motor XGBoost (`scale_pos_weight = 336.34`).

## Resultados del Prototipo
Validación sobre un caudal de prueba de +550,000 transacciones (20% de la data unseen):
* **Recall (Captura):** 99.6%
* **Precision (Pureza):** 98.9%
* **Falsos Positivos:** Apenas 17 alertas erróneas (sobre 552k transacciones legítimas).

## Estructura del Proyecto
* `data/raw/`: Caudal de datos crudos (ignorado en git).
* `data/processed/`: Tablas filtradas en formato columnar Parquet (ignorado en git).
* `notebooks/`: Prototipado y experimentación empírica.
* `src/`: (Próximamente) Módulos de extracción, entrenamiento serializado y API en FastAPI.
