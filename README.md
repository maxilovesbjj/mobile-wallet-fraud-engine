# Mobile Wallet AML Engine: Detección de Layering y Cuentas Mula

Motor de detección en tiempo real para identificar transacciones de billeteras móviles que exhiben el patrón de *layering* típico de lavado de activos — específicamente el ciclo `TRANSFER` → `CASH_OUT` usado para mover e integrar fondos ilícitos fuera del sistema financiero —, priorizando la captura de cuentas mula (usadas para vaciar saldos) sobre el fraude transaccional genérico.

No modela relaciones topológicas entre cuentas (la componente de red se aborda en análisis de grafos posteriores); se enfoca estrictamente en la señal a nivel de transacción individual y dinámica de balance. El uso de validadores lógicos y árboles de decisión (XGBoost) garantiza un bajo costo computacional, optimizado para *scoring* en tiempo real con baja latencia.

## Arquitectura y Stack Tecnológico
* **Data Processing & Pipelines:** DuckDB, Parquet
* **Machine Learning & Explainability:** XGBoost, Scikit-Learn, SHAP
* **Serving & API:** FastAPI, Pydantic v2
* **Análisis Exploratorio:** Pandas, Jupyter

## Enfoque de Ingeniería

1. **Out-of-Core Processing (DuckDB):** 
   El dataset crudo contiene +6.3 millones de transacciones. En lugar de saturar la memoria RAM cargando todo en Pandas, se implementó DuckDB como motor SQL en memoria para diagnosticar la fuga. Se demostró que el 100% del fraude ocurre en solo 2 canales (`TRANSFER` y `CASH_OUT`), permitiendo reducir el caudal de entrenamiento a 2.7M de registros en formato columnar Parquet.

2. **Feature Engineering (Física de Balances):**
   En lugar de depender de montos brutos, se construyeron sensores matemáticos que miden la discrepancia contable entre saldos iniciales, finales y el monto transferido, delatando los vaciados de saldos típicos de cuentas mula.

3. **Cost-Sensitive Learning (Cero SMOTE):**
   El dataset presenta un desbalance extremo (1:774). En lugar de inyectar ruido estadístico con datos sintéticos (SMOTE), se calculó la penalización matemática exacta de la clase minoritaria y se inyectó nativamente en el motor XGBoost (`scale_pos_weight = 336.34`).

4. **Explicabilidad Regulatoria (SHAP TreeExplainer):**
   Auditoría de caja blanca para generación de Reportes de Operación Sospechosa (ROS/SAR). La atribución mediante valores de Shapley confirma cuantitativamente que la activación de `flag_vaciado` es el principal vector de empuje (+8 a +14 SHAP value) hacia la clasificación de cuenta mula.

## Resultados del Prototipo
Validación sobre un caudal de prueba de +550,000 transacciones (20% de la data unseen):
* **Recall (Captura):** 99.6% (1,637 de 1,643 fraudes detectados)
* **Precision (Pureza):** 98.9%
* **Falsos Positivos:** Apenas 17 alertas sobre 552,439 transacciones legítimas.

## Estructura del Proyecto
```text
├── data/
│   ├── raw/                  # Dataset crudo (ignorado en git)
│   └── processed/            # Parquet filtrado (ignorado en git)
├── models/                   # Artefactos del modelo serializado (.json / .pkl)
├── notebooks/
│   └── 01_eda_and_prototyping.ipynb
├── src/
│   ├── features/             # Cálculo de sensores y balance
│   ├── models/               # Pipeline de entrenamiento y exportación
│   └── api/                  # Endpoints FastAPI para scoring en tiempo real
├── requirements.txt
└── README.md
