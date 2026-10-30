import pytest
from unittest.mock import patch, MagicMock
import numpy as np
import os

@pytest.fixture(autouse=True)
def mock_aml_model():
    """
    Mock global para desacoplar los tests de la API del artefacto real de XGBoost.
    Esto permite que CI/CD corra sin necesitar el modelo pesado o el dataset.
    """
    original_exists = os.path.exists
    
    # 1. Engañamos a FastAPI para que crea que el modelo .json existe
    def side_effect_exists(path):
        if path == "models/xgb_aml_engine.json":
            return True
        return original_exists(path)
        
    with patch("os.path.exists", side_effect=side_effect_exists):
        
        # 2. Creamos un modelo falso en memoria
        mock_model = MagicMock()
        
        # 3. Simulamos la inferencia: si la ingeniería de features detecta vaciado, lanza alerta
        def mock_predict_proba(X):
            probs = []
            for _, row in X.iterrows():
                if row.get("flag_vaciado", 0) == 1:
                    probs.append([0.05, 0.95]) # 95% prob de fraude -> Alerta SAR
                else:
                    probs.append([0.95, 0.05]) # 5% prob de fraude -> Normal
            return np.array(probs)
            
        mock_model.predict_proba.side_effect = mock_predict_proba
        
        # 4. Inyectamos este modelo falso reemplazando a XGBClassifier
        with patch("xgboost.XGBClassifier", return_value=mock_model):
            yield
