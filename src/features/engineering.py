import pandas as pd
import numpy as np

def compute_financial_sensors(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula sensores contables sobre transacciones individuales."""
    data = df.copy()
    
    # Sensor 1: Discrepancia en cuenta de origen
    data['error_balance_origen'] = np.abs((data['oldbalanceOrg'] - data['newbalanceOrig']) - data['amount'])
    
    # Sensor 2: Discrepancia en cuenta de destino
    data['error_balance_destino'] = np.abs((data['newbalanceDest'] - data['oldbalanceDest']) - data['amount'])
    
    # Sensor 3: Firma de cuenta mula (vaciado total de fondos)
    data['flag_vaciado'] = np.where((data['amount'] == data['oldbalanceOrg']) & (data['amount'] > 0), 1, 0)
    
    # Codificacion de canales vulnerables
    if 'type' in data.columns:
        data['type_encoded'] = data['type'].map({'TRANSFER': 0, 'CASH_OUT': 1})
        
    return data
