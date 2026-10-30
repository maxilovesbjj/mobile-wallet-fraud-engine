from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    MODEL_PATH: str = "models/xgb_aml_engine.json"
    DATA_PATH: str = "data/processed/paysim_filtered.parquet"
    METRICS_PATH: str = "models/metrics.json"
    SAR_ALERT_THRESHOLD: float = 0.50

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
