from functools import lru_cache
from pydantic import Field, AnyUrl, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class TrainSettings(BaseSettings):
    """Class to hold the training settings."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    mlflow_tracking_uri: AnyUrl = Field(default="http://localhost:5000", alias="MLFLOW_TRACKING_URI")
    mlflow_experiment_name: str = Field(default="Telco Customer Churn Prediction", alias="MLFLOW_EXPERIMENT_NAME")
    mlflow_registered_model_name: str = Field(default="TelcoCustomerChurnModel", alias="MLFLOW_REGISTERED_MODEL_NAME")
    
    kaggle_username: SecretStr = Field(default='usernmame', alias="KAGGLE_USERNAME")
    kaggle_key: SecretStr = Field(default='KagleKey', alias="KAGGLE_KEY") 
    kaggle_dataset_name: str = Field(default="blastchar/telco-customer-churn", alias="KAGGLE_DATASET_NAME")
    
    threshold: float = Field(default=0.3, alias="THRESHOLD")
    random_state: int = Field(default=42, alias="RANDOM_STATE")
    test_size: float = Field(default=0.2, alias="TEST_SIZE")
    val_size: float = Field(default=0.1, alias="VAL_SIZE")
    optuna_n_trials: int = Field(default=50, alias="OPTUNA_N_TRIALS")
    
    train_log_path: str = Field(default="logs/train.log", alias="TRAIN_LOG_PATH")
    
    promote_on_train: bool = Field(default=False, alias="PROMOTE_ON_TRAIN")
    
        
@lru_cache
def get_training_settings() -> TrainSettings:
    """Function to get the settings, with caching."""
    return TrainSettings()