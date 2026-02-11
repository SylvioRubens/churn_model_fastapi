from app.core.logger import setup_logger
import os
import pandas as pd
import mlflow.pyfunc

logger = setup_logger()

class Predictor:
    def __init__(self, model_uri: str):
        """Initialize the Predictor class by instantiating the model from MlFlow."""
        
        MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
        MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME")
        
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        
        self.model_uri = model_uri
        
        self.model, self.columns_expected = self.load_model()

    def load_model(self):
        """Load the trained model from the specified path."""
        try:
            return mlflow.pyfunc.load_model(self.model_uri)
        
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {self.model_uri} on MLFlow: {str(e)}")

    def predict(self, data: dict):
        """Make a prediction using the loaded model."""
        try:
            df = pd.DataFrame([data])
            
            df_encoded = pd.get_dummies(df)
            
            df_encoded = df_encoded.reindex(columns=self.columns_expected, fill_value=0)
            
            return self.model.predict_proba(df_encoded)[0][1]  # Return the probability of churn
        
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}")