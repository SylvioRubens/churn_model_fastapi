import joblib
import os
import pandas as pd
from app.core.logger import setup_logger

logger = setup_logger()

class Predictor:
    def __init__(self, model_path: str):
        """Initialize the Predictor class with the path to the model, verifying if the model exists."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        self.model_path = model_path
        
        self.model, self.columns_expected = self.load_model()

    def load_model(self):
        """Load the trained model from the specified path."""
        try:
            model = joblib.load(self.model_path)
            
            model_expected = joblib.load("models/columns.pkl")
            
            return model, model_expected
        
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {self.model_path}: {str(e)}")

    def predict(self, data: dict):
        """Make a prediction using the loaded model."""
        try:
            df = pd.DataFrame([data])
            
            df_encoded = pd.get_dummies(df)
            
            df_encoded = df_encoded.reindex(columns=self.columns_expected, fill_value=0)
            
            return self.model.predict_proba(df_encoded)[0][1]  # Return the probability of churn
        
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}")