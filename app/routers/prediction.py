from fastapi import APIRouter, HTTPException
from app.services.predictor import Predictor
from app.models.input import InputData
from app.core.logger import setup_logger

logger = setup_logger()

router = APIRouter()

@router.post("/predict")
def predict(payload: InputData):
    try:
        predictor = Predictor(model_path="models/churn_model.pkl")
        
        churn = predictor.predict(payload.model_dump())
        
        return {"result": churn}, 200
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))