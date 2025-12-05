"""
FastAPI application for Accident Risk Prediction
This API provides endpoints to predict accident risk based on road and environmental conditions.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

# Initialize FastAPI app
app = FastAPI(
    title="Accident Risk Prediction API",
    description="API for predicting accident risk based on road conditions and environmental factors",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model
try:
    # Get absolute path to the model
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent
    MODEL_PATH = project_root / "models" / "best_model.joblib"
    
    print(f"DEBUG: Current directory: {current_dir}")
    print(f"DEBUG: Project root: {project_root}")
    print(f"DEBUG: Model path: {MODEL_PATH}")
    print(f"DEBUG: Model file exists: {MODEL_PATH.exists()}")
    
    model = joblib.load(MODEL_PATH)
    print(f"✓ Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    # Try fallback to current directory just in case
    try:
        fallback_path = Path("best_model.joblib")
        print(f"DEBUG: Trying fallback path: {fallback_path.resolve()}")
        model = joblib.load(fallback_path)
        print(f"✓ Model loaded from fallback path")
    except Exception as e2:
        print(f"❌ Fallback failed: {e2}")
        model = None


# Define request schema
class RoadConditions(BaseModel):
    """Input schema for accident risk prediction"""
    road_type: Literal["urban", "rural", "highway"] = Field(
        ..., 
        description="Type of road"
    )
    num_lanes: int = Field(
        ..., 
        ge=1, 
        le=4, 
        description="Number of lanes (1-4)"
    )
    curvature: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Road curvature (0.0-1.0)"
    )
    speed_limit: Literal[25, 35, 45, 60, 70] = Field(
        ..., 
        description="Speed limit in mph"
    )
    lighting: Literal["daylight", "dim", "night"] = Field(
        ..., 
        description="Lighting conditions"
    )
    weather: Literal["clear", "rainy", "foggy"] = Field(
        ..., 
        description="Weather conditions"
    )
    road_signs_present: bool = Field(
        ..., 
        description="Whether road signs are present"
    )
    public_road: bool = Field(
        ..., 
        description="Whether it's a public road"
    )
    time_of_day: Literal["morning", "afternoon", "evening"] = Field(
        ..., 
        description="Time of day"
    )
    holiday: bool = Field(
        ..., 
        description="Whether it's a holiday"
    )
    school_season: bool = Field(
        ..., 
        description="Whether it's school season"
    )
    num_reported_accidents: int = Field(
        ..., 
        ge=0, 
        description="Number of previously reported accidents"
    )

    class Config:
        schema_extra = {
            "example": {
                "road_type": "urban",
                "num_lanes": 2,
                "curvature": 0.06,
                "speed_limit": 35,
                "lighting": "daylight",
                "weather": "rainy",
                "road_signs_present": False,
                "public_road": True,
                "time_of_day": "afternoon",
                "holiday": False,
                "school_season": True,
                "num_reported_accidents": 1
            }
        }


# Define response schema
class PredictionResponse(BaseModel):
    """Output schema for accident risk prediction"""
    accident_risk: float = Field(..., description="Predicted accident risk (0.0-1.0)")
    risk_level: str = Field(..., description="Risk level category")
    confidence: str = Field(..., description="Model confidence level")


def categorize_risk(risk_value: float) -> str:
    """Categorize risk value into human-readable levels"""
    if risk_value < 0.2:
        return "Very Low"
    elif risk_value < 0.35:
        return "Low"
    elif risk_value < 0.5:
        return "Moderate"
    elif risk_value < 0.7:
        return "High"
    else:
        return "Very High"


def get_confidence_level(risk_value: float) -> str:
    """Determine confidence level based on risk value"""
    # This is a simplified confidence metric
    # In production, you might want to use model-specific confidence scores
    if 0.3 <= risk_value <= 0.4:
        return "High"
    elif 0.2 <= risk_value <= 0.5:
        return "Medium"
    else:
        return "Medium"


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode categorical features to match model training
    """
    # Create a copy
    df_encoded = df.copy()
    
    # Define all categorical columns including booleans
    categorical_cols = [
        'road_type', 'lighting', 'weather', 'time_of_day',
        'road_signs_present', 'public_road', 'holiday', 'school_season'
    ]
    
    # One-hot encode categorical columns
    df_encoded = pd.get_dummies(df_encoded, columns=categorical_cols, drop_first=False)
    
    # Ensure all expected columns are present in the correct order
    # Based on the model expecting 24 features total
    expected_columns = [
        # Numerical features
        'num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents',
        
        # String categorical features
        'road_type_highway', 'road_type_rural', 'road_type_urban',
        'lighting_daylight', 'lighting_dim', 'lighting_night',
        'weather_clear', 'weather_foggy', 'weather_rainy',
        'time_of_day_afternoon', 'time_of_day_evening', 'time_of_day_morning',
        
        # Boolean categorical features (One-Hot Encoded)
        'road_signs_present_False', 'road_signs_present_True',
        'public_road_False', 'public_road_True',
        'holiday_False', 'holiday_True',
        'school_season_False', 'school_season_True'
    ]
    
    # Add missing columns with 0
    for col in expected_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    
    # Select only the expected columns in the correct order
    df_encoded = df_encoded[expected_columns]
    
    return df_encoded


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Accident Risk Prediction API",
        "version": "1.0.0",
        "status": "active" if model is not None else "model not loaded",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_accident_risk(conditions: RoadConditions):
    """
    Predict accident risk based on road and environmental conditions
    
    Args:
        conditions: Road conditions and environmental factors
        
    Returns:
        PredictionResponse with accident risk prediction
    """
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please check server logs."
        )
    
    try:
        # Convert input to DataFrame
        input_data = pd.DataFrame([conditions.dict()])
        
        # Encode categorical features
        input_data_encoded = encode_categorical_features(input_data)
        
        # Make prediction
        prediction = model.predict(input_data_encoded)[0]
        
        # Ensure prediction is within valid range
        prediction = np.clip(prediction, 0.0, 1.0)
        
        # Categorize risk
        risk_level = categorize_risk(prediction)
        confidence = get_confidence_level(prediction)
        
        return PredictionResponse(
            accident_risk=round(float(prediction), 4),
            risk_level=risk_level,
            confidence=confidence
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


@app.post("/predict/batch")
async def predict_batch(conditions_list: list[RoadConditions]):
    """
    Predict accident risk for multiple road conditions
    
    Args:
        conditions_list: List of road conditions
        
    Returns:
        List of predictions
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs."
        )
    
    try:
        # Convert all inputs to DataFrame
        input_data = pd.DataFrame([c.dict() for c in conditions_list])
        
        # Encode categorical features
        input_data_encoded = encode_categorical_features(input_data)
        
        # Make predictions
        predictions = model.predict(input_data_encoded)
        
        # Prepare responses
        responses = []
        for pred in predictions:
            pred = np.clip(pred, 0.0, 1.0)
            responses.append({
                "accident_risk": round(float(pred), 4),
                "risk_level": categorize_risk(pred),
                "confidence": get_confidence_level(pred)
            })
        
        return responses
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction error: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
