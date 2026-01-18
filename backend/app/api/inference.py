"""
Inference API - Test trained models with new data
"""
import joblib
import pandas as pd
from pathlib import Path
from fastapi import APIRouter, HTTPException, Body
from typing import Dict, List, Any

from app.orchestrator.pipeline import orchestrator as pipeline_orchestrator
from app.api.assets import ASSET_ROOT

router = APIRouter(prefix="/api/projects", tags=["inference"])


def _project_dir(project_id: str) -> Path:
    path = ASSET_ROOT / "projects" / project_id
    return path


@router.post("/{project_id}/predict")
async def predict(
    project_id: str,
    input_data: Dict[str, Any] = Body(..., description="Input features as key-value pairs")
):
    """
    Make predictions using the trained model
    
    Args:
        project_id: Project identifier
        input_data: Dictionary of feature names to values
        
    Returns:
        Prediction result with probability (if classifier)
        
    Example:
        POST /api/projects/demo-project/predict
        {
            "MedInc": 8.3252,
            "HouseAge": 41.0,
            "AveRooms": 6.984,
            "AveBedrms": 1.024,
            "Population": 322.0,
            "AveOccup": 2.556,
            "Latitude": 37.88,
            "Longitude": -122.23
        }
    """
    print(f"[Inference] Prediction request for project: {project_id}")
    print(f"[Inference] Input data: {input_data}")
    
    # Get project context
    async with pipeline_orchestrator._lock:
        context = pipeline_orchestrator._get_context(project_id)
    
    # Check if model is trained
    trained_model_path = context.get("trained_model_path")
    if not trained_model_path:
        raise HTTPException(
            status_code=400,
            detail="No trained model found. Complete the training stage first."
        )
    
    # Load the trained model
    try:
        model = joblib.load(trained_model_path)
        print(f"[Inference] Loaded model from: {trained_model_path}")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load model: {str(e)}"
        )
    
    # Get training metadata
    training_metrics = context.get("training_metrics", {})
    feature_names = training_metrics.get("features", [])
    
    if not feature_names:
        raise HTTPException(
            status_code=400,
            detail="No feature information found in training metadata"
        )
    
    # Prepare input data
    try:
        # Ensure input has all required features
        missing_features = set(feature_names) - set(input_data.keys())
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        
        # Create DataFrame with correct column order
        input_df = pd.DataFrame([input_data])[feature_names]
        print(f"[Inference] Input shape: {input_df.shape}")
        
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input data: {str(e)}. Expected features: {feature_names}"
        )
    
    # Make prediction
    try:
        prediction = model.predict(input_df)[0]
        
        # Get probability for classifiers
        result = {
            "prediction": float(prediction) if isinstance(prediction, (int, float)) else str(prediction),
            "model": training_metrics.get("model_name", "Unknown"),
            "features_used": feature_names,
        }
        
        # Add probabilities for classifiers
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_df)[0]
            classes = model.classes_ if hasattr(model, "classes_") else list(range(len(proba)))
            result["probabilities"] = {
                str(cls): float(prob) for cls, prob in zip(classes, proba)
            }
            result["confidence"] = float(max(proba))
        
        print(f"[Inference] Prediction: {result['prediction']}")
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@router.get("/{project_id}/model/info")
async def get_model_info(project_id: str):
    """
    Get information about the trained model
    
    Returns model metadata, training metrics, and example input format
    """
    print(f"[Inference] Model info request for project: {project_id}")
    
    # Get project context
    async with pipeline_orchestrator._lock:
        context = pipeline_orchestrator._get_context(project_id)
    
    trained_model_path = context.get("trained_model_path")
    if not trained_model_path:
        raise HTTPException(
            status_code=400,
            detail="No trained model found"
        )
    
    training_metrics = context.get("training_metrics", {})
    selected_model = context.get("selected_model", {})
    selected_dataset = context.get("selected_dataset", {})
    
    # Create example input
    feature_names = training_metrics.get("features", [])
    example_input = {feature: 0.0 for feature in feature_names}
    
    return {
        "model_path": trained_model_path,
        "model_name": training_metrics.get("model_name", "Unknown"),
        "model_type": selected_model.get("name", "Unknown"),
        "dataset": selected_dataset.get("name", "Unknown"),
        "training_metrics": {
            "train_score": training_metrics.get("train_score"),
            "test_score": training_metrics.get("test_score"),
            "train_samples": training_metrics.get("train_samples"),
            "test_samples": training_metrics.get("test_samples"),
        },
        "input_format": {
            "features": feature_names,
            "example": example_input,
        },
        "usage": {
            "endpoint": f"/api/projects/{project_id}/predict",
            "method": "POST",
            "body_example": example_input,
        }
    }
