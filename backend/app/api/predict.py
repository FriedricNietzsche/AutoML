"""
Prediction API - Real-time model inference
"""
from fastapi import APIRouter, HTTPException, Body
from pathlib import Path
from typing import Dict, Any, List
import joblib
import pandas as pd
import numpy as np

router = APIRouter()


@router.post("/{project_id}/predict")
async def predict(
    project_id: str,
    input_data: Dict[str, Any] = Body(...)
):
    """
    Make predictions using the trained model
    
    Args:
        project_id: Project identifier
        input_data: Dictionary of feature values
        
    Returns:
        Dictionary with prediction, probabilities (if classification), and metadata
    """
    print(f"[Predict API] Prediction request for project: {project_id}")
    print(f"[Predict API] Input data: {input_data}")
    
    # Find the trained model
    model_path = Path(f"data/assets/projects/{project_id}/trained_model.joblib")
    
    if not model_path.exists():
        raise HTTPException(
            status_code=404, 
            detail="No trained model found. Train a model first."
        )
    
    try:
        # Load model
        model_data = joblib.load(model_path)
        
        # Extract model and metadata
        if isinstance(model_data, dict):
            model = model_data.get('model')
            feature_names = model_data.get('feature_names', [])
            task_type = model_data.get('task_type', 'classification')
        else:
            model = model_data
            feature_names = list(input_data.keys())
            task_type = 'classification'
        
        print(f"[Predict API] Model type: {type(model).__name__}")
        print(f"[Predict API] Task type: {task_type}")
        print(f"[Predict API] Features: {len(feature_names)}")
        
        # Convert input to DataFrame with correct feature order
        df_input = pd.DataFrame([input_data])
        
        # Ensure all required features are present
        missing_features = set(feature_names) - set(df_input.columns)
        if missing_features:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required features: {list(missing_features)}"
            )
        
        # Select only the features the model expects, in the correct order
        df_input = df_input[feature_names]
        
        # Make prediction
        prediction = model.predict(df_input)
        
        result = {
            "prediction": prediction[0].tolist() if isinstance(prediction[0], np.ndarray) else int(prediction[0]) if isinstance(prediction[0], (np.integer, np.int64)) else float(prediction[0]),
            "task_type": task_type,
            "model_type": type(model).__name__,
        }
        
        # Add probabilities for classification tasks
        if hasattr(model, 'predict_proba') and task_type == 'classification':
            probabilities = model.predict_proba(df_input)
            result["probabilities"] = probabilities[0].tolist()
            result["confidence"] = float(max(probabilities[0]))
        
        print(f"[Predict API] ✅ Prediction: {result['prediction']}")
        if 'confidence' in result:
            print(f"[Predict API] ✅ Confidence: {result['confidence']:.2%}")
        
        return result
        
    except Exception as e:
        print(f"[Predict API] ❌ Prediction error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@router.get("/{project_id}/sample-data")
async def get_sample_data(project_id: str, num_samples: int = 3):
    """
    Get sample data from the dataset for testing predictions
    
    Args:
        project_id: Project identifier
        num_samples: Number of sample rows to return
        
    Returns:
        List of sample data dictionaries
    """
    print(f"[Predict API] Fetching sample data for project: {project_id}")
    
    # Try to load original dataset
    project_dir = Path(f"data/assets/projects/{project_id}")
    
    # Look for CSV files in the project directory
    csv_files = list(project_dir.glob("*.csv"))
    
    # Prefer files that aren't processed
    dataset_file = None
    for csv_file in csv_files:
        if "processed" not in csv_file.name.lower():
            dataset_file = csv_file
            break
    
    if not dataset_file and csv_files:
        dataset_file = csv_files[0]
    
    if not dataset_file:
        raise HTTPException(
            status_code=404,
            detail="No dataset found for this project"
        )
    
    try:
        # Load dataset
        df = pd.read_csv(dataset_file)
        
        # Load model to get feature names and target column
        model_path = project_dir / "trained_model.joblib"
        target_column = None
        feature_names = None
        
        if model_path.exists():
            model_data = joblib.load(model_path)
            if isinstance(model_data, dict):
                feature_names = model_data.get('feature_names', [])
        
        # If we have feature names, only include those columns
        if feature_names:
            # Find target column (column not in features)
            all_cols = set(df.columns)
            feature_cols = set(feature_names)
            target_cols = all_cols - feature_cols
            if target_cols:
                target_column = list(target_cols)[0]
        
        # Sample random rows
        num_samples = min(num_samples, len(df))
        sample_df = df.sample(n=num_samples, random_state=42)
        
        # Convert to list of dictionaries
        samples = []
        for _, row in sample_df.iterrows():
            sample = {}
            actual_value = None
            
            for col, val in row.items():
                # Skip target column in features
                if target_column and col == target_column:
                    actual_value = val
                    continue
                
                # Only include features if we know them
                if feature_names and col not in feature_names:
                    continue
                
                # Handle NaN values
                if pd.isna(val):
                    sample[col] = None
                # Convert numpy types to Python types
                elif isinstance(val, (np.integer, np.int64)):
                    sample[col] = int(val)
                elif isinstance(val, (np.floating, np.float64)):
                    sample[col] = float(val)
                else:
                    sample[col] = val
            
            samples.append({
                "features": sample,
                "actual": actual_value
            })
        
        print(f"[Predict API] ✅ Returning {len(samples)} sample rows")
        
        return {
            "samples": samples,
            "feature_names": feature_names or list(df.columns),
            "target_column": target_column
        }
        
    except Exception as e:
        print(f"[Predict API] ❌ Error loading sample data: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load sample data: {str(e)}"
        )
