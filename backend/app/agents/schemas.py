"""Pydantic schemas for AI agent structured outputs"""
from pydantic import BaseModel, Field
from typing import Literal, Optional, List


class ParsedIntent(BaseModel):
    """Structured representation of user's ML task intent"""
    
    task_type: Literal["classification", "regression", "time_series", "nlp", "vision"] = Field(
        description="The type of machine learning task"
    )
    dataset_hint: Optional[str] = Field(
        default=None,
        description="Hint about what dataset to use (e.g., 'cat dog images', 'stock prices', 'sentiment text')"
    )
    target_column: Optional[str] = Field(
        default=None,
        description="Target column name for tabular data (e.g., 'price', 'label', 'class')"
    )
    model_preferences: Optional[List[str]] = Field(
        default=None,
        description="Preferred models if specified (e.g., 'CNN', 'RandomForest', 'LSTM')"
    )
    constraints: Optional[dict] = Field(
        default_factory=dict,
        description="Additional constraints like max_time, max_samples, min_accuracy"
    )
    
    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "task_type": "vision",
                    "dataset_hint": "cat dog images",
                    "target_column": None,
                    "model_preferences": ["CNN", "ResNet"],
                    "constraints": {"max_samples": 1000}
                },
                {
                    "task_type": "classification",
                    "dataset_hint": "iris flowers",
                    "target_column": "species",
                    "model_preferences": None,
                    "constraints": {}
                }
            ]
        }


class DatasetCandidate(BaseModel):
    """Hugging Face dataset candidate with license info"""
    
    id: str = Field(description="HF dataset ID (e.g., 'microsoft/cats_vs_dogs')")
    name: str = Field(description="Dataset name")
    description: str = Field(description="Brief description")
    license: Optional[str] = Field(description="License tag")
    license_valid: bool = Field(description="Whether license allows use")
    license_reason: str = Field(description="Reason for license decision")
    downloads: int = Field(description="Number of downloads")
    size: Optional[int] = Field(default=None, description="Dataset size in bytes")


class ModelCandidate(BaseModel):
    """ML model candidate for the task"""
    
    name: str = Field(description="Model name")
    type: str = Field(description="Model type (e.g., 'CNN', 'RandomForest')")
    description: str = Field(description="Model description")
    pros: List[str] = Field(description="Advantages")
    cons: List[str] = Field(description="Disadvantages")
    estimated_time: str = Field(description="Estimated training time")
    recommended: bool = Field(default=False, description="Whether this is the recommended choice")
