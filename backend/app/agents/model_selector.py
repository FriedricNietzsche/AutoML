"""
Model Selection Agent (Task 3.3)

Recommends candidate models based on task type from PROMPT_PARSED.
Emits:
  - MODEL_CANDIDATES: {models:[{id,name,family,why,requirements}]}
  - MODEL_SELECTED: {model_id} (when user/system confirms a choice)

Contract spec (FRONTEND_BACKEND_CONTRACT.md):
  MODEL_CANDIDATES: {models:[{id,name,family,why,requirements}]}
  MODEL_SELECTED: {model_id}

Handoff: Regression -> Linear/RF/GB; Classification -> Logistic/RF/GB.
"""

import copy
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ModelCandidate(BaseModel):
    """A single model candidate."""

    id: str = Field(..., description="Unique identifier for this model (e.g., 'random_forest_classifier').")
    name: str = Field(..., description="Human-readable name (e.g., 'Random Forest').")
    family: str = Field(..., description="Model family (e.g., 'ensemble', 'linear', 'tree').")
    why: str = Field(
        ...,
        description="Short rationale for recommending this model for the task.",
    )
    requirements: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional requirements or notes (e.g., min_samples, hyperparameters, etc.).",
    )


class ModelCandidatesPayload(BaseModel):
    """Stage 1 contract payload for event `MODEL_CANDIDATES`."""

    models: List[ModelCandidate] = Field(
        default_factory=list,
        description="List of recommended model candidates.",
    )


class ModelSelectedPayload(BaseModel):
    """Stage 1 contract payload for event `MODEL_SELECTED`."""

    model_id: str = Field(..., description="ID of the selected model.")


class ModelSelectorAgent:
    """
    Selects candidate models based on task type.

    This is a rule-based agent (no LLM needed for MVP).
    - Classification -> Logistic Regression, Random Forest, Gradient Boosting
    - Regression -> Linear Regression, Random Forest, Gradient Boosting
    - Other tasks -> fallback recommendations

    Usage:
        agent = ModelSelectorAgent()
        candidates = agent.get_candidates(task_type="classification")
        # Later, confirm selection:
        selected = agent.select_model(model_id="random_forest_classifier")
    """

    # Model registry: maps task types to candidate models
    MODEL_REGISTRY = {
        "classification": [
            {
                "id": "logistic_regression",
                "name": "Logistic Regression",
                "family": "linear",
                "why": "Fast, interpretable baseline for binary/multiclass classification.",
                "requirements": {"scaling": "recommended", "max_iter": 100, "solver": "lbfgs"},
            },
            {
                "id": "random_forest_classifier",
                "name": "Random Forest",
                "family": "ensemble",
                "why": "Robust, handles non-linearity and feature interactions well.",
                "requirements": {"n_estimators": 100, "max_depth": None},
            },
            {
                "id": "gradient_boosting_classifier",
                "name": "Gradient Boosting",
                "family": "ensemble",
                "why": "Often best accuracy; handles complex patterns and missing values.",
                "requirements": {"n_estimators": 100, "learning_rate": 0.1},
            },
        ],
        "regression": [
            {
                "id": "linear_regression",
                "name": "Linear Regression",
                "family": "linear",
                "why": "Simple, interpretable baseline for continuous targets.",
                "requirements": {"scaling": "recommended"},
            },
            {
                "id": "random_forest_regressor",
                "name": "Random Forest Regressor",
                "family": "ensemble",
                "why": "Robust to outliers, handles non-linearity well.",
                "requirements": {"n_estimators": 100, "max_depth": None},
            },
            {
                "id": "gradient_boosting_regressor",
                "name": "Gradient Boosting Regressor",
                "family": "ensemble",
                "why": "Best for complex non-linear relationships; often top performer.",
                "requirements": {"n_estimators": 100, "learning_rate": 0.1},
            },
        ],
        "clustering": [
            {
                "id": "kmeans",
                "name": "K-Means",
                "family": "centroid",
                "why": "Fast and simple for spherical clusters.",
                "requirements": {"n_clusters": "to be determined"},
            },
            {
                "id": "dbscan",
                "name": "DBSCAN",
                "family": "density",
                "why": "Finds arbitrary-shaped clusters; no need to specify k.",
                "requirements": {"eps": "auto", "min_samples": 5},
            },
        ],
        "timeseries": [
            {
                "id": "arima",
                "name": "ARIMA",
                "family": "statistical",
                "why": "Classic univariate time series forecasting.",
                "requirements": {"stationarity": "check ADF test"},
            },
            {
                "id": "prophet",
                "name": "Prophet",
                "family": "additive",
                "why": "Robust to missing data, seasonal patterns, and holidays.",
                "requirements": {"daily_seasonality": True},
            },
        ],
        "nlp": [
            {
                "id": "tfidf_logistic",
                "name": "TF-IDF + Logistic Regression",
                "family": "linear",
                "why": "Simple and effective baseline for text classification.",
                "requirements": {"max_features": 5000},
            },
            {
                "id": "bert_finetuning",
                "name": "BERT Fine-tuning",
                "family": "transformer",
                "why": "State-of-the-art for many NLP tasks; requires GPU.",
                "requirements": {"pretrained_model": "bert-base-uncased", "gpu": True},
            },
        ],
        "vision": [
            {
                "id": "cnn_simple",
                "name": "Simple CNN",
                "family": "neural_network",
                "why": "Baseline for image classification; fast training.",
                "requirements": {"input_size": "224x224", "gpu": "recommended"},
            },
            {
                "id": "resnet50",
                "name": "ResNet50 (transfer learning)",
                "family": "neural_network",
                "why": "Pre-trained on ImageNet; excellent for transfer learning.",
                "requirements": {"pretrained": True, "gpu": "recommended"},
            },
        ],
        "tabular": [
            {
                "id": "auto_sklearn",
                "name": "Auto-sklearn",
                "family": "automl",
                "why": "Automated model selection + hyperparameter tuning for tabular data.",
                "requirements": {"time_budget": "30 minutes"},
            },
        ],
        "other": [
            {
                "id": "generic_baseline",
                "name": "Generic Baseline Model",
                "family": "fallback",
                "why": "Task type unclear; recommend clarifying requirements first.",
                "requirements": {"needs_clarification": True},
            },
        ],
    }

    def __init__(self):
        """Initialize the ModelSelectorAgent."""
        pass

    def get_candidates(
        self,
        task_type: str,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Get model candidates for a given task type.

        Args:
            task_type: The ML task type (classification, regression, etc.)
            constraints: Optional constraints from PROMPT_PARSED (e.g., time limits, compute)

        Returns:
            MODEL_CANDIDATES payload: {models: [...]}
        """
        task_type = (task_type or "other").strip().lower()
        constraints = constraints or {}

        # Get base candidates from registry
        candidates = self.MODEL_REGISTRY.get(task_type, self.MODEL_REGISTRY["other"])

        # Deep copy to avoid mutating the registry
        candidates = copy.deepcopy(candidates)

        # Apply constraint filtering (optional enhancement)
        # For MVP, we just return all candidates; in production, you could filter by:
        # - training_time_limit
        # - gpu_available
        # - interpretability requirements
        # etc.

        # Convert to Pydantic models for validation
        model_list = [ModelCandidate(**c) for c in candidates]

        payload = ModelCandidatesPayload(models=model_list)
        return payload.model_dump()

    def select_model(self, model_id: str) -> Dict[str, Any]:
        """
        Confirm model selection.

        Args:
            model_id: The ID of the selected model

        Returns:
            MODEL_SELECTED payload: {model_id: ...}
        """
        if not model_id:
            raise ValueError("model_id cannot be empty")

        payload = ModelSelectedPayload(model_id=model_id)
        return payload.model_dump()

    def validate_model_exists(self, model_id: str, task_type: str) -> bool:
        """
        Check if a model_id is valid for the given task_type.

        Args:
            model_id: The model ID to validate
            task_type: The task type to check against

        Returns:
            True if model exists in registry for this task type
        """
        task_type = (task_type or "other").strip().lower()
        candidates = self.MODEL_REGISTRY.get(task_type, self.MODEL_REGISTRY["other"])
        valid_ids = {c["id"] for c in candidates}
        return model_id in valid_ids