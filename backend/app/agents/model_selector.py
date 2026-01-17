from typing import Dict, List

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.llm.langchain_client import get_openrouter_llm


class ModelSelectorAgent:
    def __init__(self, temperature: float = 0.0, timeout_s: int = 20):
        self.models = self.load_models()
        self.models_by_id = {m["id"]: m for m in self.models.values()}
        self.llm = get_openrouter_llm(temperature=temperature, timeout=timeout_s)
        self.parser = JsonOutputParser()
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a model selection assistant for AutoML on tabular data. "
                    "Return JSON {\"candidates\": [model_id,...]} choosing up to 3 model_ids from the allowed list. "
                    "Allowed ids: {allowed_ids}. Only pick ids from that list.",
                ),
                (
                    "user",
                    "Task type: {task_type}\nTarget: {target}\nConstraints: {constraints}\n"
                    "Return JSON only. Use format: {format_instructions}",
                ),
            ]
        )

    def load_models(self):
        return {
            "logistic_regression": {
                "id": "logreg",
                "name": "Logistic Regression",
                "family": "classification",
                "why": "Fast baseline for binary/multiclass classification",
                "requirements": {"max_iter": 100, "solver": "lbfgs"},
            },
            "xgboost_classifier": {
                "id": "xgb_clf",
                "name": "XGBoost Classifier",
                "family": "classification",
                "why": "Strong baseline for tabular classification",
                "requirements": {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1},
            },
            "random_forest": {
                "id": "rf_clf",
                "name": "Random Forest",
                "family": "classification",
                "why": "Nonlinear, handles mixed features, robust",
                "requirements": {"n_estimators": 100, "max_depth": None},
            },
            "gradient_boosting": {
                "id": "gb_clf",
                "name": "Gradient Boosting",
                "family": "classification",
                "why": "Strong tabular performer with modest cost",
                "requirements": {"n_estimators": 100, "learning_rate": 0.1},
            },
            "linear_regression": {
                "id": "linreg",
                "name": "Linear Regression",
                "family": "regression",
                "why": "Baseline for regression",
                "requirements": {},
            },
            "random_forest_regressor": {
                "id": "rf_reg",
                "name": "Random Forest Regressor",
                "family": "regression",
                "why": "Handles nonlinearities and interactions",
                "requirements": {"n_estimators": 100, "max_depth": None},
            },
            "gradient_boosting_regressor": {
                "id": "gb_reg",
                "name": "Gradient Boosting Regressor",
                "family": "regression",
                "why": "Strong tabular baseline",
                "requirements": {"n_estimators": 100, "learning_rate": 0.1},
            },
            "xgboost_regressor": {
                "id": "xgb_reg",
                "name": "XGBoost Regressor",
                "family": "regression",
                "why": "Strong baseline for tabular regression",
                "requirements": {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1},
            },
        }

    def _deterministic_select(self, task_type: str) -> List[Dict]:
        if task_type not in ["classification", "regression"]:
            raise ValueError("Invalid task type. Must be 'classification' or 'regression'.")

        return [model for model in self.models.values() if model["family"] == task_type]

    def select_model(self, task_type: str, target: str | None = None, constraints: Dict | None = None) -> List[Dict]:
        task_type = (task_type or "").strip().lower()
        constraints = constraints or {}
        if task_type not in ["classification", "regression"]:
            raise ValueError("Invalid task type. Must be 'classification' or 'regression'.")

        # LangChain path if available
        if self.llm:
            try:
                allowed_ids = [m["id"] for m in self.models.values() if m["family"] == task_type]
                chain = self.prompt | self.llm | self.parser
                resp = chain.invoke(
                    {
                        "task_type": task_type,
                        "target": target or "",
                        "constraints": constraints,
                        "allowed_ids": ", ".join(allowed_ids),
                        "format_instructions": self.parser.get_format_instructions(),
                    }
                )
                candidate_ids = [cid for cid in (resp.get("candidates") or []) if cid in allowed_ids]
                if candidate_ids:
                    return [self.models_by_id[cid] for cid in candidate_ids]
            except Exception:
                pass

        # Fallback deterministic selection
        return self._deterministic_select(task_type)
