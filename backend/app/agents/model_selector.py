class ModelSelectorAgent:
    def __init__(self):
        self.models = self.load_models()

    def load_models(self):
        return {
            "logistic_regression": {
                "name": "Logistic Regression",
                "family": "classification",
                "requirements": {"max_iter": 100, "solver": "lbfgs"},
            },
            "random_forest": {
                "name": "Random Forest",
                "family": "classification",
                "requirements": {"n_estimators": 100, "max_depth": None},
            },
            "gradient_boosting": {
                "name": "Gradient Boosting",
                "family": "classification",
                "requirements": {"n_estimators": 100, "learning_rate": 0.1},
            },
            "linear_regression": {
                "name": "Linear Regression",
                "family": "regression",
                "requirements": {},
            },
            "random_forest_regressor": {
                "name": "Random Forest Regressor",
                "family": "regression",
                "requirements": {"n_estimators": 100, "max_depth": None},
            },
            "gradient_boosting_regressor": {
                "name": "Gradient Boosting Regressor",
                "family": "regression",
                "requirements": {"n_estimators": 100, "learning_rate": 0.1},
            },
        }

    def select_model(self, task_type):
        if task_type not in ["classification", "regression"]:
            raise ValueError("Invalid task type. Must be 'classification' or 'regression'.")

        return [
            model for model in self.models.values() if model["family"] == task_type
        ]