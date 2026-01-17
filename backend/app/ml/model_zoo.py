class ModelZoo:
    def __init__(self):
        self.models = {
            "LogisticRegression": {
                "class": "sklearn.linear_model.LogisticRegression",
                "params": {
                    "solver": "lbfgs",
                    "max_iter": 100,
                },
                "description": "Logistic Regression model for binary classification."
            },
            "RandomForestClassifier": {
                "class": "sklearn.ensemble.RandomForestClassifier",
                "params": {
                    "n_estimators": 100,
                    "max_depth": None,
                },
                "description": "Random Forest model for classification tasks."
            },
            "GradientBoostingClassifier": {
                "class": "sklearn.ensemble.GradientBoostingClassifier",
                "params": {
                    "n_estimators": 100,
                    "learning_rate": 0.1,
                    "max_depth": 3,
                },
                "description": "Gradient Boosting model for classification tasks."
            },
            "LinearRegression": {
                "class": "sklearn.linear_model.LinearRegression",
                "params": {},
                "description": "Linear Regression model for regression tasks."
            },
            "RandomForestRegressor": {
                "class": "sklearn.ensemble.RandomForestRegressor",
                "params": {
                    "n_estimators": 100,
                    "max_depth": None,
                },
                "description": "Random Forest model for regression tasks."
            },
            "GradientBoostingRegressor": {
                "class": "sklearn.ensemble.GradientBoostingRegressor",
                "params": {
                    "n_estimators": 100,
                    "learning_rate": 0.1,
                    "max_depth": 3,
                },
                "description": "Gradient Boosting model for regression tasks."
            },
        }

    def get_model(self, model_name):
        return self.models.get(model_name)

    def list_models(self):
        return list(self.models.keys())