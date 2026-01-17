from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import joblib

class TrainerAgent:
    def __init__(self, model, X, y):
        self.model = model
        self.X = X
        self.y = y
        self.trained_model = None

    def train(self):
        # Split the data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        
        # Train the model
        self.trained_model = self.model.fit(X_train, y_train)
        
        # Validate the model
        predictions = self.trained_model.predict(X_val)
        self.evaluate(predictions, y_val)

    def evaluate(self, predictions, y_val):
        # Evaluate the model based on the type of task
        if hasattr(self.model, "score"):
            accuracy = accuracy_score(y_val, predictions)
            print(f"Model Accuracy: {accuracy:.2f}")
        else:
            mse = mean_squared_error(y_val, predictions)
            print(f"Model Mean Squared Error: {mse:.2f}")

    def save_model(self, filepath):
        # Save the trained model to a file
        joblib.dump(self.trained_model, filepath)
        print(f"Model saved to {filepath}")