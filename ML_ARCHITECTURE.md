# ML Architecture - Expert-Grade Model Selection

## Problem Solved

**Before**: Text data was being preprocessed incorrectly. The IMDB dataset (movie reviews) was being converted to a single float value, which is completely wrong for text classification.

**After**: Each ML task now uses industry-standard, best-practice models:

## Architecture Philosophy

**One trainer per model type** - Each file is a self-contained trainer for a specific algorithm, using the library that experts recommend for that task.

## Model Selection Strategy

### 1. **Text Classification** → `sentiment_classifier.py`
- **Primary**: HuggingFace Transformers (DistilBERT, BERT, RoBERTa)
- **Fallback**: TF-IDF + Logistic Regression (sklearn)
- **Why**: Transformers are state-of-the-art for NLP, achieving 95%+ accuracy on sentiment tasks
- **Library**: `transformers` + `torch` (PyTorch)
- **Example Use Cases**:
  - Movie review sentiment (IMDB)
  - Product reviews
  - Social media sentiment
  - News classification

### 2. **Tabular Classification/Regression** → `xgboost_trainer.py`
- **Primary**: XGBoost
- **Why**: Dominates Kaggle competitions, best for structured/tabular data
- **Library**: `xgboost`
- **Features**:
  - Handles missing values automatically
  - Built-in regularization (prevents overfitting)
  - GPU acceleration available
  - Feature importance analysis
- **Example Use Cases**:
  - Titanic survival prediction
  - Customer churn prediction
  - Credit risk scoring
  - House price prediction

### 3. **Tabular Classification/Regression (Alternative)** → `random_forest_trainer.py`
- **Fallback**: Random Forest (when XGBoost not available)
- **Why**: Reliable, interpretable, no hyperparameter tuning needed
- **Library**: `sklearn`
- **Features**:
  - Robust to outliers
  - Built-in feature importance
  - No scaling required
  - Good for smaller datasets

## File Structure

```
backend/app/ml/
├── trainer_factory.py          # Smart model selection
├── text/
│   ├── sentiment_classifier.py # Transformers for text
│   └── preprocessing.py         # TF-IDF utilities (fallback)
└── tabular/
    ├── xgboost_trainer.py      # XGBoost (best for tabular)
    └── random_forest_trainer.py # RandomForest (solid alternative)
```

## How It Works

### TrainerFactory (Smart Dispatcher)

```python
from app.ml.trainer_factory import TrainerFactory

# Automatically picks the best model for your data
task_type = TrainerFactory.detect_task_type(df, target_column="label")
# Returns: "text_classification", "tabular_classification", or "tabular_regression"

# Train with best-in-class model
result = TrainerFactory.train_model(
    task_type=task_type,
    df=preprocessed_df,
    target_column="label",
    model_name="auto",  # Picks best model automatically
)

# Result contains:
# - trainer: The trained model object
# - metrics: accuracy, F1, R², etc.
# - task_type: What task was detected
# - model_name: Which model was used
```

### Example: Text Classification (IMDB)

```python
# IMDB dataset: {"text": "movie review...", "label": 0/1}

# Factory detects it's text classification
trainer = TrainerFactory.get_trainer("text_classification")

# Trains DistilBERT transformer model
metrics = trainer.train(
    train_texts=["Great movie!", "Terrible film"],
    train_labels=[1, 0],
    num_epochs=3,
)

# Predict
predictions = trainer.predict(["This was amazing!"])
# Returns: [1] (positive sentiment)
```

### Example: Tabular Classification (Titanic)

```python
# Titanic dataset: {Age, Sex, Pclass, ... → Survived}

# Factory detects it's tabular classification
trainer = TrainerFactory.get_trainer("tabular_classification")

# Trains XGBoost classifier
metrics = trainer.train(
    X_train=features_df,
    y_train=survived_labels,
    n_estimators=100,
)

# Predict
predictions = trainer.predict(new_passengers_df)
# Returns: [0, 1, 1, 0] (survived/died)
```

## Why This Approach?

### ✅ Best Practices
- **Text** → Transformers (what GPT-3, ChatGPT use internally)
- **Tabular** → XGBoost (what Kaggle winners use)
- Each model uses its optimal library

### ✅ Separation of Concerns
- One file = One algorithm
- Easy to add new models (just create new trainer)
- Easy to test/debug individual models

### ✅ Graceful Fallbacks
- Text: Transformers → TF-IDF + sklearn
- Tabular: XGBoost → Random Forest
- Always returns a working model

### ✅ Production-Ready
- Models save/load properly
- Feature names tracked
- Metrics tracked
- GPU support (transformers, XGBoost)

## Current Status

### ✅ Implemented
- [x] SentimentClassifier (transformers)
- [x] XGBoostTrainer (tabular)
- [x] RandomForestTrainer (tabular)
- [x] TrainerFactory (smart selection)

### 📋 TODO (Future)
- [ ] LightGBM trainer (another Kaggle favorite)
- [ ] Neural network trainer (for complex tabular data)
- [ ] Image classification (CNN with transfer learning)
- [ ] Time series forecasting (LSTM, Prophet)

## Dependencies

Required libraries (in `requirements.txt`):

```
# Core ML
scikit-learn>=1.3.0
xgboost>=2.0.0

# Text Classification (Transformers)
transformers>=4.30.0
torch>=2.0.0
sentencepiece>=0.1.99
accelerate>=0.20.0

# Data
pandas>=2.0.0
numpy>=1.24.0
datasets>=3.1.0
```

## Testing Your Fix

### Before (Broken)
```bash
curl http://localhost:8000/api/projects/demo-project/model/info
# Output: {"features": ["text"], "example": {"text": 0.0}}  ❌ Wrong!
```

### After (Fixed)
```bash
# 1. Install new dependencies
pip install transformers torch accelerate

# 2. Re-train model (it will use new trainer)
# Frontend: "Build a sentiment classifier for movie reviews"
# Select IMDB dataset → Training starts

# 3. Check model info
curl http://localhost:8000/api/projects/demo-project/model/info
# Output: Proper transformer model with text input  ✅
```

## Key Insight

**Sklearn is great for tabular data, terrible for text.**

- ❌ **Don't**: Convert text to single float with label encoding
- ❌ **Don't**: Use RandomForest directly on text
- ✅ **Do**: Use transformers for text (DistilBERT, BERT)
- ✅ **Do**: Use XGBoost for tabular data
- ✅ **Do**: Let TrainerFactory pick the right tool for the job

---

**Summary**: You're 100% right that we should use the best library for each task. The new architecture does exactly that - transformers for text, XGBoost for tabular, with clean separation between model types.
