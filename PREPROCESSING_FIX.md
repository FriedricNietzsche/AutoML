# Preprocessing Fix - Text Classification

## Problem
The preprocessing pipeline was **destroying text data** by label-encoding it into a single number, making NLP impossible.

**Old behavior:**
```
IMDB Review: "This movie was amazing!" → 12345 (label encoded)
```

This caused:
- Text classification to fail (model expected float, not text)
- Training to complete in 5 seconds (sklearn on metadata)
- Inference to show `{"text": 0.0}` instead of accepting actual text

## Solution

### 1. Smart Preprocessing (agents/preprocess.py)
The preprocessing now **detects and preserves** text columns:

```python
def encode_categorical_variables(df, target_column="label"):
    for col in df.columns:
        unique_ratio = unique_count / len(df)
        avg_length = df[col].astype(str).str.len().mean()
        
        # DETECT TEXT COLUMNS
        is_text_column = (unique_ratio > 0.9 and avg_length > 50)
        
        if is_text_column and col != target_column:
            # PRESERVE TEXT - DO NOT ENCODE
            print(f"🔤 Preserved TEXT column '{col}' for NLP")
            # Leave as raw text strings
            
        elif col == target_column:
            # Label encode target only
            df[col] = pd.Categorical(df[col]).codes
            
        elif unique_count <= 10:
            # One-hot encode true categoricals
            df = pd.get_dummies(df, columns=[col])
            
        else:
            # Label encode medium cardinality
            df[col] = pd.Categorical(df[col]).codes
```

**Criteria for text detection:**
- `unique_ratio > 0.9` (most values are unique - not a categorical)
- `avg_length > 50` (long strings - actual text, not labels)

### 2. Task-Aware Training (orchestrator/pipeline.py)
The training pipeline now:

1. **Loads ORIGINAL data** to detect task type (before preprocessing)
2. **Detects text columns** using the same criteria
3. **Routes to appropriate trainer:**
   - Text → `SentimentClassifier` (DistilBERT transformers)
   - Tabular → `XGBoostTrainer` (gradient boosting)
   - Image → `ImageClassifier` (ResNet/EfficientNet)

```python
# Load ORIGINAL data (with text preserved)
original_df = pd.read_csv(original_data_path)

# Detect task type from ORIGINAL data
actual_task_type = TrainerFactory.detect_task_type(original_df, target_column)

# Create appropriate trainer
trainer = TrainerFactory.get_trainer(
    task_type=actual_task_type,  # "text_classification"
    model_name=model_name,
    num_classes=num_classes
)

# For text: Pass original text strings
if actual_task_type == "text_classification":
    train_texts = original_df[text_col].tolist()  # Actual text!
    train_labels = original_df[target_column].tolist()
    
    metrics = trainer.train(
        train_texts=train_texts,
        train_labels=train_labels,
        num_epochs=3
    )
```

### 3. Model ID Mapping
Maps frontend model IDs to trainer names:

```python
model_name_map = {
    "rf_clf": "random_forest",   # Random Forest → Use RandomForest or XGBoost
    "xgb_clf": "xgboost",         # XGBoost → Use XGBoost
    "lr_clf": "auto",             # Logistic Regression → Auto-select best
}
```

## Data Flow

### IMDB Sentiment Classification (Before Fix)
```
1. Download: "This movie was great!" → imdb.csv
2. Preprocess: DESTROY TEXT → {"text": 0} (label encoded)
3. Train: sklearn on single number → 5 seconds
4. Inference: Expects float input ❌
```

### IMDB Sentiment Classification (After Fix)
```
1. Download: "This movie was great!" → imdb.csv
2. Preprocess: PRESERVE TEXT → {"text": "This movie was great!", "label": 1}
3. Detect: Text column found → text_classification
4. Train: DistilBERT transformer on actual text → 2-3 minutes
5. Inference: Accepts text strings ✅
```

## Files Modified

1. **`agents/preprocess.py`** - Smart text detection and preservation
   - `encode_categorical_variables()` - Added text detection logic
   - `preprocess()` - Added target_column parameter

2. **`orchestrator/pipeline.py`** - Task-aware training
   - `_execute_preprocess()` - Pass target_column to preprocessing
   - `_execute_train()` - Load original data, detect task type, map model IDs

3. **`ml/trainer_factory.py`** - Already has text detection (unchanged)

## Testing

### Test 1: IMDB Text Classification
```bash
# Frontend: "Build a sentiment classifier for movie reviews"
# Select: IMDB Movie Reviews
# Model: Random Forest (will use DistilBERT!)

# Expected:
# - Training: 2-3 minutes (transformers)
# - Model accepts: {"text": "This is a review"}
# - Returns: {"prediction": 1, "confidence": 0.95}
```

### Test 2: Tabular Data (Titanic)
```bash
# Frontend: "Predict Titanic survival"
# Select: Titanic dataset
# Model: Random Forest

# Expected:
# - Training: 10-30 seconds (XGBoost)
# - Model accepts: {"age": 25, "sex": "male", "class": "3rd"}
# - Returns: {"prediction": 0, "confidence": 0.78}
```

## Key Improvements

✅ **Robust text detection** - High cardinality + long strings = text
✅ **Preserves raw text** - No label encoding for NLP columns
✅ **Task-aware preprocessing** - Different handling for text vs categorical
✅ **Uses original data** - Training loads unprocessed data for text tasks
✅ **No fake data fallbacks** - Real preprocessing, real training, real models

## How It Works

1. **Preprocessing Stage:**
   ```
   Input: {"text": "This movie...", "label": "positive"}
   ↓
   Detect: text column (ratio=0.99, length=200)
   ↓
   Output: {"text": "This movie...", "label": 1}  ← TEXT PRESERVED!
   ```

2. **Training Stage:**
   ```
   Load: original_data.csv (with text preserved)
   ↓
   Detect: text_classification (from TrainerFactory)
   ↓
   Create: SentimentClassifier (DistilBERT)
   ↓
   Train: transformer.train(texts=["This movie..."], labels=[1])
   ↓
   Save: model/ (transformer checkpoint)
   ```

3. **Inference Stage:**
   ```
   Input: {"text": "Great film!"}
   ↓
   Tokenize: [101, 2307, 2143, 999, 102]
   ↓
   Predict: DistilBERT forward pass
   ↓
   Output: {"prediction": "positive", "confidence": 0.95}
   ```

## No More Fake Data!

The system now:
- ✅ Actually trains on your data (not mock/fake data)
- ✅ Uses proper models for each task (transformers for text, XGBoost for tabular)
- ✅ Preserves data integrity (no destructive preprocessing)
- ✅ Realistic training times (2-3 min for transformers, 10-30 sec for XGBoost)
- ✅ Proper input formats (text strings, not floats)
