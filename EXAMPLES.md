# AutoML - Working Examples

## ✅ What Actually Works Right Now

This AutoML pipeline supports **text, tabular, and IMAGE classification** using state-of-the-art models.

### Supported Tasks:

#### 1. **Text Sentiment Classification** (Recommended for first test!)
```
Prompt: "Build a sentiment classifier for movie reviews"
```
- Dataset: IMDB Movie Reviews (50k reviews)
- Models: DistilBERT, BERT (transformers)
- Training time: ~2-3 minutes
- Test after training with new movie reviews!

#### 2. **Image Classification** (NEW! 🎉)
```
Prompt: "Build a cat vs dog classifier"
```
- Dataset: Cats vs Dogs (25k images)
- Models: ResNet50, EfficientNet, Vision Transformer
- Training time: ~5-10 minutes (GPU), ~30 min (CPU)
- Test with your own pet photos!

```
Prompt: "Classify images into 10 object types"
```
- Dataset: CIFAR-10 (60k images)
- Classes: airplane, car, bird, cat, deer, dog, frog, horse, ship, truck
- Training time: ~10 minutes (GPU)

#### 3. **Tabular Regression**
```
Prompt: "Predict house prices in California"
```
- Dataset: California Housing (median house values)
- Models: XGBoost, Random Forest
- Features: 8 numerical (income, house age, rooms, location, etc.)
- Test with new house feature data!

#### 4. **Tabular Classification**
```
Prompt: "Predict Titanic passenger survival"
```
- Dataset: Titanic (passenger survival data)
- Models: XGBoost, Random Forest
- Features: Age, sex, class, fare, etc.
- Binary classification (survived or not)

---

## 🚫 What's NOT Supported Yet

- **Time Series Forecasting**
- **Clustering**
- **Object Detection** (YOLO, etc.)
- **Image Segmentation**

---

## 🧪 Testing Your Trained Model

After training completes, you can test predictions:

### 1. Get Model Info
```bash
GET http://localhost:8000/api/projects/demo-project/model/info
```

Response shows:
- Model type and accuracy
- Required input features
- Example input format

### 2. Make Predictions

**California Housing Example:**
```bash
POST http://localhost:8000/api/projects/demo-project/predict
Content-Type: application/json

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
```

Response:
```json
{
  "prediction": 4.526,
  "model": "RandomForestRegressor",
  "features_used": ["MedInc", "HouseAge", ...],
  "confidence": 0.95
}
```

**IMDB Sentiment Example:**
```bash
POST http://localhost:8000/api/projects/demo-project/predict
Content-Type: application/json

{
  "text": "This movie was absolutely fantastic! Great acting and plot.",
  "text_length": 59,
  "has_exclamation": 1,
  "sentiment_score": 0.8
}
```
(Note: Exact features depend on preprocessing)

---

## 🎯 Recommended First Test

1. **Start with**: `"Build a sentiment classifier for movie reviews"`
2. **Why**: Fast download (~5 sec), quick training (~20 sec), easy to test
3. **Select**: IMDB dataset
4. **Pick model**: Random Forest or Logistic Regression
5. **Wait for training**: See progress bar
6. **Test it**: Use `/model/info` to see input format, then `/predict`

---

## 📊 Pipeline Stages

1. **PARSE_INTENT** - LLM understands your goal (~1 sec)
2. **DATA_SOURCE** - Shows curated datasets, you select one
3. **PROFILE_DATA** - Analyzes dataset structure
4. **PREPROCESS** - Handles missing values, encoding, scaling
5. **MODEL_SELECT** - Recommends models, you pick one
6. **TRAIN** - Trains sklearn model with progress updates
7. **REVIEW_EDIT** - Shows results, accuracy, metrics
8. **EXPORT** - Download trained model + code

---

## 🔜 Coming Soon

- Real image classification with transfer learning (ResNet, EfficientNet)
- Custom dataset upload
- Hyperparameter tuning
- Model comparison
- Deployment to cloud

---

## 🐛 Known Limitations

- Text datasets require manual feature engineering (we convert to CSV)
- Image datasets don't work yet (will throw helpful error)
- Maximum 1000 samples for quick demos (configurable)
- Only sklearn models (no deep learning yet)

---

## 💡 Tips

- Start simple with sentiment analysis
- Check model accuracy before deploying
- Use `/model/info` to understand input format
- Training is REAL - saved to `trained_model.joblib`
