"""
DatasetFinderAgent - Provides curated datasets with optional LLM ranking
Uses curated datasets for common tasks to ensure quality recommendations
"""
from typing import List, Dict, Any
import os
import json

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


class DatasetFinderAgent:
    def __init__(self):
        # Initialize Gemini for intelligent dataset curation (optional)
        self.llm = None
        if GENAI_AVAILABLE:
            gemini_key = os.getenv("GEMINI_API_KEY")
            if gemini_key:
                genai.configure(api_key=gemini_key)
                self.llm = genai.GenerativeModel('gemini-2.0-flash-exp')
                print("[DatasetFinder] ✅ LLM-powered ranking enabled")
            else:
                print("[DatasetFinder] ℹ️ No Gemini key - using curated datasets")
        else:
            print("[DatasetFinder] ℹ️ google-generativeai not installed - using curated datasets")

    def find_datasets(self, user_input: str, task_type: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Find relevant datasets using curated lists with optional LLM ranking.
        
        Args:
            user_input: User's prompt or description
            task_type: Type of task (classification, regression, etc.)
            limit: Maximum number of datasets to return
            
        Returns:
            List of dataset dictionaries with id, name, description, etc.
        """
        print(f"[DatasetFinder] 🔍 Finding datasets for: {user_input[:100]}...")
        print(f"[DatasetFinder] Task type: {task_type}")
        
        # Get curated datasets based on task
        curated = self._get_curated_datasets_for_task(user_input, task_type)
        
        if self.llm and len(curated) > limit:
            # Use LLM to rank and select the best datasets
            selected = self._llm_select_datasets(user_input, task_type, curated, limit)
        else:
            # Just return curated datasets in order
            selected = curated[:limit]
        
        print(f"[DatasetFinder] ✅ Selected {len(selected)} datasets:")
        for i, ds in enumerate(selected, 1):
            print(f"  {i}. {ds['name']} - {ds['description'][:60]}...")
        
        return selected

    def _llm_select_datasets(self, user_input: str, task_type: str, candidates: List[Dict], limit: int) -> List[Dict]:
        """Use LLM to intelligently select the most relevant datasets"""
        try:
            prompt = f"""You are a dataset recommendation expert. The user wants to: "{user_input}"
Task type: {task_type}

Here are available datasets:
{json.dumps([{"name": d["name"], "description": d["description"], "tags": d.get("tags", [])} for d in candidates], indent=2)}

Select the {limit} MOST RELEVANT datasets for this task. Consider:
1. Data modality (image, text, audio, tabular)
2. Task type (classification, regression, etc.)
3. Domain relevance (animals, medical, etc.)

Respond with ONLY a JSON array of dataset names in order of relevance, like: ["dataset1", "dataset2", ...]
"""
            
            response = self.llm.generate_content(prompt)
            selected_names = json.loads(response.text.strip())
            
            # Return datasets in the order selected by LLM
            selected = []
            for name in selected_names[:limit]:
                for ds in candidates:
                    if ds['name'] == name or ds['id'] == name:
                        selected.append(ds)
                        break
            
            return selected if selected else candidates[:limit]
            
        except Exception as e:
            print(f"[DatasetFinder] ⚠️ LLM selection error: {e}")
            return candidates[:limit]

    def _get_curated_datasets_for_task(self, user_input: str, task_type: str) -> List[Dict[str, Any]]:
        """Get a curated list of datasets based on detected task/domain"""
        input_lower = user_input.lower()
        curated = []
        
        # ============================================================
        # IMAGE CLASSIFICATION - DISABLED (Coming Soon!)
        # ============================================================
        # Image classification requires CNN/transfer learning with PyTorch/TensorFlow
        # Current AutoML supports text, tabular, and IMAGE classification!
        
        # IMAGE CLASSIFICATION DATASETS (NOW SUPPORTED with PyTorch!)
        if any(term in input_lower for term in ["image", "photo", "picture", "visual", "vision", "cat", "dog", "animal", "pet", "cifar", "mnist", "face", "object"]):
            curated.extend([
                {
                    "id": "cifar10",
                    "name": "CIFAR-10",
                    "full_name": "cifar10",
                    "description": "60,000 32x32 color images in 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.",
                    "downloads": 1000000,
                    "likes": 1500,
                    "tags": ["image-classification", "computer-vision", "multiclass"]
                },
                {
                    "id": "cats_vs_dogs",
                    "name": "Cats vs Dogs",
                    "full_name": "microsoft/cats_vs_dogs",
                    "description": "25,000 images of cats and dogs for binary classification. Classic computer vision benchmark.",
                    "downloads": 500000,
                    "likes": 800,
                    "tags": ["image-classification", "computer-vision", "binary", "animals"]
                },
                {
                    "id": "fashion_mnist",
                    "name": "Fashion MNIST",
                    "full_name": "fashion_mnist",
                    "description": "70,000 grayscale images of 10 fashion categories (t-shirt, trouser, dress, coat, etc.). Modern MNIST alternative.",
                    "downloads": 800000,
                    "likes": 1000,
                    "tags": ["image-classification", "computer-vision", "fashion"]
                },
            ])
        
        # TEXT CLASSIFICATION DATASETS
        if any(term in input_lower for term in ["text", "sentiment", "review", "nlp", "language", "movie", "news"]):
            curated.extend([
                {
                    "id": "imdb",
                    "name": "IMDB Movie Reviews",
                    "full_name": "imdb",
                    "description": "50,000 movie reviews for sentiment analysis (positive/negative).",
                    "downloads": 500000,
                    "likes": 600,
                    "tags": ["text-classification", "sentiment", "nlp"]
                },
                {
                    "id": "ag_news",
                    "name": "AG News",
                    "full_name": "ag_news",
                    "description": "News articles in 4 classes: World, Sports, Business, Sci/Tech. 120,000 training samples.",
                    "downloads": 300000,
                    "likes": 400,
                    "tags": ["text-classification", "nlp", "news"]
                },
            ])
        
        # TABULAR/REGRESSION DATASETS
        if any(term in input_lower for term in ["tabular", "csv", "regression", "predict", "price", "sales", "house", "housing", "titanic", "survival"]):
            curated.extend([
                {
                    "id": "california_housing",
                    "name": "California Housing",
                    "full_name": "california_housing",
                    "description": "Housing prices in California districts. Regression task with 8 features.",
                    "downloads": 200000,
                    "likes": 300,
                    "tags": ["regression", "tabular", "housing"]
                },
                {
                    "id": "titanic",
                    "name": "Titanic Survival",
                    "full_name": "titanic",
                    "description": "Predict survival on the Titanic. Binary classification with passenger data.",
                    "downloads": 400000,
                    "likes": 500,
                    "tags": ["classification", "tabular", "binary"]
                },
            ])
        
        # If no specific datasets matched, provide helpful examples
        if not curated:
            raise ValueError(
                f"No datasets found for: '{user_input}'\n\n"
                "✅ Supported AutoML tasks:\n"
                "  • Text Classification: 'Build a sentiment classifier for movie reviews'\n"
                "  • Tabular Regression: 'Predict house prices in California'\n"
                "  • Tabular Classification: 'Predict Titanic passenger survival'\n\n"
                "🚫 Not yet supported:\n"
                "  • Image classification (requires CNNs - coming soon!)\n"
                "  • Time series forecasting\n"
                "  • Clustering\n"
            )
        
        return curated
