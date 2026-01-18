"""
Notebook Generator Agent - Creates tailored inference notebooks using AI

Uses OpenRouter to generate custom Jupyter notebooks with:
- Specific model loading code
- Dataset-specific examples
- Task-appropriate inference examples
- Best practices and tips
"""
import os
import json
import requests
from typing import Dict, Any, List
from pathlib import Path


class NotebookGeneratorAgent:
    def __init__(self):
        self.openrouter_key = os.getenv("OPENROUTER_API_KEY")
        self.model = os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-flash")
        
    def generate_notebook(
        self,
        task_type: str,
        model_name: str,
        dataset_name: str,
        feature_names: List[str],
        target_column: str,
        metrics: Dict[str, Any],
        model_path: str,
        sample_data: Dict[str, Any] = None
    ) -> Dict:
        """
        Generate a custom inference notebook using AI
        
        Args:
            task_type: Type of task (classification, regression, etc.)
            model_name: Name of the trained model
            dataset_name: Name of the dataset used
            feature_names: List of feature column names
            target_column: Name of the target column
            metrics: Training metrics
            model_path: Path to the saved model file
            sample_data: Optional sample data row for examples
            
        Returns:
            Jupyter notebook structure (dict with cells)
        """
        print(f"[NotebookGenerator] Generating custom notebook for {model_name} on {dataset_name}")
        
        # Prepare context for AI
        context = self._prepare_context(
            task_type, model_name, dataset_name, feature_names, 
            target_column, metrics, model_path, sample_data
        )
        
        # Generate notebook content using AI
        notebook_cells = self._generate_cells_with_ai(context)
        
        # Create notebook structure
        notebook = {
            "cells": notebook_cells,
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                },
                "language_info": {
                    "codemirror_mode": {"name": "ipython", "version": 3},
                    "file_extension": ".py",
                    "mimetype": "text/x-python",
                    "name": "python",
                    "nbconvert_exporter": "python",
                    "pygments_lexer": "ipython3",
                    "version": "3.10.0"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 5
        }
        
        return notebook
    
    def _prepare_context(
        self,
        task_type: str,
        model_name: str,
        dataset_name: str,
        feature_names: List[str],
        target_column: str,
        metrics: Dict[str, Any],
        model_path: str,
        sample_data: Dict[str, Any] = None
    ) -> str:
        """Prepare context string for AI prompt"""
        
        # Get model file name
        model_file = Path(model_path).name
        
        # Format metrics
        metrics_str = "\n".join([f"  - {k}: {v}" for k, v in metrics.items() 
                                 if k not in ["model_name", "task_type", "dataset"]])
        
        # Create sample data example
        if sample_data and len(sample_data) > 0:
            sample_example = json.dumps(sample_data, indent=2)
        else:
            # Create dummy sample based on feature names
            sample_example = "{\n"
            for i, feat in enumerate(feature_names[:5]):
                sample_example += f'  "{feat}": <example_value>,\n'
            if len(feature_names) > 5:
                sample_example += "  ...\n"
            sample_example += "}"
        
        context = f"""
Task: Create a Jupyter notebook for model inference

Dataset: {dataset_name}
Model: {model_name}
Task Type: {task_type}
Target Column: {target_column}
Number of Features: {len(feature_names)}

Feature Names (first 10):
{', '.join(feature_names[:10])}{"..." if len(feature_names) > 10 else ""}

Model Performance:
{metrics_str}

Model File: {model_file}
Model Format: joblib dictionary with keys: 'model', 'feature_names', 'task_type'

Sample Data Structure:
{sample_example}
"""
        return context
    
    def _generate_cells_with_ai(self, context: str) -> List[Dict]:
        """Use AI to generate notebook cells"""
        
        prompt = f"""You are an expert at creating Jupyter notebooks for machine learning inference.

Create a Jupyter notebook with code cells and markdown cells that shows how to:
1. Load the trained model from a joblib file
2. Load and explore the model metadata files
3. Make predictions on new data
4. Interpret the results

IMPORTANT REQUIREMENTS:
- The model is saved as a dictionary: {{'model': sklearn_model, 'feature_names': list, 'task_type': str}}
- You MUST extract the model: model_data = joblib.load('file'); model = model_data['model']
- Two metadata files are included: model_metadata.json (JSON) and model_metadata.pkl (pickle)
- Show how to load the metadata to get feature names and model parameters
- Use the EXACT feature names provided in the context
- Create realistic example data based on the actual features
- Include helpful comments and explanations
- Add markdown cells for structure and documentation

Context:
{context}

Additional Files Available:
- model_metadata.json: Contains feature names, types, parameters, preprocessing info, and metrics
- model_metadata.pkl: Same information in Python pickle format

Return a JSON array of notebook cells in this EXACT format:
[
  {{
    "cell_type": "markdown",
    "source": ["# Title\\n", "Description text"]
  }},
  {{
    "cell_type": "code",
    "execution_count": null,
    "metadata": {{}},
    "outputs": [],
    "source": ["import pandas as pd\\n", "import joblib"]
  }}
]

Make the notebook practical, well-documented, and ready to use.
Include at least:
- Title markdown
- Imports code cell
- Load model code cell (with dictionary extraction!)
- Load metadata code cell (show both JSON and pickle methods)
- Explore metadata markdown (explain what's in the metadata)
- Example prediction code cell
- Results interpretation markdown

Return ONLY valid JSON, no other text.
"""

        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.openrouter_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/AutoML",
                    "X-Title": "AutoML Notebook Generator"
                },
                json={
                    "model": self.model,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "temperature": 0.3,  # Lower temperature for more consistent code
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"].strip()
                
                # Extract JSON from response (might be wrapped in ```json```)
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()
                
                cells = json.loads(content)
                print(f"[NotebookGenerator] ✅ Generated {len(cells)} cells using AI")
                return cells
            else:
                print(f"[NotebookGenerator] ⚠️ OpenRouter error: {response.status_code}")
                print(f"[NotebookGenerator] Response: {response.text}")
                return self._fallback_cells()
                
        except Exception as e:
            print(f"[NotebookGenerator] ❌ Error generating with AI: {e}")
            import traceback
            traceback.print_exc()
            return self._fallback_cells()
    
    def _fallback_cells(self) -> List[Dict]:
        """Fallback notebook cells if AI generation fails"""
        return [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Model Inference Notebook\n",
                    "\n",
                    "This notebook shows how to load and use the trained model.\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import joblib\n",
                    "import pandas as pd\n",
                    "import numpy as np\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Load the trained model\n",
                    "model_data = joblib.load('trained_model.joblib')\n",
                    "\n",
                    "# Extract model and metadata\n",
                    "if isinstance(model_data, dict):\n",
                    "    model = model_data['model']\n",
                    "    feature_names = model_data.get('feature_names', [])\n",
                    "    print(f\"Model loaded: {type(model).__name__}\")\n",
                    "    print(f\"Features: {len(feature_names)}\")\n",
                    "else:\n",
                    "    model = model_data\n",
                    "    print(f\"Model loaded: {type(model).__name__}\")\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Make Predictions\n",
                    "\n",
                    "Replace the example data with your actual data.\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Example prediction\n",
                    "# Replace with your actual data\n",
                    "new_data = pd.DataFrame({\n",
                    "    # Add your features here\n",
                    "})\n",
                    "\n",
                    "# Make predictions\n",
                    "predictions = model.predict(new_data)\n",
                    "print(\"Predictions:\", predictions)\n"
                ]
            }
        ]
