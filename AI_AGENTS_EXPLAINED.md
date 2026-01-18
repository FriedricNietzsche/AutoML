# AI Agent Integration - Complete Explanation

## Question 1: How Does Prompt → Dataset Finder Flow Work?

### The Complete Flow:

```
USER PROMPT: "Build me a classifier for cat/dog"
    ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 1: PROMPT PARSER AGENT (AI-Powered via LangChain)      │
├─────────────────────────────────────────────────────────────┤
│ File: backend/app/agents/prompt_parser.py                   │
│ Model: Llama 3.1 8B via OpenRouter API                      │
│                                                               │
│ Input: "Build me a classifier for cat/dog"                  │
│ Processing:                                                   │
│   - Sends to LangChain with structured output parser        │
│   - Extracts: task classification, target, hints            │
│                                                               │
│ Output (Structured JSON):                                    │
│ {                                                             │
│   "task_type": "classification",                            │
│   "target": "cat vs dog image",                             │
│   "dataset_hint": "cats and dogs images",                   │
│   "constraints": {}                                          │
│ }                                                             │
└─────────────────────────────────────────────────────────────┘
    ↓ (passes to Dataset Finder)
┌─────────────────────────────────────────────────────────────┐
│ STEP 2: DATASET FINDER AGENT (HuggingFace API)              │
├─────────────────────────────────────────────────────────────┤
│ File: backend/app/agents/dataset_finder.py                  │
│ API: HuggingFace Hub Python SDK                             │
│                                                               │
│ Receives from Prompt Parser:                                 │
│   task_type = "classification"                              │
│   dataset_hint = "cats and dogs images"                     │
│                                                               │
│ Search Strategy (line 109-115):                             │
│   1. Takes dataset_hint: "cats and dogs images"             │
│   2. Adds task-specific keywords for "classification":      │
│      ["classification", "labeled", "categories"]            │
│   3. Builds search query:                                    │
│      "cats and dogs images classification labeled categories"│
│                                                               │
│ HuggingFace API Call (line 118-125):                        │
│   api.list_datasets(                                         │
│     search="cats and dogs images classification...",        │
│     limit=10,                                                │
│     sort="downloads",  # Most popular first                 │
│     direction=-1                                             │
│   )                                                           │
│                                                               │
│ Returns: List of datasets from HF Hub                        │
│   Example: microsoft/cats_vs_dogs, Oxford-IIIT Pet, etc.    │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 3: LICENSE VALIDATOR                                    │
├─────────────────────────────────────────────────────────────┤
│ File: backend/app/agents/license_validator.py               │
│                                                               │
│ For each dataset found:                                      │
│   - Extracts license tag (e.g., "mit", "apache-2.0")        │
│   - Checks against allowed list                             │
│   - Rejects GPL, proprietary, etc.                          │
│                                                               │
│ Output: Filtered list with only valid licenses              │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 4: AUTO-SELECT BEST DATASET                            │
├─────────────────────────────────────────────────────────────┤
│ Sorting criteria:                                            │
│   1. License valid? (Yes first)                             │
│   2. Downloads (Most popular first)                          │
│                                                               │
│ Selects: microsoft/cats_vs_dogs (MIT license, 50K downloads)│
└─────────────────────────────────────────────────────────────┘
```

### Code References:

**demo.py (line 60-75):**
```python
# 1. Parse prompt
parser = PromptParserAgent()
parsed = parser.parse("Build me a classifier for cat/dog")
# Returns: {task_type: "classification", dataset_hint: "cats and dogs images"}

# 2. Search datasets
task_type = parsed.get("task_type")  # "classification"
dataset_hint = parsed.get("dataset_hint")  # "cats and dogs images"

finder = DatasetFinderAgent()
candidates = finder.find_datasets(
    task_type=task_type,        # Used to add task-specific keywords
    dataset_hint=dataset_hint,   # Added to search query
    max_results=5
)
```

**dataset_finder.py (line 107-117):**
```python
# Build search query from hints
search_terms = []
if dataset_hint:
    search_terms.append(dataset_hint)  # "cats and dogs images"

# Add task keywords
task_keywords = {
    "classification": ["classification", "labeled", "categories"],
    "vision": ["image", "vision", "visual"],
}
search_terms.extend(task_keywords.get(task_type, []))

query = " ".join(search_terms)
# Final: "cats and dogs images classification labeled categories"
```

---

## Question 2: Where is AI Used in Model Selector?

### Current Status: **NOT USING AI YET** (Rule-Based)

**File**: `backend/app/agents/model_selector.py`

Current implementation uses **if/else rules**:
```python
def select_model(self, task_type: str):
    if task_type == "vision":
        return [
            {"id": "cnn", "name": "CNN", "pros": ["Good for images"]},
            {"id": "resnet", "name": "ResNet", "pros": ["State of the art"]}
        ]
    elif task_type == "classification":
        return [{"id": "random_forest", "name": "Random Forest"}]
```

### How to Add AI (Recommendation):

**Option 1: Use LangChain to Decide**
```python
# In model_selector.py
from langchain_openai import ChatOpenAI
from app.config import Config

class ModelSelectorAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=Config.DEFAULT_MODEL,
            openai_api_key=Config.OPENROUTER_API_KEY,
            openai_api_base=Config.OPENROUTER_BASE_URL,
        )
    
    def select_model(self, task_type: str, dataset_size: int, constraints: dict):
        prompt = f"""
        Given:
        - Task: {task_type}
        - Dataset size: {dataset_size} samples
        - Constraints: {constraints}
        
        Recommend the best ML model. Return JSON:
        {{
            "model": "model_name",
            "reason": "why this model is best",
            "pros": ["advantage 1", "advantage 2"],
            "cons": ["limitation 1"]
        }}
        """
        
        result = self.llm.invoke(prompt)
        return parse_json(result.content)
```

**Current**: Rule-based (fast, deterministic, no AI cost)
**Future**: Can add LangChain-powered model selection

---

## Question 3: Where are AI Agents Used in Trainer/Verifier?

### Current AI Agent Usage Map:

```
┌──────────────────────────────────────────────────────────────┐
│ COMPONENT            │ AI USED?  │ IMPLEMENTATION             │
├──────────────────────────────────────────────────────────────┤
│ PromptParserAgent    │ ✅ YES    │ LangChain + Llama 3.1 8B   │
│ DatasetFinderAgent   │ ✅ YES    │ HuggingFace API search     │
│ LicenseValidator     │ ❌ NO     │ Rule-based (license list)  │
│ ModelSelectorAgent   │ ❌ NO     │ Rule-based (if/else)       │
│ TabularTrainer       │ ❌ NO     │ Sklearn (RandomForest)     │
│ ImageTrainer         │ ❌ NO     │ TF/PyTorch (CNN training)  │
│ Verifier             │ ❌ NO     │ Simple validation checks   │
└──────────────────────────────────────────────────────────────┘
```

### Detailed Breakdown:

**1. Trainer (backend/app/ml/trainers/)**
- **tabular_trainer.py**: Uses sklearn pipelines (RandomForest, XGBoost)
- **image_trainer.py**: Uses TensorFlow/PyTorch for CNN training
- **NO AI AGENTS**: These use traditional ML libraries

**Why?** Training itself doesn't need LLMs - we need domain-specific ML algorithms

**2. Verifier (backend/app/agents/verifier.py)**
```python
class VerifierAgent:
    def verify(self, ...):
        # Basic checks: file exists, columns match, no nulls, etc.
        # NO AI - just validation logic
```

**Could add AI**: Use LangChain to suggest fixes for data quality issues

---

## Question 4: Complete AI Agent Integration Summary

### Where AI is ACTUALLY Used:

```
USER INPUT
    ↓
┌─────────────────────────────────────────────────────────────┐
│ AI AGENT #1: PROMPT PARSER                                   │
│ Technology: LangChain + OpenRouter (Llama 3.1 8B)           │
│ Purpose: Understand natural language → structured intent    │
│ File: backend/app/agents/prompt_parser.py                   │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ AI AGENT #2: DATASET FINDER                                  │
│ Technology: HuggingFace Hub API                             │
│ Purpose: Search 1000s of datasets based on AI-parsed intent │
│ File: backend/app/agents/dataset_finder.py                  │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ RULE-BASED: LICENSE VALIDATOR                                │
│ Technology: Python list matching                            │
│ Purpose: Legal compliance - check licenses                  │
│ File: backend/app/agents/license_validator.py               │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ RULE-BASED: MODEL SELECTOR (Could be AI)                    │
│ Technology: If/else rules                                   │
│ Purpose: Pick ML model for task                             │
│ File: backend/app/agents/model_selector.py                  │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ TRADITIONAL ML: TRAINER                                      │
│ Technology: Sklearn, TensorFlow, PyTorch                    │
│ Purpose: Actual model training                              │
│ Files: backend/app/ml/trainers/*.py                         │
└─────────────────────────────────────────────────────────────┘
```

### The Value of AI Agents:

**What AI Does Well:**
1. ✅ Understanding natural language (Prompt Parser)
2. ✅ Searching large databases intelligently (Dataset Finder uses HF's AI-powered search)
3. ✅ Making recommendations based on context

**What AI Doesn't Need to Do:**
1. ❌ License validation (simple list matching is faster/cheaper)
2. ❌ Actual ML training (use specialized libraries like sklearn)
3. ❌ Data validation (rule-based is more reliable)

---

## Improving AI Integration (Recommendations):

### Short Term (Easy Wins):
1. ✅ **Already Done**: Prompt parsing with LangChain
2. ✅ **Already Done**: Dataset search with HF API
3. 🔄 **In Progress**: WebSocket streaming for real-time updates

### Medium Term (Add More AI):
1. **Model Selector with AI**:
   - Use LangChain to recommend best model based on dataset characteristics
   - Consider compute constraints, time limits, accuracy goals

2. **Hyperparameter Tuning with AI**:
   - Use LangChain to suggest good hyperparameters
   - Learn from past runs (store results in DB)

3. **Error Analysis with AI**:
   - If training fails, use LLM to suggest fixes
   - "Your dataset is too small - try data augmentation"

### Long Term (Advanced):
1. **Custom Model Architecture Generation**:
   - Use LLM to generate PyTorch/TensorFlow code
   - Auto-adjust architecture based on data shape

2. **Automated Debugging**:
   - When errors occur, LLM analyzes stack trace
   - Suggests fixes or workarounds

3. **User Guidance**:
   - Chatbot that explains what's happening at each stage
   - Answers questions about model performance

---

## Testing the Integration

Run this to verify AI agents work:
```bash
cd backend
python3 test_agents.py
```

Expected output:
```
✓ Prompt Parser: "Build cat/dog classifier" → classification
✓ License Validator: MIT ✓, GPL ✗
✓ Dataset Finder: Searches HuggingFace Hub
```

---

## Summary

**AI is used for:**
- 🤖 Prompt understanding (LangChain + Llama 3.1)
- 🤖 Dataset discovery (HuggingFace Hub API)

**AI is NOT used for (and doesn't need to be):**
- License validation (rules work better)
- Model training (sklearn/TF/PyTorch are specialized)
- Data validation (deterministic checks are faster)

**The sweet spot**: Use AI where understanding/search is needed, use traditional code where determinism/speed matters.
