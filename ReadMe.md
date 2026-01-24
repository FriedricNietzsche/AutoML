# AIAI

## Video Demo of AIAI
https://github.com/user-attachments/assets/50d751b9-205e-4141-9e6a-f58265f890f0

## Overview

**AIAI** is an autonomous system where specialized AI agents collaborate to build, train, and deploy machine learning models. This system uses an **Agentic Orchestration Engine** to reason through the data science process—profiling data, selecting architectures using Chain-of-Thought reasoning, and generating transparent, exportable Jupyter Notebooks.

It features a **chat-first interface** that guides users through the entire ML lifecycle: from raw data to a deployed model, supported by real-time visualization of training metrics.

## Core Architecture

### The Agent Integration
The backend is powered by a team of specialized agents:
- **Intent Agent**: Parses natural language requests (e.g., "Build a churn predictor") into structured ML tasks.
- **Data Profiler Agent**: analyizes datasets to detect missingness, schema types, and distribution skews.
- **Model Selector Agent**: Uses **Zero-Shot Reasoning** to pick the best architecture (e.g., "Use XGBoost because input is tabular with high cardinality").
- **Notebook Generator Agent**: Converts the abstract pipeline steps into clean, executable Python code (Jupyter Notebooks) that users can download.

### The Stage Machine
The system moves through a strict Finite State Machine (FSM) to ensure reliability:
1.  **Ingestion:** Loading data from CSVs, JSON, or external URLs.
2.  **Profiling:** Generating distribution plots, correlation matrices, and quality warnings.
3.  **Preprocessing:** Automatic imputation, scaling, and encoding.
4.  **Training:** Real-time streaming of loss curves and accuracy metrics via WebSockets.
5.  **Export:** Packaging the model and code for deployment.

## Tech Stack

### Backend (AI & Engineering)
*   **Framework:** FastAPI + Uvicorn
*   **Real-time:** Socket.io for live training feedback
*   **LLM Integration:** LangChain, OpenAI API / Google Gemini
*   **ML Libraries:** PyTorch, Scikit-learn, XGBoost, Transformers (HuggingFace)
*   **Processing:** Pandas, NumPy
*   **State Management:** Redis (for pipeline state)

### Frontend (User Interface)
*   **Framework:** React 19 + Vite
*   **Language:** TypeScript
*   **Styling:** Tailwind CSS + Framer Motion (for animations)
*   **Visualizations:** Recharts (loss curves), Three.js (3D elements)
*   **Editor:** Monaco Editor (for viewing generated code)

## Supported AI Models

The platform automatically selects state-of-the-art models based on the data modality:

### Computer Vision
*   **ResNet50:** Transfer learning for robust image classification.
*   **EfficientNet-B0:** For resource-constrained environments.
*   **Vision Transformer (ViT-B/16):** For complex visual pattern recognition.

### Natural Language Processing (NLP)
*   **DistilBERT:** Fine-tuned transformer for sentiment analysis and text classification.
*   **TF-IDF + Logistic Regression:** High-performance fallback for simpler text tasks.

### Tabular / Structured Data
*   **XGBoost:** Extreme Gradient Boosting for classification/regression.
*   **Random Forest:** Ensemble bagging methods.
*   **Gradient Boosting (sklearn):** Standard boosting implementations.
*   **Linear/Logistic Regression:** Baseline models.

## Quick Start

### 1. Backend Setup
```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies (ML heavy)
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your OpenRouter or LLM API keys

# Run the server
uvicorn app.main:app --reload
```
The backend will start at `http://localhost:8000`.

### 2. Frontend Setup
```bash
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev
```
The UI will run at `http://localhost:5173`.

## Project Structure

```text
├── backend/
│   ├── app/
│   │   ├── agents/          # AI Agents (Intent, Profiler, Model Selection)
│   │   ├── api/             # FastAPI Routes
│   │   ├── ml/              # Machine Learning Logic (Vision, Text, Tabular)
│   │   ├── orchestrator/    # Pipeline State Machine & Conductor
│   │   └── ws/              # WebSocket Connection Managers
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/      # React UI Components
│   │   ├── pages/           # Main Application Views
│   │   └── lib/             # Utilities (API, Websockets)
│   └── package.json
└── README.md
```
