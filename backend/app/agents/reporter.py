"""
Reporter / notebook generator.
Uses OpenRouter via simple chat; falls back to template when unavailable.
"""
from typing import Dict, Any, List

import nbformat as nbf
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

from app.events.schema import StageID
from app.llm.langchain_client import get_openrouter_llm
from app.llm.openrouter import OpenRouterClient


def build_context_summary(context: Dict[str, Any]) -> str:
    lines = []
    stage = context.get("stage") or context.get("stage_id") or StageID.REVIEW_EDIT.value
    profile = context.get("profile") or {}
    metrics = context.get("metrics") or {}
    lines.append(f"Stage: {stage}")
    if profile:
        lines.append(f"Profile: rows={profile.get('n_rows')} cols={profile.get('n_cols')} missing={profile.get('missing_pct')}")
    if metrics:
        lines.append("Metrics:")
        for k, v in metrics.items():
            lines.append(f"- {k}: {v}")
    return "\n".join(lines)


def _infer_task_type(context: Dict[str, Any]) -> str:
    metrics = context.get("metrics") or {}
    task_type = (context.get("task_type") or "").lower()
    if task_type:
        return task_type
    if any(k in metrics for k in ["accuracy", "f1"]):
        return "classification"
    if any(k in metrics for k in ["rmse", "r2"]):
        return "regression"
    return "classification"


def _infer_target(context: Dict[str, Any]) -> str:
    if context.get("target"):
        return context["target"]
    profile = context.get("profile") or {}
    columns: List[str] = profile.get("columns") or []
    return columns[-1] if columns else "target"


def _data_source_metadata(context: Dict[str, Any]) -> Dict[str, Any]:
    """Extract data source hints (upload path, Kaggle slug, file) for notebook."""
    source = context.get("source") or {}
    return {
        "data_path": source.get("data_path") or context.get("data_path") or "sample.csv",
        "kaggle_slug": source.get("kaggle_slug") or context.get("kaggle_slug"),
        "kaggle_file": source.get("kaggle_file") or context.get("kaggle_file"),
    }


def _model_block(task_type: str, model_id: str | None) -> str:
    """Return Python code that defines `model` and `metric_name`."""
    model_id = (model_id or "").lower() or ("rf" if task_type == "classification" else "rf_reg")
    if task_type == "regression":
        if model_id in {"linear", "linreg"}:
            return "\n".join(
                [
                    "from sklearn.linear_model import LinearRegression",
                    "model = LinearRegression()",
                    "metric_name = 'rmse'",
                ]
            )
        return "\n".join(
            [
                "from sklearn.ensemble import RandomForestRegressor",
                "model = RandomForestRegressor(n_estimators=300, random_state=42)",
                "metric_name = 'rmse'",
            ]
        )
    # classification
    if model_id in {"logreg", "logistic"}:
        return "\n".join(
            [
                "from sklearn.linear_model import LogisticRegression",
                "model = LogisticRegression(max_iter=1000)",
                "metric_name = 'accuracy'",
            ]
        )
    return "\n".join(
        [
            "from sklearn.ensemble import RandomForestClassifier",
            "model = RandomForestClassifier(n_estimators=300, random_state=42)",
            "metric_name = 'accuracy'",
        ]
    )


def build_executable_notebook(context: Dict[str, Any]) -> nbf.NotebookNode:
    """Create a runnable notebook that re-trains the model on the user's machine."""
    summary = build_context_summary(context)
    task_type = _infer_task_type(context)
    target = _infer_target(context)
    model_id = (context.get("model_id") or context.get("chosen_model") or "auto")
    source_meta = _data_source_metadata(context)
    kaggle_slug = source_meta["kaggle_slug"]
    kaggle_file = source_meta["kaggle_file"]
    data_path = source_meta["data_path"]

    nb = new_notebook()
    nb.cells.append(new_markdown_cell("# AutoML Run\n\nGenerated notebook to re-train locally."))
    nb.cells.append(
        new_markdown_cell(
            f"## Context\n{summary}\n\n- Task type: **{task_type}**\n- Target column (settable): `{target}`\n"
            f"- Model: `{model_id}`\n- Data source: {'Kaggle '+kaggle_slug if kaggle_slug else data_path}"
        )
    )

    imports_block = [
        "import os, glob",
        "import pandas as pd",
        "from sklearn.model_selection import train_test_split",
        "from sklearn.compose import ColumnTransformer",
        "from sklearn.pipeline import Pipeline",
        "from sklearn.preprocessing import OneHotEncoder",
        "from sklearn.impute import SimpleImputer",
        "from sklearn.metrics import (",
        "    accuracy_score, f1_score, classification_report, confusion_matrix,",
        "    mean_squared_error, r2_score",
        ")",
        "import joblib",
    ]
    nb.cells.append(new_code_cell("\n".join(imports_block)))

    if kaggle_slug:
        kaggle_block = [
            "# Kaggle download (requires KAGGLE_USERNAME/KAGGLE_KEY env vars)",
            "import kaggle",
            "download_dir = 'data'",
            "os.makedirs(download_dir, exist_ok=True)",
            f"slug = '{kaggle_slug}'",
            f"preferred_file = '{kaggle_file}'" if kaggle_file else "preferred_file = None",
            "kaggle.api.authenticate()",
            "kaggle.api.dataset_download_files(slug, path=download_dir, unzip=True, quiet=False)",
            "if preferred_file:",
            "    DATA_PATH = os.path.join(download_dir, preferred_file)",
            "else:",
            "    csvs = glob.glob(os.path.join(download_dir, '*.csv'))",
            "    if not csvs:",
            "        raise FileNotFoundError('No CSV found after Kaggle download')",
            "    DATA_PATH = csvs[0]",
            "print('Using DATA_PATH', DATA_PATH)",
        ]
    else:
        kaggle_block = [
            "# Local/previously downloaded data",
            f"DATA_PATH = os.getenv('DATA_PATH', '{data_path}')",
            "print('Using DATA_PATH', DATA_PATH)",
        ]
    nb.cells.append(new_code_cell("\n".join(kaggle_block)))

    nb.cells.append(
        new_code_cell(
            "\n".join(
                [
                    f"TARGET = os.getenv('TARGET_COLUMN', '{target}')",
                    "df = pd.read_csv(DATA_PATH)",
                    "print('Loaded dataset with shape:', df.shape)",
                    "print(df.head())",
                    "if TARGET not in df.columns:",
                    "    raise ValueError(f'TARGET column {TARGET} not found. Columns: {df.columns.tolist()}')",
                    "X = df.drop(columns=[TARGET])",
                    "y = df[TARGET]",
                    "categorical = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()",
                    "numeric = [c for c in X.columns if c not in categorical]",
                    "print('Numeric columns:', numeric)",
                    "print('Categorical columns:', categorical)",
                ]
            )
        )
    )

    nb.cells.append(
        new_code_cell(
            "\n".join(
                [
                    "categorical_transformer = Pipeline(steps=[",
                    "    ('imputer', SimpleImputer(strategy='most_frequent')),",
                    "    ('encoder', OneHotEncoder(handle_unknown='ignore'))",
                    "])",
                    "numeric_transformer = Pipeline(steps=[",
                    "    ('imputer', SimpleImputer(strategy='median'))",
                    "])",
                    "preprocess = ColumnTransformer(",
                    "    transformers=[",
                    "        ('categorical', categorical_transformer, categorical),",
                    "        ('numeric', numeric_transformer, numeric),",
                    "    ]",
                    ")",
                ]
            )
        )
    )

    nb.cells.append(new_code_cell(_model_block(task_type, model_id)))

    if task_type == "regression":
        eval_block = "\n".join(
            [
                "preds = clf.predict(X_test)",
                "rmse = mean_squared_error(y_test, preds, squared=False)",
                "r2 = r2_score(y_test, preds)",
                "print('RMSE:', rmse)",
                "print('R2:', r2)",
            ]
        )
    else:
        eval_block = "\n".join(
            [
                "preds = clf.predict(X_test)",
                "acc = accuracy_score(y_test, preds)",
                "f1 = f1_score(y_test, preds, average='weighted')",
                "print('Accuracy:', acc)",
                "print('F1:', f1)",
                "print(classification_report(y_test, preds))",
            ]
        )

    nb.cells.append(
        new_code_cell(
            "\n".join(
                [
                    "clf = Pipeline(steps=[('preprocess', preprocess), ('model', model)])",
                    "X_train, X_test, y_train, y_test = train_test_split(",
                    "    X, y, test_size=0.2, random_state=42, stratify=y if metric_name=='accuracy' else None",
                    ")",
                    "clf.fit(X_train, y_train)",
                    eval_block,
                ]
            )
        )
    )

    nb.cells.append(
        new_code_cell(
            "\n".join(
                [
                    "joblib.dump(clf, 'model.joblib')",
                    "print('Saved trained model to model.joblib')",
                ]
            )
        )
    )

    return nb


class ReporterAgent:
    def __init__(self, model: str | None = None):
        self.client = OpenRouterClient(model=model)
        self.llm = get_openrouter_llm(model=model, temperature=0.2, timeout=30)
        self.report_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are an AutoML reporter. Write a concise (<=200 words) summary of profile + metrics."),
                ("user", "Context:\n{summary}\n\nReturn concise prose only."),
            ]
        )
        self.report_parser = StrOutputParser()

    def generate(self, context: Dict[str, Any]) -> Dict[str, str]:
        summary = build_context_summary(context)
        notebook_nb = build_executable_notebook(context)

        # Preferred: LangChain chain if available
        if self.llm:
            try:
                chain = self.report_prompt | self.llm | self.report_parser
                text = chain.invoke({"summary": summary})
                return {"notebook_nb": notebook_nb, "report": text}
            except Exception:
                pass

        # Fallback to plain OpenRouter client if available
        if self.client.available():
            prompt = (
                "You are an AutoML reporter. Produce a concise prose summary of dataset profile and training metrics."
                " Keep it short (<=200 words) and action-oriented.\n\n"
                f"Context:\n{summary}"
            )
            try:
                text = self.client.chat_text(
                    [
                        {"role": "system", "content": "AutoML reporter."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                )
                return {"notebook_nb": notebook_nb, "report": text}
            except Exception:
                pass

        # Deterministic fallback
        report = f"# AutoML Report\n\n{summary}\n\n(LLM unavailable; deterministic report.)\n"
        return {"notebook_nb": notebook_nb, "report": report}
