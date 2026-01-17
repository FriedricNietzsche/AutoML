import json
from pathlib import Path
from typing import Dict, Any

import nbformat as nbf

from app.api.assets import ASSET_ROOT


def project_dir(project_id: str) -> Path:
    path = ASSET_ROOT / "projects" / project_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json_asset(project_id: str, rel_path: str, data: Any) -> Path:
    base = project_dir(project_id)
    path = base / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def save_text_asset(project_id: str, rel_path: str, text: str) -> Path:
    base = project_dir(project_id)
    path = base / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def save_notebook_asset(project_id: str, rel_path: str, nb_node: nbf.NotebookNode) -> Path:
    """Persist a Jupyter notebook node to disk."""
    base = project_dir(project_id)
    path = base / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb_node, path)
    return path


def asset_url(path: Path) -> str:
    return f"/api/assets/{path.relative_to(ASSET_ROOT)}"
