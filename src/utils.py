from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np

SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "nvda.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "features.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"


def ensure_directories() -> None:
    """Create project directories if they do not exist."""
    for directory in [
        DATA_DIR / "raw",
        DATA_DIR / "processed",
        MODELS_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
        NOTEBOOKS_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int = SEED) -> None:
    """Set deterministic seeds for reproducible runs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def to_serializable(value: Any) -> Any:
    """Convert numpy-heavy structures into JSON-serializable objects."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): to_serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_serializable(item) for item in value]
    return value


def write_json(data: dict[str, Any], path: Path) -> None:
    """Write JSON to disk using consistent formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_serializable(data), indent=2), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    """Read JSON file from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


def format_pct(value: float) -> str:
    """Format decimal value as percentage."""
    return f"{value * 100:.2f}%"
