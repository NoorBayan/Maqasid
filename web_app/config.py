# -*- coding: utf-8 -*-

import os
import logging
from pathlib import Path

# Configure a logger specific to the configuration module
logger = logging.getLogger(__name__)

# --- Dynamic Path Configuration ---
# This is the most robust way to handle paths. It ensures that no matter where
# you run the app from (e.g., from the root directory or from within web_app),
# the paths to your data and models will always be correct.

try:
    # The base directory of the entire project (the parent of 'src' and 'web_app')
    # It goes up two levels from this file's location (web_app/config.py -> web_app -> project_root)
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
except NameError:
    # Fallback for interactive environments like Jupyter notebooks where __file__ is not defined
    PROJECT_ROOT = Path('.').resolve()

logger.info(f"Project Root determined as: {PROJECT_ROOT}")


# --- Helper function for path validation ---
def validate_path(path: Path, description: str):
    """Checks if a given path exists and logs a warning if it doesn't."""
    if not path.exists():
        logger.warning(
            f"Configuration Warning: The path for '{description}' does not exist. "
            f"Path: '{path}'"
        )
    return path


# ==============================================================================
#  1. MODEL AND DATA PATHS CONFIGURATION
# ==============================================================================
# All paths are constructed relative to the PROJECT_ROOT.

# --- Saved Models Paths ---
SAVED_MODELS_DIR = PROJECT_ROOT / "saved_models"

# Classifier model
CLASSIFIER_MODEL_PATH = validate_path(
    SAVED_MODELS_DIR / "classifier" / "best_poetry_classifier.pth",
    "Classifier Model Weights"
)
CLASSIFIER_CONFIG_PATH = validate_path(
    SAVED_MODELS_DIR / "classifier" / "model_config.json",
    "Classifier Model Configuration"
)

# Embedding model
EMBEDDING_MODEL_PATH = validate_path(
    SAVED_MODELS_DIR / "embeddings" / "fasttext_poetry.bin",
    "FastText Embedding Model"
)


# --- Data and Schema Paths ---
DATA_DIR = PROJECT_ROOT / "data"

# The main annotated corpus for the exploration dashboard
ANNOTATED_CORPUS_PATH = validate_path(
    DATA_DIR / "annotated" / "diwan_corpus_annotated.csv",
    "Annotated Corpus CSV"
)

# The hierarchical schema file for mapping labels to names
LABEL_SCHEMA_PATH = validate_path(
    DATA_DIR / "schema" / "thematic_schema.json",
    "Thematic Schema JSON"
)

# Example poems for the analysis page
EXAMPLE_POEMS_PATH = validate_path(
    PROJECT_ROOT / "web_app" / "static" / "example_poems.json",
    "Example Poems JSON"
)


# ==============================================================================
#  2. MODEL AND PREDICTION PARAMETERS
# ==============================================================================

# --- Segmentation Parameters ---
# Hyperparameters for the DBSCAN segmenter used in the predictor.
# These should ideally match the values that yielded the best results in your research.
SEGMENTER_EPS = 0.4
SEGMENTER_MIN_SAMPLES = 1


# --- Classification Parameters ---
# The confidence threshold for a theme to be considered "predicted".
# A value between 0.0 and 1.0.
CLASSIFICATION_THRESHOLD = 0.5


# ==============================================================================
#  3. WEB APPLICATION UI CONFIGURATION
# ==============================================================================

# --- Page Titles and Icons ---
# Centralize UI elements for consistency across the app.
APP_TITLE = "Arabic Poetry Thematic Classifier"
APP_ICON = "📜"

# Page-specific configurations
PAGE_CONFIG = {
    "home": {
        "title": APP_TITLE,
        "icon": APP_ICON
    },
    "analyze_poem": {
        "title": "Analyze a Poem",
        "icon": "✍️"
    },
    "explore_corpus": {
        "title": "Explore The Corpus",
        "icon": "📚"
    }
}


# --- UI Defaults ---
# Default values for interactive widgets.
DEFAULT_TOP_K_THEMES = 7
MAX_TOP_K_THEMES = 20


# --- External Links ---
# Centralize URLs for easy updates.
RESEARCH_PAPER_URL = "https://your-paper-link.com"
GITHUB_REPO_URL = "https://github.com/your-username/your-repo"

# ==============================================================================
#  END OF CONFIGURATION
# ==============================================================================

# You could add a final check here if needed, for example:
# if not all([CLASSIFIER_MODEL_PATH.exists(), CLASSIFIER_CONFIG_PATH.exists()]):
#     raise FileNotFoundError("Critical model files are missing. The application cannot start.")