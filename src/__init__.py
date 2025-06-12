# -*- coding: utf-8 -*-

"""
Poetry Classifier Package
=========================

This package provides a comprehensive toolkit for multi-label thematic classification
of Arabic poetry. It includes modules for data processing, embedding, segmentation,
model definition, and training.

To get started, you can directly import the main classes from this top-level package.

Example:
    from poetry_classifier import (
        PoetryPreprocessor,
        PoetryEmbedder,
        PoemSegmenter,
        HybridPoetryClassifier,
        ModelTrainer,
        PoetryThematicDataset
    )

This __init__.py file handles:
- Defining package-level metadata like version and author.
- Setting up a default logger for the package.
- Making key classes and functions available at the top level for a cleaner API.
"""

import logging
import os

# --- Package Metadata ---
# This is a good practice for package management and distribution.
__version__ = "1.0.0"
__author__ = "Your Name / Your Team"
__email__ = "your.email@example.com"


# --- Setup a Null Logger ---
# This prevents log messages from being propagated to the root logger if the
# library user has not configured logging. It's a standard practice for libraries.
# The user of the library can then configure logging as they see fit.
logging.getLogger(__name__).addHandler(logging.NullHandler())


# --- Cleaner API: Import key classes to the top level ---
# This allows users to import classes directly from the package,
# e.g., `from poetry_classifier import PoetryPreprocessor`
# instead of `from poetry_classifier.data_processing import PoetryPreprocessor`.

# A try-except block is used here to handle potential circular imports or
# missing dependencies gracefully, although it's less likely with a flat structure.
try:
    from .data_processing import PoetryPreprocessor
    from .embedding import PoetryEmbedder
    from .segmentation import PoemSegmenter
    from .model import HybridPoetryClassifier
    from .dataset import PoetryThematicDataset
    from .trainer import ModelTrainer
    from .utils import setup_logging, set_seed, load_config

except ImportError as e:
    # This error might occur if a dependency is not installed.
    # For example, if `pyfarasa` is missing, `data_processing` might fail to import.
    logging.getLogger(__name__).warning(
        f"Could not import all modules from the poetry_classifier package. "
        f"Please ensure all dependencies are installed. Original error: {e}"
    )

# --- Define what is exposed with `from poetry_classifier import *` ---
# It's a good practice to explicitly define `__all__` to control what gets imported.
__all__ = [
    # Metadata
    "__version__",
    "__author__",
    "__email__",
    
    # Core Classes
    "PoetryPreprocessor",
    "PoetryEmbedder",
    "PoemSegmenter",
    "HybridPoetryClassifier",
    "PoetryThematicDataset",
    "ModelTrainer",
    
    # Utility Functions
    "setup_logging",
    "set_seed",
    "load_config"
]

logger = logging.getLogger(__name__)
logger.info(f"Poetry Classifier package version {__version__} loaded.")