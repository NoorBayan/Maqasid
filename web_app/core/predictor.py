# -*- coding: utf-8 -*-

import json
import logging
import os
import sys
from typing import List, Dict, Any

import numpy as np
import streamlit as st
import torch

# Add project root to path for src imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.poetry_classifier.data_processing import PoetryPreprocessor
from src.poetry_classifier.embedding import PoetryEmbedder
from src.poetry_classifier.model import HybridPoetryClassifier
from src.poetry_classifier.segmentation import PoemSegmenter

# Import configurations
from web_app.config import (
    MODEL_PATH, CONFIG_PATH, EMBEDDING_PATH, LABEL_MAPPING_PATH,
    SEGMENTER_EPS, SEGMENTER_MIN_SAMPLES, CLASSIFICATION_THRESHOLD
)

# Configure logger
logger = logging.getLogger(__name__)


class PoetryPredictor:
    """
    A professional, robust, and efficient predictor class for the Arabic Poetry Classifier.

    This class encapsulates the entire prediction pipeline, from text preprocessing to
    returning formatted thematic predictions. It is designed to be a singleton-like
    resource within a Streamlit application, loading heavy models only once.
    """

    def __init__(self):
        """
        Initializes the predictor by loading all necessary components (preprocessor, models, etc.).
        Relies on Streamlit's caching mechanisms passed via factory functions.
        """
        logger.info("Initializing PoetryPredictor...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load components using factory methods decorated with Streamlit caching
        self.preprocessor = self._load_preprocessor()
        self.embedder = self._load_embedder()
        self.segmenter = self._load_segmenter()
        self.classifier = self._load_classifier()
        self.label_map = self._load_label_map()
        
        logger.info(f"PoetryPredictor initialized successfully on device: {self.device}")

    # The following methods use Streamlit's caching.
    # The `_self` argument is a convention to make these methods cacheable
    # when called from an instance.
    @st.cache_resource
    def _load_preprocessor(_self) -> PoetryPreprocessor:
        logger.info("Loading PoetryPreprocessor...")
        return PoetryPreprocessor()

    @st.cache_resource
    def _load_embedder(_self) -> PoetryEmbedder:
        logger.info(f"Loading PoetryEmbedder from {EMBEDDING_PATH}...")
        embedder = PoetryEmbedder()
        embedder.load_model(EMBEDDING_PATH)
        return embedder

    @st.cache_resource
    def _load_segmenter(_self) -> PoemSegmenter:
        logger.info(f"Loading PoemSegmenter with eps={SEGMENTER_EPS}, min_samples={SEGMENTER_MIN_SAMPLES}...")
        return PoemSegmenter(eps=SEGMENTER_EPS, min_samples=SEGMENTER_MIN_SAMPLES)

    @st.cache_resource
    def _load_classifier(_self) -> HybridPoetryClassifier:
        logger.info(f"Loading HybridPoetryClassifier from {MODEL_PATH}...")
        model = HybridPoetryClassifier.from_pretrained(MODEL_PATH, CONFIG_PATH, device=_self.device)
        return model

    @st.cache_data
    def _load_label_map(_self) -> Dict[int, str]:
        logger.info(f"Loading label map from {LABEL_MAPPING_PATH}...")
        try:
            # This is a placeholder for how you might load your schema.
            # Assuming the JSON has a structure that can be mapped to {index: name}.
            # You will need to adapt this to your actual schema file structure.
            with open(LABEL_MAPPING_PATH, 'r', encoding='utf-8') as f:
                schema = json.load(f)
            # Example of extracting leaf nodes from a hierarchical schema
            label_map = {}
            index = 0
            def extract_leaves(node):
                nonlocal index
                if "subcategories" not in node or not node["subcategories"]:
                    label_map[index] = node["name"]
                    index += 1
                else:
                    for sub_node in node["subcategories"]:
                        extract_leaves(sub_node)
            
            # Assuming your schema is a list of root nodes
            for root_node in schema:
                 extract_leaves(root_node)

            if not label_map:
                raise ValueError("Label map is empty. Check schema file and parsing logic.")
                
            return label_map
        except (FileNotFoundError, KeyError, ValueError) as e:
            logger.error(f"Failed to load or parse the label map: {e}")
            # Fallback to generic labels if loading fails
            num_classes = _self._load_classifier().config['num_classes']
            return {i: f"Theme_{i+1}" for i in range(num_classes)}

    def _preprocess_text(self, text: str) -> List[str]:
        """Splits text into verses and applies preprocessing."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input text must be a non-empty string.")
        
        verses = [v.strip() for v in text.split('\n') if v.strip()]
        if not verses:
            raise ValueError("No valid verses found in the input text after cleaning.")
            
        return [self.preprocessor.process_text(v) for v in verses]

    def _embed_and_segment(self, processed_verses: List[str]) -> np.ndarray:
        """
        Embeds verses, segments them, and returns a single feature vector for the poem.
        This part can be enhanced to produce segment-level features.
        """
        if not processed_verses:
            raise ValueError("Cannot embed an empty list of processed verses.")
            
        # For simplicity, we create a single feature vector for the whole poem.
        # This averages all verse embeddings.
        full_processed_text = " ".join(processed_verses)
        poem_embedding = self.embedder.get_document_vector(full_processed_text)
        
        if poem_embedding is None:
            raise RuntimeError("Embedding model failed to produce a vector.")
            
        return poem_embedding

    def _classify(self, feature_vector: np.ndarray) -> np.ndarray:
        """Runs the feature vector through the classifier model."""
        # Model expects a batch, so we add batch and sequence dimensions
        input_tensor = torch.tensor(feature_vector, dtype=torch.float32)
        input_tensor = input_tensor.unsqueeze(0).unsqueeze(0).to(self.device)

        self.classifier.eval()
        with torch.no_grad():
            logits = self.classifier(input_tensor)
            probabilities = torch.sigmoid(logits).squeeze().cpu().numpy()
        return probabilities

    def predict(self, poem_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        The main prediction pipeline. Orchestrates all steps.

        Args:
            poem_text (str): The raw Arabic poetry text.
            top_k (int): The number of top themes to return. If None, returns all.

        Returns:
            A list of dictionaries, each containing the theme name and its probability.
        """
        try:
            logger.info("Starting a new prediction...")
            
            # 1. Preprocess
            processed_verses = self._preprocess_text(poem_text)
            
            # 2. Embed
            poem_embedding = self._embed_and_segment(processed_verses)
            
            # 3. Classify
            probabilities = self._classify(poem_embedding)

            # 4. Format results
            results = []
            for i, prob in enumerate(probabilities):
                if prob > CLASSIFICATION_THRESHOLD:
                    results.append({
                        "theme": self.label_map.get(i, f"Unknown_Theme_{i}"),
                        "probability": float(prob)
                    })
            
            # Sort by probability in descending order
            sorted_results = sorted(results, key=lambda x: x['probability'], reverse=True)
            
            logger.info(f"Prediction successful. Found {len(sorted_results)} themes above threshold.")
            
            return sorted_results[:top_k] if top_k else sorted_results

        except (ValueError, TypeError) as e:
            logger.warning(f"Prediction failed due to invalid input: {e}")
            raise  # Re-raise to be caught by the UI
        except Exception as e:
            logger.error(f"An unexpected error occurred in the prediction pipeline: {e}", exc_info=True)
            raise RuntimeError("An internal error occurred during analysis.") from e