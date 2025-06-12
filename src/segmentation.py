# -*- coding: utf-8 -*-

import logging
from typing import List, Dict, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import davies_bouldin_score
from sklearn.exceptions import NotFittedError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class PoemSegmenter:
    """
    A class for segmenting poems into thematically coherent units using DBSCAN clustering
    on verse-level semantic embeddings. It also provides tools for evaluating clustering quality.
    """

    def __init__(self, eps: float = 0.5, min_samples: int = 2):
        """
        Initializes the PoemSegmenter with DBSCAN hyperparameters.

        Args:
            eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
                         This is the most important DBSCAN parameter.
            min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.
        """
        if eps <= 0:
            raise ValueError("DBSCAN 'eps' parameter must be positive.")
        if min_samples < 1:
            raise ValueError("DBSCAN 'min_samples' parameter must be at least 1.")

        self.eps = eps
        self.min_samples = min_samples
        self.dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric='cosine')
        logging.info(f"PoemSegmenter initialized with DBSCAN(eps={self.eps}, min_samples={self.min_samples}, metric='cosine').")

    def _validate_embeddings(self, embeddings: np.ndarray) -> bool:
        """
        Validates the input embeddings to ensure they are in the correct format.
        """
        if not isinstance(embeddings, np.ndarray):
            logging.error("Embeddings must be a numpy array.")
            return False
        if embeddings.ndim != 2:
            logging.error(f"Embeddings must be a 2D array, but got shape {embeddings.shape}.")
            return False
        if embeddings.shape[0] < self.min_samples:
            logging.warning(f"Number of verses ({embeddings.shape[0]}) is less than min_samples ({self.min_samples}). "
                            "All points may be classified as noise.")
        return True

    def segment_poem(self, verse_embeddings: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Segments a single poem into topical units based on its verse embeddings.

        Args:
            verse_embeddings (np.ndarray): A 2D numpy array where each row is the embedding for a verse.

        Returns:
            Tuple[np.ndarray, Dict[str, Any]]:
                - A numpy array of cluster labels for each verse. Label -1 indicates noise (a transitional verse).
                - A dictionary containing clustering metrics (number of clusters, noise points, Davies-Bouldin Index).
        """
        if not self._validate_embeddings(verse_embeddings):
            raise ValueError("Invalid embeddings format.")

        try:
            # Fit DBSCAN to the embeddings
            self.dbscan.fit(verse_embeddings)
            labels = self.dbscan.labels_
        except Exception as e:
            logging.error(f"An error occurred during DBSCAN fitting: {e}")
            # Return a result indicating failure (e.g., all noise)
            return np.full(verse_embeddings.shape[0], -1), {'error': str(e)}

        # --- Calculate Metrics ---
        metrics = {}
        # Number of clusters in labels, ignoring noise if present.
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = np.sum(labels == -1)
        
        metrics['num_clusters'] = n_clusters
        metrics['num_noise_points'] = n_noise
        metrics['total_points'] = len(labels)

        # Davies-Bouldin score can only be calculated if there is more than 1 cluster.
        if n_clusters > 1:
            try:
                db_index = davies_bouldin_score(verse_embeddings, labels)
                metrics['davies_bouldin_index'] = db_index
            except ValueError as e:
                logging.warning(f"Could not calculate Davies-Bouldin Index: {e}")
                metrics['davies_bouldin_index'] = None
        else:
            metrics['davies_bouldin_index'] = None
            logging.info(f"Davies-Bouldin Index not calculated (requires at least 2 clusters, found {n_clusters}).")
            
        logging.debug(f"Segmentation complete. Found {n_clusters} clusters and {n_noise} noise points.")
        
        return labels, metrics

    def group_verses_by_segment(self, verses: List[str], labels: np.ndarray) -> List[Dict[str, Any]]:
        """
        Groups original verse texts into segments based on the clustering labels.

        Args:
            verses (List[str]): The original list of verse texts.
            labels (np.ndarray): The cluster labels from `segment_poem`.

        Returns:
            List[Dict[str, Any]]: A list of segments. Each segment is a dictionary
                                  containing the 'segment_label' and a list of 'verses'.
                                  Noise points are grouped into their own individual segments.
        """
        if len(verses) != len(labels):
            raise ValueError("The number of verses must match the number of labels.")

        segments = {}
        noise_count = 0
        
        for i, label in enumerate(labels):
            if label == -1:
                # Treat each noise point as a separate, transitional segment
                segment_id = f"noise_{noise_count}"
                noise_count += 1
            else:
                segment_id = f"segment_{label}"

            if segment_id not in segments:
                segments[segment_id] = {'segment_label': label, 'verses': []}
            
            segments[segment_id]['verses'].append(verses[i])

        # Convert the dictionary of segments to a list, maintaining order if possible (though DBSCAN is not ordered).
        # We sort by the original segment label to keep some consistency.
        # This is an approximation of order. For true order, one would need to post-process.
        sorted_segments = sorted(segments.values(), key=lambda x: str(x['segment_label']))

        return sorted_segments

    def process_poem_df(self, df: pd.DataFrame, poem_id_col: str, verse_text_col: str, vector_col: str) -> pd.DataFrame:
        """
        Processes a DataFrame containing multiple poems, segmenting each one.

        Args:
            df (pd.DataFrame): DataFrame with columns for poem ID, verse text, and verse vectors.
            poem_id_col (str): The column name for the poem identifier.
            verse_text_col (str): The column name for the verse text.
            vector_col (str): The column name for the verse embeddings.

        Returns:
            pd.DataFrame: A new DataFrame with segmentation results for each poem, including
                          labels, metrics, and grouped segments.
        """
        required_cols = [poem_id_col, verse_text_col, vector_col]
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain the following columns: {required_cols}")

        results = []
        
        # Group by poem ID to process each poem individually
        for poem_id, group in df.groupby(poem_id_col):
            logging.info(f"Segmenting poem ID: {poem_id}")
            
            verses = group[verse_text_col].tolist()
            # Stack the list of arrays into a single 2D array
            embeddings = np.vstack(group[vector_col].values)

            try:
                labels, metrics = self.segment_poem(embeddings)
                grouped_segments = self.group_verses_by_segment(verses, labels)

                results.append({
                    poem_id_col: poem_id,
                    'segmentation_labels': labels,
                    'segmentation_metrics': metrics,
                    'thematic_segments': grouped_segments
                })
            except Exception as e:
                logging.error(f"Failed to process poem ID {poem_id}: {e}")
                results.append({
                    poem_id_col: poem_id,
                    'segmentation_labels': None,
                    'segmentation_metrics': {'error': str(e)},
                    'thematic_segments': None
                })
        
        if not results:
            logging.warning("Processing completed with no results. Check if the input DataFrame was empty.")
            return pd.DataFrame()

        return pd.DataFrame(results)