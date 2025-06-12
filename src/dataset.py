# -*- coding: utf-8 -*-

import logging
from typing import Tuple, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Configure logging
logger = logging.getLogger(__name__)


class PoetryThematicDataset(Dataset):
    """
    A robust, custom PyTorch Dataset for the Arabic Poetry Thematic Classification task.

    This class is responsible for:
    1. Loading data from a CSV file.
    2. Validating the necessary columns.
    3. Converting data into PyTorch Tensors.
    4. Optionally caching the processed data in memory for faster access during training.
    """

    def __init__(
        self,
        csv_path: str,
        vector_column: str,
        label_columns: List[str],
        text_column: Optional[str] = None,
        use_caching: bool = True
    ):
        """
        Initializes the Dataset.

        Args:
            csv_path (str): Path to the CSV file containing the dataset (e.g., train.csv).
            vector_column (str): The name of the column containing the pre-computed segment embeddings.
                                 The data in this column should be list-like or array-like.
            label_columns (List[str]): A list of column names representing the multi-label targets.
            text_column (Optional[str]): The name of the column containing the original text.
                                         Useful for debugging and analysis, but not used by the model.
            use_caching (bool): If True, the entire dataset will be loaded and converted to tensors
                                in memory upon initialization for faster access. Recommended for
                                datasets that fit in RAM.
        """
        super().__init__()
        
        self.csv_path = csv_path
        self.vector_column = vector_column
        self.label_columns = label_columns
        self.text_column = text_column
        self.use_caching = use_caching
        
        self.data = self._load_and_validate_data()
        self.num_samples = len(self.data)
        
        self._cached_data: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
        if self.use_caching:
            logger.info("Caching enabled. Pre-loading and converting all data to tensors...")
            self._cache_data()
            logger.info(f"Successfully cached {self.num_samples} samples.")

        logger.info(f"Dataset initialized from {csv_path}. Found {self.num_samples} samples.")

    def _load_and_validate_data(self) -> pd.DataFrame:
        """Loads the CSV file and validates its structure."""
        try:
            df = pd.read_csv(self.csv_path)
        except FileNotFoundError:
            logger.error(f"Dataset file not found at: {self.csv_path}")
            raise
        except Exception as e:
            logger.error(f"Failed to read CSV file at {self.csv_path}. Error: {e}")
            raise

        # Validate required columns
        required_cols = [self.vector_column] + self.label_columns
        if self.text_column:
            required_cols.append(self.text_column)
            
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in {self.csv_path}: {missing_cols}")

        return df

    def _parse_row(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, Optional[str]]:
        """
        Parses a single row from the DataFrame and converts it to tensors.
        
        Returns:
            A tuple of (feature_tensor, label_tensor, optional_text).
        """
        row = self.data.iloc[index]
        
        # --- Process Features (Vectors) ---
        try:
            # Vectors might be stored as string representations of lists, e.g., '[1, 2, 3]'
            # We need to handle this safely.
            vector_data = row[self.vector_column]
            if isinstance(vector_data, str):
                # A common case when reading from CSV
                import json
                feature_list = json.loads(vector_data)
            else:
                # Assumes it's already a list or numpy array
                feature_list = vector_data

            features = torch.tensor(feature_list, dtype=torch.float32)

        except Exception as e:
            logger.error(f"Failed to parse vector data at index {index} from column '{self.vector_column}'. Error: {e}")
            # Return a dummy tensor to avoid crashing the loader, or raise an error
            # For now, we'll raise to highlight the data quality issue.
            raise TypeError(f"Invalid data format in vector column at index {index}") from e

        # --- Process Labels ---
        try:
            labels = row[self.label_columns].values.astype(np.float32)
            labels = torch.from_numpy(labels)
        except Exception as e:
            logger.error(f"Failed to parse label data at index {index}. Error: {e}")
            raise TypeError(f"Invalid data format in label columns at index {index}") from e

        # --- Process Optional Text ---
        text = row[self.text_column] if self.text_column else None
        
        return features, labels, text
    
    def _cache_data(self):
        """Pre-processes and stores all samples in a list for fast retrieval."""
        self._cached_data = []
        for i in range(self.num_samples):
            try:
                # We only cache features and labels, not the text, to save memory
                features, labels, _ = self._parse_row(i)
                self._cached_data.append((features, labels))
            except TypeError as e:
                logger.error(f"Skipping sample at index {i} due to parsing error: {e}")
                # In a real scenario, you might want to handle this more gracefully,
                # e.g., by removing bad samples from self.data
                continue

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return self.num_samples

    def __getitem__(self, index: int) -> Union[Tuple[torch.Tensor, torch.Tensor], Dict]:
        """
        Retrieves a single sample from the dataset.

        If caching is enabled, it fetches pre-processed tensors from memory.
        Otherwise, it parses the row from the DataFrame on the fly.

        Returns:
            If text_column is provided, returns a dictionary.
            Otherwise, returns a tuple of (features, labels).
            This dictionary format is more flexible and explicit.
        """
        if self.use_caching and self._cached_data:
            if index >= len(self._cached_data):
                 raise IndexError(f"Index {index} out of range for cached data of size {len(self._cached_data)}")
            features, labels = self._cached_data[index]
            
            # If text is needed, we still have to fetch it from the DataFrame
            if self.text_column:
                text = self.data.iloc[index][self.text_column]
                return {"features": features, "labels": labels, "text": text}
            else:
                return features, labels
        else:
            # Process on the fly
            features, labels, text = self._parse_row(index)
            if self.text_column:
                return {"features": features, "labels": labels, "text": text}
            else:
                return features, labels