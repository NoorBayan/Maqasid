# -*- coding: utf-8 -*-

import json
import logging
import os
import pickle
import random
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset


# Configure a root logger for the utility functions
logger = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO, log_file: Optional[str] = None):
    """
    Sets up a standardized logging configuration for the entire project.

    Args:
        level (int): The logging level (e.g., logging.INFO, logging.DEBUG).
        log_file (Optional[str]): Path to a file to save logs. If None, logs to console only.
    """
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers to avoid duplicate logs
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format))
    root_logger.addHandler(console_handler)

    # File handler
    if log_file:
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
            file_handler.setFormatter(logging.Formatter(log_format))
            root_logger.addHandler(file_handler)
            logger.info(f"Logging is set up. Logs will also be saved to {log_file}")
        except Exception as e:
            logger.error(f"Failed to set up file logger at {log_file}. Error: {e}")
    
    logger.info(f"Logging level set to {logging.getLevelName(level)}.")


def set_seed(seed: int = 42):
    """
    Sets the random seed for reproducibility across all relevant libraries.

    Args:
        seed (int): The seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        # The following two lines are needed for full reproducibility with CUDA
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed} for reproducibility.")


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Loads a configuration file in YAML format.

    Args:
        config_path (str): The path to the YAML configuration file.

    Returns:
        Dict[str, Any]: A dictionary containing the configuration.
    
    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the file is not a valid YAML.
    """
    logger.info(f"Loading configuration from {config_path}")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        if not isinstance(config, dict):
            raise TypeError("Configuration file content is not a dictionary.")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found at: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file at {config_path}. Please check syntax. Error: {e}")
        raise
    except TypeError as e:
        logger.error(f"Error in config file structure: {e}")
        raise


def save_object(obj: Any, filepath: str):
    """
    Saves a Python object to a file using pickle.

    Args:
        obj (Any): The Python object to save.
        filepath (str): The path where the object will be saved.
    """
    logger.info(f"Saving object to {filepath}")
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f)
    except (IOError, pickle.PicklingError) as e:
        logger.error(f"Failed to save object to {filepath}. Error: {e}")
        raise


def load_object(filepath: str) -> Any:
    """

    Loads a Python object from a pickle file.

    Args:
        filepath (str): The path to the pickle file.

    Returns:
        Any: The loaded Python object.
    """
    logger.info(f"Loading object from {filepath}")
    try:
        with open(filepath, 'rb') as f:
            obj = pickle.load(f)
        return obj
    except FileNotFoundError:
        logger.error(f"Object file not found at: {filepath}")
        raise
    except (IOError, pickle.UnpicklingError) as e:
        logger.error(f"Failed to load object from {filepath}. Error: {e}")
        raise


# --- Data Handling Utilities ---

class PoetryDataset(Dataset):
    """
    A custom PyTorch Dataset for poetry classification.
    Assumes data is pre-processed and consists of feature tensors and label tensors.
    """
    def __init__(self, features: Union[np.ndarray, torch.Tensor], labels: Union[np.ndarray, torch.Tensor]):
        if len(features) != len(labels):
            raise ValueError("Features and labels must have the same length.")
        
        self.features = torch.tensor(features, dtype=torch.float32) if isinstance(features, np.ndarray) else features
        self.labels = torch.tensor(labels, dtype=torch.float32) if isinstance(labels, np.ndarray) else labels
        logger.info(f"Dataset created with {len(self.features)} samples.")

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


def pad_collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]], pad_value: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    A custom collate_fn for a DataLoader to handle padding of sequences with variable lengths.
    This is crucial if your poems (or segments) have different numbers of vectors.

    Args:
        batch (List[Tuple[torch.Tensor, torch.Tensor]]): A list of (features, label) tuples.
        pad_value (float): The value to use for padding.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Padded features and their corresponding labels.
    """
    # Separate features and labels
    features, labels = zip(*batch)
    
    # Pad the features (sequences)
    # torch.nn.utils.rnn.pad_sequence handles padding efficiently
    features_padded = torch.nn.utils.rnn.pad_sequence(features, batch_first=True, padding_value=pad_value)
    
    # Stack the labels
    labels = torch.stack(labels, 0)
    
    return features_padded, labels