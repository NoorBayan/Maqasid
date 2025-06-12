# -*- coding: utf-8 -*-

import logging
import os
import time
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import precision_recall_f1_score, hamming_loss

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    A robust trainer class for PyTorch models, specifically designed for multi-label text classification.
    This class encapsulates the training and evaluation loops, metric calculation, early stopping,
    and model checkpointing.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        config: Dict[str, Any]
    ):
        """
        Initializes the ModelTrainer.

        Args:
            model (nn.Module): The PyTorch model to be trained.
            train_loader (DataLoader): DataLoader for the training set.
            val_loader (DataLoader): DataLoader for the validation set.
            optimizer (torch.optim.Optimizer): The optimizer for training (e.g., Adam).
            criterion (nn.Module): The loss function (e.g., nn.BCEWithLogitsLoss).
            device (torch.device): The device to run training on ('cpu' or 'cuda').
            config (Dict[str, Any]): A dictionary containing training configurations, including:
                - 'epochs' (int): Total number of training epochs.
                - 'checkpoint_dir' (str): Directory to save model checkpoints.
                - 'model_name' (str): Name for the saved model file.
                - 'early_stopping_patience' (int): Patience for early stopping.
                - 'classification_threshold' (float): Threshold for converting probabilities to binary predictions.
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.config = config

        # Extract config values with defaults
        self.epochs = self.config.get('epochs', 20)
        self.checkpoint_dir = self.config.get('checkpoint_dir', 'saved_models/classifier/')
        self.model_name = self.config.get('model_name', 'best_poetry_classifier.pth')
        self.patience = self.config.get('early_stopping_patience', 5)
        self.threshold = self.config.get('classification_threshold', 0.5)

        # Early stopping attributes
        self.best_val_f1 = -1.0
        self.epochs_no_improve = 0
        
        # Ensure checkpoint directory exists
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        logger.info("ModelTrainer initialized successfully.")

    def _train_one_epoch(self) -> float:
        """Runs a single training epoch."""
        self.model.train()
        total_loss = 0
        
        # Progress bar for visual feedback
        progress_bar = tqdm(self.train_loader, desc="Training", leave=False)

        for batch in progress_bar:
            # Unpack batch and move to device
            # Assuming batch is a tuple/list: (inputs, labels)
            try:
                inputs, labels = batch
                inputs = inputs.to(self.device)
                labels = labels.to(self.device).float()
            except ValueError as e:
                logger.error(f"Error unpacking batch. Ensure DataLoader yields (inputs, labels). Error: {e}")
                continue # Skip this batch

            # --- Forward Pass ---
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            # --- Loss Calculation ---
            loss = self.criterion(outputs, labels)
            
            # --- Backward Pass and Optimization ---
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0) # Gradient clipping
            self.optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / len(self.train_loader)

    def _evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """Runs evaluation on a given DataLoader."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            progress_bar = tqdm(data_loader, desc="Evaluating", leave=False)
            for batch in progress_bar:
                try:
                    inputs, labels = batch
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device).float()
                except ValueError as e:
                    logger.error(f"Error unpacking batch during evaluation. Error: {e}")
                    continue

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                # Convert logits to probabilities and then to binary predictions
                preds = torch.sigmoid(outputs) > self.threshold
                
                # Append to lists for overall metric calculation
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        # Concatenate all batch results
        try:
            all_preds = np.concatenate(all_preds, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)
        except ValueError:
            logger.error("Evaluation resulted in empty predictions or labels. Check your data loader.")
            return {'loss': 0, 'precision': 0, 'recall': 0, 'f1_score': 0, 'hamming_loss': 1}

        # Calculate metrics
        # Use macro average to treat all classes equally, important for imbalanced datasets
        precision, recall, f1, _ = precision_recall_f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
        # Hamming loss is useful for multi-label classification
        h_loss = hamming_loss(all_labels, all_preds)

        return {
            'loss': total_loss / len(data_loader),
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'hamming_loss': h_loss
        }

    def _early_stopping(self, val_f1: float) -> bool:
        """
        Checks for early stopping condition and saves the best model.

        Returns:
            bool: True if training should stop, False otherwise.
        """
        if val_f1 > self.best_val_f1:
            self.best_val_f1 = val_f1
            self.epochs_no_improve = 0
            # Save the best model
            best_model_path = os.path.join(self.checkpoint_dir, self.model_name)
            try:
                torch.save(self.model.state_dict(), best_model_path)
                logger.info(f"New best model saved to {best_model_path} with F1-score: {val_f1:.4f}")
            except Exception as e:
                logger.error(f"Failed to save best model: {e}")
            return False
        else:
            self.epochs_no_improve += 1
            if self.epochs_no_improve >= self.patience:
                logger.info(f"Early stopping triggered after {self.patience} epochs with no improvement.")
                return True
            return False

    def train(self) -> Dict[str, Any]:
        """
        Executes the full training and validation loop for the specified number of epochs.
        Implements early stopping.

        Returns:
            Dict[str, Any]: A dictionary containing training history and best model path.
        """
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_f1': [],
            'best_model_path': None
        }
        
        logger.info(f"Starting training for {self.epochs} epochs on device '{self.device}'.")
        start_time = time.time()

        try:
            for epoch in range(self.epochs):
                epoch_start_time = time.time()

                # --- Training ---
                train_loss = self._train_one_epoch()
                history['train_loss'].append(train_loss)

                # --- Validation ---
                val_metrics = self._evaluate(self.val_loader)
                val_loss = val_metrics['loss']
                val_f1 = val_metrics['f1_score']
                history['val_loss'].append(val_loss)
                history['val_f1'].append(val_f1)

                epoch_duration = time.time() - epoch_start_time
                
                logger.info(
                    f"Epoch {epoch+1}/{self.epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val F1: {val_f1:.4f} | "
                    f"Val Precision: {val_metrics['precision']:.4f} | "
                    f"Val Recall: {val_metrics['recall']:.4f} | "
                    f"Duration: {epoch_duration:.2f}s"
                )

                # --- Early Stopping Check ---
                if self._early_stopping(val_f1):
                    break
            
            total_duration = time.time() - start_time
            logger.info(f"Training finished in {total_duration:.2f} seconds.")
            
            history['best_model_path'] = os.path.join(self.checkpoint_dir, self.model_name)
            
            return history

        except KeyboardInterrupt:
            logger.warning("Training interrupted by user.")
            return history
        except Exception as e:
            logger.error(f"An unexpected error occurred during training: {e}", exc_info=True)
            raise # Re-raise the exception after logging for debugging