# -*- coding: utf-8 -*-

import logging
import os
import sys

import optuna
import torch
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader

# Add src to path to allow direct imports
# This is necessary if you run the script directly
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from poetry_classifier.model import HybridPoetryClassifier
from poetry_classifier.trainer import ModelTrainer
from poetry_classifier.dataset import PoetryThematicDataset
from poetry_classifier.utils import set_seed

# Configure logging
logger = logging.getLogger(__name__)


def objective(trial: optuna.trial.Trial, hpo_config: dict) -> float:
    """
    The objective function for Optuna to optimize.
    This function defines a single trial of HPO:
    1. Suggests a set of hyperparameters.
    2. Builds a model with these hyperparameters.
    3. Trains and evaluates the model.
    4. Reports the final performance metric to Optuna.

    Args:
        trial (optuna.trial.Trial): An Optuna trial object.
        hpo_config (dict): A dictionary containing fixed configurations for HPO
                           (e.g., data paths, input/output dimensions).

    Returns:
        float: The performance metric to be optimized (e.g., validation F1-score).
    """
    set_seed(42) # Ensure reproducibility for each trial

    try:
        # --- 1. Suggest Hyperparameters ---
        logger.info(f"--- Starting Trial {trial.number} ---")
        
        # Optimizer selection
        optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW"])
        learning_rate = trial.suggest_float("lr", 1e-5, 1e-2, log=True)

        # Model architecture
        cnn_filters = trial.suggest_categorical("cnn_filters", [64, 128, 256])
        # Example of suggesting combinations of kernel sizes
        kernel_sizes_str = trial.suggest_categorical("cnn_kernel_sizes", ["[2,3,4]", "[3,4,5]", "[2,4,6]"])
        cnn_kernel_sizes = eval(kernel_sizes_str)

        lstm_hidden_dim = trial.suggest_categorical("lstm_hidden_dim", [64, 128, 256])
        lstm_layers = trial.suggest_int("lstm_layers", 1, 2)
        dropout_rate = trial.suggest_float("dropout", 0.1, 0.5)

        # Training related
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

        # --- 2. Load Data ---
        # Assuming data paths are fixed and provided in hpo_config
        train_dataset = PoetryThematicDataset(
            csv_path=hpo_config['train_data_path'],
            vector_column=hpo_config['vector_column'],
            label_columns=hpo_config['label_columns']
        )
        val_dataset = PoetryThematicDataset(
            csv_path=hpo_config['val_data_path'],
            vector_column=hpo_config['vector_column'],
            label_columns=hpo_config['label_columns']
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # --- 3. Build Model and Optimizer ---
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = HybridPoetryClassifier(
            input_dim=hpo_config['input_dim'],
            num_classes=len(hpo_config['label_columns']),
            cnn_filters=cnn_filters,
            cnn_kernel_sizes=cnn_kernel_sizes,
            lstm_hidden_dim=lstm_hidden_dim,
            lstm_layers=lstm_layers,
            dropout_rate=dropout_rate
        ).to(device)
        
        optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=learning_rate)
        criterion = torch.nn.BCEWithLogitsLoss()
        
        # --- 4. Train and Evaluate ---
        trainer_config = {
            'epochs': hpo_config.get('epochs_per_trial', 15), # Limit epochs for HPO
            'checkpoint_dir': os.path.join(hpo_config['hpo_output_dir'], f'trial_{trial.number}'),
            'model_name': 'best_model.pth',
            'early_stopping_patience': hpo_config.get('patience', 3),
            'classification_threshold': 0.5
        }
        
        trainer = ModelTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            config=trainer_config
        )

        # --- Integrate with Optuna's Pruning ---
        # We need to adapt the trainer to report intermediate values.
        # For simplicity, here we'll just run the full training and get the best F1.
        # A more advanced integration would involve callbacks.
        history = trainer.train()

        # Check if training was successful
        if not history or not history['val_f1']:
            logger.warning(f"Trial {trial.number} did not produce any validation results. Returning a low score.")
            return 0.0 # Return a poor score to penalize this trial

        best_val_f1 = max(history['val_f1'])
        logger.info(f"--- Trial {trial.number} Finished --- Best Val F1: {best_val_f1:.4f}")
        
        return best_val_f1

    except Exception as e:
        logger.error(f"Trial {trial.number} failed with an exception: {e}", exc_info=True)
        # Return a poor score to indicate failure, but allow the study to continue.
        return 0.0


def run_hpo(hpo_config: dict):
    """
    Main function to orchestrate the hyperparameter optimization process.

    Args:
        hpo_config (dict): Main configuration for the HPO study.
    """
    n_trials = hpo_config.get('n_trials', 50)
    output_dir = hpo_config.get('hpo_output_dir', 'hpo_results')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # --- Create an Optuna Study ---
    # We want to maximize the F1-score, so the direction is 'maximize'.
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5), # Prune unpromising trials early
        study_name="arabic_poetry_classification_hpo"
    )

    logger.info(f"Starting HPO study with {n_trials} trials...")
    
    # The lambda function is used to pass the fixed hpo_config to the objective function.
    study.optimize(lambda trial: objective(trial, hpo_config), n_trials=n_trials, timeout=hpo_config.get('timeout_seconds', None))

    # --- Print and Save Results ---
    logger.info("HPO study finished.")
    
    # Best trial
    best_trial = study.best_trial
    logger.info(f"Best trial number: {best_trial.number}")
    logger.info(f"Best value (max F1-score): {best_trial.value:.4f}")
    logger.info("Best hyperparameters:")
    for key, value in best_trial.params.items():
        logger.info(f"  {key}: {value}")
        
    # Save results to a CSV file
    results_df = study.trials_dataframe()
    results_csv_path = os.path.join(output_dir, "hpo_results.csv")
    results_df.to_csv(results_csv_path, index=False)
    logger.info(f"Full HPO results saved to {results_csv_path}")

    # --- Visualization ---
    # This requires installing plotly: pip install plotly
    try:
        fig_hist = optuna.visualization.plot_optimization_history(study)
        fig_hist.write_image(os.path.join(output_dir, "optimization_history.png"))

        fig_slice = optuna.visualization.plot_slice(study)
        fig_slice.write_image(os.path.join(output_dir, "slice_plot.png"))
        
        fig_importance = optuna.visualization.plot_param_importances(study)
        fig_importance.write_image(os.path.join(output_dir, "param_importances.png"))
        
        logger.info(f"Visualization plots saved to {output_dir}")
    except (ImportError, Exception) as e:
        logger.warning(f"Could not generate visualization plots. Is plotly installed? Error: {e}")

if __name__ == '__main__':
    # This is an example of how to run the HPO script.
    # In a real project, this config would be loaded from a YAML file.
    
    # Dummy data for demonstration. Replace with your actual data paths.
    # Creating dummy files for the script to run without errors.
    dummy_data = {
        'vector': [[0.1]*10 for _ in range(100)],
        'label_1': [0, 1]*50,
        'label_2': [1, 0]*50
    }
    import pandas as pd
    os.makedirs('data/annotated', exist_ok=True)
    pd.DataFrame(dummy_data).to_csv('data/annotated/dummy_train.csv', index=False)
    pd.DataFrame(dummy_data).to_csv('data/annotated/dummy_val.csv', index=False)

    hpo_main_config = {
        'train_data_path': 'data/annotated/dummy_train.csv',
        'val_data_path': 'data/annotated/dummy_val.csv',
        'vector_column': 'vector',
        'label_columns': ['label_1', 'label_2'],
        'input_dim': 10, # Must match the dummy data vector size
        'n_trials': 10, # Use a small number for quick testing
        'epochs_per_trial': 3,
        'patience': 2,
        'hpo_output_dir': 'hpo_run_example'
    }
    
    run_hpo(hpo_main_config)