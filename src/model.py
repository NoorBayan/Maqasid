# -*- coding: utf-8 -*-

import json
import logging
from typing import List, Dict, Any, Union

import torch
import torch.nn as nn

# Configure logging for clear and informative output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HybridPoetryClassifier(nn.Module):
    """
    A robust and flexible hybrid deep learning model combining CNNs and a Bi-LSTM network
    for multi-label thematic classification of Arabic poetry.

    This enhanced version includes rigorous input validation, flexible forward pass,
    and clear configuration management for production-ready use.

    Architecture:
    1. Input: Sequences of semantic segment embeddings (batch, seq_len, input_dim).
    2. 1D CNNs: Parallel convolutions with different kernel sizes to capture local features.
       Padding is applied to maintain sequence length for the smaller kernels.
    3. Activation & Pooling: ReLU activation followed by Max-over-time pooling.
    4. Bi-LSTM: Processes concatenated CNN features to model sequential context.
    5. Output: A fully connected layer with a Sigmoid activation (handled by loss function)
       for independent multi-label predictions.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        cnn_filters: int,
        cnn_kernel_sizes: List[int],
        lstm_hidden_dim: int,
        lstm_layers: int = 1,
        dropout_rate: float = 0.5,
    ):
        """
        Initializes the HybridPoetryClassifier model.

        Args:
            input_dim (int): The dimension of input embeddings.
            num_classes (int): The number of output classes (thematic labels).
            cnn_filters (int): The number of filters for each CNN kernel.
            cnn_kernel_sizes (List[int]): A list of kernel sizes for parallel CNNs.
            lstm_hidden_dim (int): The number of hidden units in the Bi-LSTM per direction.
            lstm_layers (int): The number of Bi-LSTM layers.
            dropout_rate (float): The dropout probability for regularization.
        """
        super(HybridPoetryClassifier, self).__init__()

        # --- Rigorous Input Validation ---
        self._validate_constructor_args(
            input_dim, num_classes, cnn_filters, cnn_kernel_sizes,
            lstm_hidden_dim, lstm_layers, dropout_rate
        )

        # Store configuration for saving and loading
        self.config = {
            'input_dim': input_dim,
            'num_classes': num_classes,
            'cnn_filters': cnn_filters,
            'cnn_kernel_sizes': cnn_kernel_sizes,
            'lstm_hidden_dim': lstm_hidden_dim,
            'lstm_layers': lstm_layers,
            'dropout_rate': dropout_rate,
        }

        # --- CNN Layers ---
        # We use a ModuleList to hold the parallel convolutional layers.
        # Padding is added to handle sequences shorter than the kernel size.
        # padding = (kernel_size - 1) // 2 for 'same' padding on one side.
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=input_dim,
                out_channels=cnn_filters,
                kernel_size=k,
                padding=(k - 1) // 2  # Add padding to preserve length for pooling
            ) for k in cnn_kernel_sizes
        ])

        # --- Bi-LSTM Layer ---
        lstm_input_dim = len(cnn_kernel_sizes) * cnn_filters
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout_rate if lstm_layers > 1 else 0.0
        )

        # --- Dropout and Output Layer ---
        self.dropout = nn.Dropout(dropout_rate)
        
        # The input to the final linear layer is the concatenated output of the Bi-LSTM
        self.fc = nn.Linear(lstm_hidden_dim * 2, num_classes)
        
        logger.info("HybridPoetryClassifier model initialized successfully.")
        self.log_model_summary()

    def _validate_constructor_args(self, *args):
        """A helper method to centralize constructor argument validation."""
        (input_dim, num_classes, cnn_filters, cnn_kernel_sizes,
         lstm_hidden_dim, lstm_layers, dropout_rate) = args
        
        if not all(isinstance(i, int) and i > 0 for i in [input_dim, num_classes, cnn_filters, lstm_hidden_dim, lstm_layers]):
            raise ValueError("Dimensions, filters, and layer counts must be positive integers.")
        if not (isinstance(cnn_kernel_sizes, list) and all(isinstance(k, int) and k > 0 for k in cnn_kernel_sizes)):
            raise ValueError("cnn_kernel_sizes must be a list of positive integers.")
        if not (0.0 <= dropout_rate < 1.0):
            raise ValueError(f"Dropout rate must be in [0.0, 1.0), but got {dropout_rate}.")


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, seq_len, input_dim).
                              `seq_len` corresponds to the number of thematic segments in a poem.

        Returns:
            torch.Tensor: The output tensor of raw logits of shape (batch_size, num_classes).
        """
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input tensor (batch, seq_len, dim), but got {x.dim()}D.")
        
        # --- CNN Path ---
        # Permute input for Conv1d: (batch, seq_len, input_dim) -> (batch, input_dim, seq_len)
        x_permuted = x.permute(0, 2, 1)

        # Apply convolutions and activation
        conv_outputs = [torch.relu(conv(x_permuted)) for conv in self.convs]
        
        # Max-over-time pooling: pool over the sequence dimension
        # The output of each pool is (batch, cnn_filters)
        pooled_outputs = [torch.max(conv, dim=2)[0] for conv in conv_outputs]

        # Concatenate features from all parallel CNNs
        # Shape: (batch, len(cnn_kernel_sizes) * cnn_filters)
        cnn_features = torch.cat(pooled_outputs, dim=1)

        # --- Bi-LSTM Path ---
        # Add a sequence dimension of 1 for the LSTM, treating CNN output as a single time step
        # Shape: (batch, 1, lstm_input_dim)
        lstm_input = self.dropout(cnn_features.unsqueeze(1))
        
        # LSTM forward pass
        # lstm_output shape: (batch, seq_len=1, num_directions * hidden_dim)
        lstm_output, (hidden, cell) = self.lstm(lstm_input)
        
        # Squeeze to remove the sequence dimension of 1
        # Shape: (batch, 2 * lstm_hidden_dim)
        lstm_final_output = lstm_output.squeeze(1)

        # --- Output Path ---
        final_features = self.dropout(lstm_final_output)
        logits = self.fc(final_features)
        
        return logits

    def save_model(self, model_path: str, config_path: str):
        """Saves the model's state_dict and its configuration to disk."""
        logger.info(f"Saving model to {model_path} and config to {config_path}")
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            
            torch.save(self.state_dict(), model_path)
            
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=4)
            logger.info("Model and configuration saved successfully.")
            
        except IOError as e:
            logger.error(f"IOError saving model: Could not write to path. Check permissions. Error: {e}")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred while saving the model: {e}")
            raise

    @classmethod
    def from_pretrained(cls, model_path: str, config_path: str, device: Union[str, torch.device] = 'cpu'):
        """Loads a model from a saved state_dict and a configuration file."""
        logger.info(f"Loading model from {model_path} and config from {config_path}")
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        except FileNotFoundError:
            logger.error(f"Configuration file not found at {config_path}")
            raise
        except json.JSONDecodeError:
            logger.error(f"Could not decode JSON from config file at {config_path}")
            raise

        # Instantiate the model with the loaded configuration
        model = cls(**config)
        
        # Determine device
        device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        
        try:
            # Load the state dictionary onto the specified device
            model.load_state_dict(torch.load(model_path, map_location=device))
            logger.info(f"Model state_dict loaded successfully from {model_path}")
        except FileNotFoundError:
            logger.error(f"Model state_dict file not found at {model_path}")
            raise
        except RuntimeError as e:
            logger.error(f"RuntimeError loading state_dict. This often means the model architecture in the code "
                         f"does not match the saved weights. Check your config. Error: {e}")
            raise

        model.to(device)
        model.eval()  # Set to evaluation mode by default after loading
        logger.info(f"Model loaded on device: '{device}' and set to evaluation mode.")
        return model

    def log_model_summary(self):
        """Logs a formatted summary of the model's architecture and parameter count."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        summary = f"""
        ================================================================
        Model: {self.__class__.__name__}
        ----------------------------------------------------------------
        Configuration:
        {json.dumps(self.config, indent=4)}
        ----------------------------------------------------------------
        Total Parameters:      {total_params:,}
        Trainable Parameters:  {trainable_params:,}
        ================================================================
        """
        logger.info(summary)

    def freeze_layers(self, layers_to_freeze: List[str]):
        """
        Freezes specified layers of the model to prevent them from being trained.
        Helpful for fine-tuning.

        Args:
            layers_to_freeze (List[str]): A list of layer names to freeze, e.g., ['convs', 'lstm'].
        """
        logger.warning(f"Freezing layers: {layers_to_freeze}")
        for name, param in self.named_parameters():
            # Check if the parameter's name starts with any of the specified layer names
            if any(name.startswith(layer_name) for layer_name in layers_to_freeze):
                param.requires_grad = False
        
        # Re-log the summary to show the new trainable parameter count
        self.log_model_summary()