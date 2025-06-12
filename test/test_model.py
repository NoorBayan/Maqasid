# -*- coding: utf-8 -*-

import json
import os
import sys

import pytest
import torch
import torch.nn as nn

# Add the source directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from poetry_classifier.model import HybridPoetryClassifier

# --- Fixtures for Pytest ---

@pytest.fixture(scope="module")
def default_model_config():
    """Provides a default, valid configuration for the model."""
    return {
        "input_dim": 100,
        "num_classes": 10,
        "cnn_filters": 32,
        "cnn_kernel_sizes": [2, 3, 4],
        "lstm_hidden_dim": 50,
        "lstm_layers": 1,
        "dropout_rate": 0.3,
    }

@pytest.fixture
def model(default_model_config):
    """Provides a default instance of the HybridPoetryClassifier."""
    return HybridPoetryClassifier(**default_model_config)

@pytest.fixture
def dummy_input_batch():
    """Provides a dummy input batch tensor for testing the forward pass."""
    batch_size = 8
    seq_len = 15
    input_dim = 100
    # Create a random tensor of the expected shape
    return torch.randn(batch_size, seq_len, input_dim)


# --- Test Cases ---

class TestHybridPoetryClassifier:

    def test_initialization_success(self, model, default_model_config):
        """Test that the model initializes correctly with valid parameters."""
        assert model is not None
        assert isinstance(model, nn.Module)
        # Check if the configuration is stored correctly
        for key, value in default_model_config.items():
            assert model.config[key] == value
        
        # Check if sub-modules are created
        assert isinstance(model.convs, nn.ModuleList)
        assert len(model.convs) == len(default_model_config["cnn_kernel_sizes"])
        assert isinstance(model.lstm, nn.LSTM)
        assert isinstance(model.fc, nn.Linear)

    @pytest.mark.parametrize("invalid_config, error_msg", [
        ({"dropout_rate": 1.1}, "Dropout rate must be in"),
        ({"dropout_rate": -0.1}, "Dropout rate must be in"),
        ({"input_dim": 0}, "Dimensions, filters, and layer counts must be positive integers"),
        ({"cnn_kernel_sizes": [2, 0, 3]}, "cnn_kernel_sizes must be a list of positive integers"),
        ({"cnn_kernel_sizes": "not a list"}, "cnn_kernel_sizes must be a list"),
    ])
    def test_initialization_failure(self, default_model_config, invalid_config, error_msg):
        """Test that the model raises ValueError with invalid configuration parameters."""
        config = default_model_config.copy()
        config.update(invalid_config)
        
        with pytest.raises(ValueError, match=error_msg):
            HybridPoetryClassifier(**config)

    def test_forward_pass_output_shape(self, model, dummy_input_batch, default_model_config):
        """Test the forward pass and ensure the output shape is correct."""
        model.eval() # Set to evaluation mode
        with torch.no_grad():
            output = model(dummy_input_batch)
        
        expected_batch_size = dummy_input_batch.shape[0]
        expected_num_classes = default_model_config["num_classes"]
        
        assert isinstance(output, torch.Tensor)
        assert output.shape == (expected_batch_size, expected_num_classes)

    def test_forward_pass_with_different_batch_sizes(self, model):
        """Test that the model handles different batch sizes correctly."""
        model.eval()
        with torch.no_grad():
            # Test with batch size 1
            input_bs1 = torch.randn(1, 15, 100)
            output_bs1 = model(input_bs1)
            assert output_bs1.shape == (1, 10)
            
            # Test with a larger batch size
            input_bs32 = torch.randn(32, 15, 100)
            output_bs32 = model(input_bs32)
            assert output_bs32.shape == (32, 10)

    def test_forward_pass_invalid_input_dim(self, model):
        """Test that the model raises an error for input with incorrect dimensions."""
        # 2D input instead of 3D
        invalid_input = torch.randn(8, 100)
        with pytest.raises(ValueError, match="Expected 3D input tensor"):
            model(invalid_input)

    def test_trainability_and_gradient_flow(self, model, dummy_input_batch, default_model_config):
        """
        Perform a single training step to ensure that gradients are computed and
        model parameters are updated.
        """
        model.train() # Set to training mode
        
        # Create a dummy loss function and optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.BCEWithLogitsLoss()
        
        # Get a dummy target
        dummy_target = torch.randint(0, 2, (dummy_input_batch.shape[0], default_model_config["num_classes"])).float()
        
        # Get initial parameter values to check for updates
        initial_param_sum = sum(p.sum() for p in model.parameters())
        
        # --- Single training step ---
        optimizer.zero_grad()
        output = model(dummy_input_batch)
        loss = criterion(output, dummy_target)
        loss.backward()
        
        # Check if gradients are computed for the final layer
        assert model.fc.weight.grad is not None
        assert model.fc.weight.grad.ne(0).any() # Check that gradient is not all zeros
        
        optimizer.step()
        
        # Check if parameters have been updated
        final_param_sum = sum(p.sum() for p in model.parameters())
        assert initial_param_sum != final_param_sum

    def test_save_and_load_model(self, model, dummy_input_batch, tmp_path):
        """
        Test the model's save and load functionality.
        - Save the model state and config.
        - Load it back into a new model instance.
        - Verify that the new model produces the same output as the original.
        """
        model.eval()
        
        # Define paths for saving
        model_path = tmp_path / "test_model.pth"
        config_path = tmp_path / "test_config.json"
        
        # Save the model
        model.save_model(str(model_path), str(config_path))
        
        assert os.path.exists(model_path)
        assert os.path.exists(config_path)
        
        # Load the model into a new instance
        loaded_model = HybridPoetryClassifier.from_pretrained(str(model_path), str(config_path))
        loaded_model.eval()
        
        assert isinstance(loaded_model, HybridPoetryClassifier)
        
        # --- Verification ---
        # 1. Check that the configurations are identical
        with open(config_path, 'r') as f:
            saved_config = json.load(f)
        assert model.config == saved_config
        assert loaded_model.config == saved_config
        
        # 2. Check that the state dictionaries are identical
        assert all(
            torch.equal(p1, p2) for p1, p2 in zip(model.state_dict().values(), loaded_model.state_dict().values())
        )
        
        # 3. Most importantly, check that they produce the same output for the same input
        with torch.no_grad():
            original_output = model(dummy_input_batch)
            loaded_output = loaded_model(dummy_input_batch)
            
        assert torch.allclose(original_output, loaded_output, atol=1e-6)

    def test_device_movement(self, model, dummy_input_batch):
        """Test that the model can be moved to CUDA if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available, skipping device movement test.")
        
        # Move model and data to GPU
        device = torch.device("cuda")
        model.to(device)
        gpu_input = dummy_input_batch.to(device)
        
        # Check if parameters are on the correct device
        assert next(model.parameters()).is_cuda
        
        # Perform a forward pass on GPU
        model.eval()
        with torch.no_grad():
            try:
                output = model(gpu_input)
                assert output.is_cuda
            except Exception as e:
                pytest.fail(f"Forward pass on GPU failed: {e}")