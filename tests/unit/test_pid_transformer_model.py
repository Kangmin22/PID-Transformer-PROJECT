# FILE: tests/unit/test_pid_transformer_model.py
import torch
import pytest
from pid_transformer.model import PIDTransformer

@pytest.fixture
def model_config():
    """Provides a standard configuration for the PIDTransformer model."""
    # This now matches the structure of our main config file
    return {
        'vocab_size': 1000,
        'hidden_dim': 512,
        'control_dim': 128,
        'n_layers': 3,
        'kp': 0.1,
        'ki': 0.01,
        'kd': 0.05,
        'windup_limit': 100.0,
        'd_filter_window_size': 1,
        'use_group_pid': False,
    }

@pytest.fixture
def pid_transformer_model(model_config):
    """Creates a PIDTransformer model instance for testing."""
    # BUG FIX: Pass the entire config dictionary
    return PIDTransformer(config=model_config)

def test_model_initialization(pid_transformer_model, model_config):
    """Tests if the model and its submodules are initialized correctly."""
    assert pid_transformer_model.embedding.num_embeddings == model_config['vocab_size']
    assert pid_transformer_model.embedding.embedding_dim == model_config['hidden_dim']
    assert len(pid_transformer_model.pid_layers) == model_config['n_layers']
    assert pid_transformer_model.output_head.out_features == model_config['vocab_size']

def test_model_forward_pass_shape(pid_transformer_model, model_config):
    """Tests the end-to-end forward pass for correct output shape."""
    input_tensor = torch.randint(0, model_config['vocab_size'], (4, 20))
    # BUG FIX: Unpack the new 4-tuple output from the model
    output_logits, _, _, _ = pid_transformer_model(input_tensor)
    assert output_logits.shape == (4, 20, model_config['vocab_size'])

def test_model_parameters_are_learnable(pid_transformer_model):
    """Checks if the model's parameters require gradients."""
    for param in pid_transformer_model.parameters():
        assert param.requires_grad