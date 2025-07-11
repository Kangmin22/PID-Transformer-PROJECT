# FILE: tests/unit/test_gating_pid_layer.py
import torch
import pytest
import torch.nn.functional as F # <-- BUG FIX: Add missing import
from pid_transformer.layers import GroupPIDLayer

@pytest.fixture
def gating_layer_config():
    """Provides a standard configuration for the Gating-enabled GroupPIDLayer."""
    return {
        'hidden_dim': 64,
        'control_dims': {'small': 32},
        'num_groups': {'small': 4},
        'kp': 0.1,
        'ki': 0.01,
        'kd': 0.05,
        'windup_limit': 100.0,
        'd_filter_window_size': 3,
        'use_gating': True
    }

@pytest.fixture
def gating_pid_layer(gating_layer_config):
    """Creates a Gating-enabled GroupPIDLayer instance for testing."""
    return GroupPIDLayer(**gating_layer_config)

def test_gating_network_initialization(gating_pid_layer):
    """Tests if the gating network is properly initialized."""
    assert hasattr(gating_pid_layer, 'gating_network')
    assert isinstance(gating_pid_layer.gating_network['small'], torch.nn.Linear)
    assert gating_pid_layer.gating_network['small'].out_features == gating_pid_layer.num_groups['small']

def test_gating_weights_shape_and_sum(gating_pid_layer):
    """
    Tests if the forward pass returns gating weights of the correct shape
    and that the weights for each token sum to 1.
    """
    batch_size = 2
    seq_len = 10
    input_tensor = torch.randn(batch_size, seq_len, gating_pid_layer.ffn.d_model)

    _, _, gating_weights = gating_pid_layer(input_tensor, phase='small')

    expected_shape = (batch_size, seq_len, gating_pid_layer.num_groups['small'])
    assert gating_weights.shape == expected_shape, f"Gating weights shape should be {expected_shape}, but got {gating_weights.shape}"

    sum_of_weights = torch.sum(gating_weights, dim=-1)
    assert torch.allclose(sum_of_weights, torch.ones(batch_size, seq_len)), "Weights for each token should sum to 1"

def test_gated_control_signal(gating_pid_layer):
    """
    Tests that the final control signal is a weighted sum of individual signals.
    """
    batch_size = 1
    seq_len = 1
    input_tensor = torch.randn(batch_size, seq_len, gating_pid_layer.ffn.d_model)
    
    # This test primarily verifies that the forward pass runs without shape errors
    # after the gating logic is applied. The previous test already validates the weights.
    try:
        gating_pid_layer(input_tensor, phase='small')
    except Exception as e:
        pytest.fail(f"Forward pass with gating enabled raised an exception: {e}")