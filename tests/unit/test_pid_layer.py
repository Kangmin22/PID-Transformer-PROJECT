# FILE: tests/unit/test_pid_layer.py
import torch
import pytest
from pid_transformer.layers import PIDLayer

@pytest.fixture
def layer_config():
    """Provides a standard configuration for the PIDLayer."""
    return {
        'hidden_dim': 512,
        'control_dim': 128,
        'kp': 0.1,
        'ki': 0.01,
        'kd': 0.05,
        # BUG FIX: Add missing required arguments
        'windup_limit': 100.0,
        'd_filter_window_size': 3,
    }

@pytest.fixture
def pid_layer(layer_config):
    """Creates a PIDLayer instance for testing."""
    return PIDLayer(**layer_config)

def test_pid_layer_initialization(pid_layer, layer_config):
    """Tests if the layer and its submodules are initialized correctly."""
    assert pid_layer.projection.in_features == layer_config['hidden_dim']
    assert pid_layer.projection.out_features == layer_config['control_dim']
    assert pid_layer.pid_controller.Kp == layer_config['kp']

def test_pid_layer_forward_pass_shape(pid_layer, layer_config):
    """Tests if the forward pass returns a tensor of the correct shape."""
    input_tensor = torch.randn(8, 10, layer_config['hidden_dim'])
    # BUG FIX: Unpack the new 3-tuple output
    output_tensor, _, _ = pid_layer(input_tensor)
    assert output_tensor.shape == input_tensor.shape

def test_pid_layer_parameters_are_learnable(pid_layer):
    """Tests that parameters of the main layer and projection are registered as learnable."""
    found_ffn_params = False
    found_projection_params = False

    for name, param in pid_layer.named_parameters():
        if 'ffn.' in name:
            found_ffn_params = True
        if 'projection.' in name:
            found_projection_params = True
        assert param.requires_grad

    assert found_ffn_params
    assert found_projection_params

def test_pid_controller_state_updates_in_layer(pid_layer):
    """Ensures the internal PID controller's state is updated after a forward pass."""
    input_tensor = torch.randn(1, 1, pid_layer.ffn.d_model)
    initial_integral = pid_layer.pid_controller.integral.clone()

    pid_layer(input_tensor)

    updated_integral = pid_layer.pid_controller.integral
    assert not torch.allclose(initial_integral, updated_integral)