# FILE: tests/unit/test_group_pid_layer.py
import torch
import pytest
from pid_transformer.layers import GroupPIDLayer

@pytest.fixture
def group_layer_config():
    """Provides a standard configuration for the GroupPIDLayer."""
    # BUG FIX: Update to new expected dictionary format for dimensions and groups
    return {
        'hidden_dim': 512,
        'control_dims': {'small': 128},
        'num_groups': {'small': 4},
        'kp': 0.1,
        'ki': 0.01,
        'kd': 0.05,
        'windup_limit': 100.0,
        'd_filter_window_size': 3
    }

@pytest.fixture
def group_pid_layer(group_layer_config):
    """Creates a GroupPIDLayer instance for testing."""
    return GroupPIDLayer(**group_layer_config)

def test_group_pid_layer_initialization(group_pid_layer, group_layer_config):
    """Tests if the layer initializes the correct number of controller groups."""
    assert len(group_pid_layer.pid_controllers['small']) == group_layer_config['num_groups']['small']
    group_dim = group_layer_config['control_dims']['small'] // group_layer_config['num_groups']['small']
    assert group_pid_layer.pid_controllers['small'][0].integral.shape[1] == group_dim

def test_group_pid_layer_forward_pass_shape(group_pid_layer, group_layer_config):
    """Tests if the forward pass returns a tensor of the correct shape."""
    input_tensor = torch.randn(2, 10, group_layer_config['hidden_dim'])
    # BUG FIX: Unpack the new 3-tuple output
    output, _, _ = group_pid_layer(input_tensor, phase='small')
    assert output.shape == input_tensor.shape

def test_independent_state_updates(group_pid_layer, group_layer_config):
    """
    Tests if each PID controller group updates its state independently.
    """
    input_tensor = torch.randn(1, 1, group_layer_config['hidden_dim'])

    # Get initial integral states
    initial_integrals = [ctrl.integral.clone() for ctrl in group_pid_layer.pid_controllers['small']]

    # Forward pass
    # BUG FIX: Call with phase and unpack the new 3-tuple output
    group_pid_layer(input_tensor, phase='small')

    # The integral of at least one group should change
    integrals_changed = any(
        not torch.allclose(initial_integrals[i], group_pid_layer.pid_controllers['small'][i].integral)
        for i in range(len(initial_integrals))
    )
    assert integrals_changed, "At least one group's integral should have been updated."