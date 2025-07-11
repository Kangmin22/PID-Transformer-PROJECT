# FILE: tests/unit/test_pid_controller.py
import torch
import pytest
from pid_transformer.modules.pid_controller import GeometricPIDController

@pytest.fixture
def pid_config():
    """Provides a standard configuration for the PID controller."""
    return {
        'kp': 0.1,
        'ki': 0.01,
        'kd': 0.05,
        'dim': 128
    }

@pytest.fixture
def pid_controller(pid_config):
    """Creates a PID controller instance for testing."""
    return GeometricPIDController(
        kp=pid_config['kp'],
        ki=pid_config['ki'],
        kd=pid_config['kd'],
        dim=pid_config['dim']
    )

def test_initialization(pid_controller, pid_config):
    """Tests if the controller is initialized with the correct parameters."""
    assert pid_controller.Kp == pid_config['kp']
    assert pid_controller.Ki == pid_config['ki']
    assert pid_controller.Kd == pid_config['kd']
    assert pid_controller.integral.shape == (1, pid_config['dim'])
    assert pid_controller.prev_error.shape == (1, pid_config['dim'])
    assert torch.all(pid_controller.integral == 0)
    assert torch.all(pid_controller.prev_error == 0)

def test_forward_pass_shape(pid_controller, pid_config):
    """Tests if the forward pass returns a tensor of the correct shape."""
    batch_size = 4
    # Create a dummy error tensor
    error_vector = torch.randn(batch_size, pid_config['dim'])
    
    # The controller now returns a tuple (control_signal, pid_terms)
    output, _ = pid_controller(error_vector)
    
    assert output.shape == error_vector.shape

def test_state_updates(pid_controller, pid_config):
    """Tests if the integral and prev_error states are updated correctly."""
    error_vector = torch.ones(1, pid_config['dim']) # Simple error of all ones
    
    # First pass
    pid_controller(error_vector)
    
    # The integral should now be the error_vector itself
    # We test the value before clipping for this specific test
    assert torch.allclose(pid_controller.integral, error_vector)
    # The prev_error should now be the error_vector
    assert torch.allclose(pid_controller.prev_error.detach(), error_vector) # Detach for comparison
    
    # Second pass with a different error
    new_error_vector = torch.ones(1, pid_config['dim']) * 0.5
    pid_controller(new_error_vector)
    
    # The integral should be the sum of the first and second errors (1.0 + 0.5)
    expected_integral = torch.ones(1, pid_config['dim']) * 1.5
    assert torch.allclose(pid_controller.integral, expected_integral)
    # The prev_error should now be the new_error_vector
    assert torch.allclose(pid_controller.prev_error.detach(), new_error_vector) # Detach for comparison
    
def test_reset_states(pid_controller):
    """Tests if the controller's states can be reset to zero."""
    error_vector = torch.randn(1, pid_controller.integral.shape[1])
    pid_controller(error_vector)

    # States should not be zero after one pass
    assert not torch.all(pid_controller.integral == 0)
    assert not torch.all(pid_controller.prev_error == 0)

    pid_controller.reset_states()

    # States should be zero after reset
    assert torch.all(pid_controller.integral == 0)
    assert torch.all(pid_controller.prev_error == 0)

def test_integral_windup_clipping(pid_config):
    """Tests if the integral term's norm is correctly clipped by the windup_limit."""
    windup_limit = 5.0
    controller = GeometricPIDController(
        kp=pid_config['kp'],
        ki=pid_config['ki'],
        kd=pid_config['kd'],
        dim=pid_config['dim'],
        windup_limit=windup_limit
    )

    # Create an error that would push the integral norm over the limit
    error_vector = torch.ones(1, pid_config['dim']) * (windup_limit + 1.0)
    
    # Forward pass
    controller(error_vector)
    
    # The integral norm should be clipped at the windup_limit
    integral_norm = torch.norm(controller.integral)
    # Use a small tolerance for floating point comparison
    assert torch.isclose(integral_norm, torch.tensor(windup_limit))

def test_derivative_low_pass_filter(pid_config):
    """Tests that the low-pass filter smooths the derivative term."""
    controller_no_filter = GeometricPIDController(**pid_config, d_filter_window_size=1)
    
    # Initialize controller with filtering
    controller_with_filter = GeometricPIDController(**pid_config, d_filter_window_size=3)
    
    # A noisy error signal (alternating between +1 and -1)
    error1 = torch.ones(1, pid_config['dim'])
    error2 = -torch.ones(1, pid_config['dim'])
    
    # --- No Filter ---
    controller_no_filter(error1)
    _, terms_no_filter = controller_no_filter(error2)
    
    # --- With Filter ---
    controller_with_filter(error1)
    _, terms_with_filter = controller_with_filter(error2)

    # The D-term norm should be smaller with the filter because the signal is smoother
    assert terms_with_filter['d_norm'] < terms_no_filter['d_norm']

def test_learnable_lambdas(pid_controller):
    """Tests if lambda parameters are learnable and initialized to 1.0."""
    assert hasattr(pid_controller, 'lambda_p')
    assert isinstance(pid_controller.lambda_p, torch.nn.Parameter)
    
    # Check initialization to 1.0
    assert torch.isclose(pid_controller.lambda_p.data, torch.tensor(1.0))
    assert torch.isclose(pid_controller.lambda_i.data, torch.tensor(1.0))
    assert torch.isclose(pid_controller.lambda_d.data, torch.tensor(1.0))

    # Check if they are included in parameters
    param_names = [name for name, _ in pid_controller.named_parameters()]
    assert 'lambda_p' in param_names
    assert 'lambda_i' in param_names
    assert 'lambda_d' in param_names