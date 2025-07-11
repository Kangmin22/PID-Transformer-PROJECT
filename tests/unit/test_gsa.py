# FILE: tests/unit/test_gsa.py
import torch
import pytest
from pid_transformer.utils.gsa import get_advanced_gsa_metrics, calculate_gradient_spectrum

def test_spectrum_calculation_shape():
    """Tests if the output spectrum has the correct shape."""
    dummy_gradient = torch.randn(32, 64)
    spectrum, freqs = calculate_gradient_spectrum(dummy_gradient)
    n_fft = dummy_gradient.numel()
    expected_len = n_fft // 2 + 1
    assert spectrum.shape == (expected_len,), "Spectrum shape is incorrect."
    assert freqs.shape == (expected_len,), "Frequencies shape is incorrect."


def test_advanced_metrics_with_known_signals():
    """Tests entropy, flatness, and kurtosis with predictable signals."""
    n_points = 512 # Increase points for more stable statistics

    # 1. Pure sine wave (tonal signal)
    t = torch.linspace(0, 1, n_points)
    signal_pure = torch.sin(2 * torch.pi * 10 * t)

    # 2. White noise (standard normal signal)
    signal_noise = torch.randn(n_points)

    # 3. Heavy-tailed distribution signal
    signal_heavy_tailed = torch.cat([torch.randn(n_points // 2), torch.randn(n_points // 2) * 10])

    # --- Metrics for pure signal ---
    metrics_pure = get_advanced_gsa_metrics(signal_pure)
    assert metrics_pure['spectral_entropy'] < 1.5, "Entropy of a pure tone should be very low"
    assert metrics_pure['spectral_flatness'] < 0.1, "Flatness of a pure tone should be near zero"

    # --- Metrics for noise signal ---
    metrics_noise = get_advanced_gsa_metrics(signal_noise)
    assert metrics_noise['spectral_entropy'] > 5.0, "Entropy of white noise should be high"
    assert metrics_noise['spectral_flatness'] > 0.4, "Flatness of white noise should be high"
    
    # --- Metrics for kurtosis ---
    metrics_heavy = get_advanced_gsa_metrics(signal_heavy_tailed)
    
    # BUG FIX: Instead of checking against a fixed value,
    # check that the heavy-tailed signal's kurtosis is significantly
    # larger than the normal signal's kurtosis (which should be near 0).
    assert metrics_heavy['gradient_kurtosis'] > (metrics_noise['gradient_kurtosis'] + 1.0), \
        "Kurtosis of a heavy-tailed signal should be significantly higher than normal noise."