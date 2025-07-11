# FILE: config/base_config.py
import torch

def get_config():
    """Returns a dictionary containing the configuration."""
    config = {
        # Experiment
        "experiment_name": "pid_transformer_baseline",

        # Data
        "dataset_name": "xsum",
        "subset": "3.0.0",
        "text_column": "document",
        "summary_column": "summary",
        "max_seq_len": 512,

        # Model
        "vocab_size": 32000,
        "hidden_dim": 512,
        "n_layers": 4,

        # Adaptive Dimension Config
        "use_adaptive_dim": False,
        "dimension_switch_step": 2500, # 5000 step training threshold
        "control_dim_large": 256,
        "control_dim_small": 128,
        "num_groups_large": 8,
        "num_groups_small": 4,

        # Standard PID Config
        "control_dim": 128,
        "use_group_pid": False,
        "num_groups": 4,
        "kp": 0.002852, # HPO Optimal
        "ki": 0.001240, # HPO Optimal
        "kd": 0.012805, # HPO Optimal
        "windup_limit": 100.0,
        "d_filter_window_size": 3,
        "ortho_weight": 0.01,

        # --- Gating Config ---
        "use_gating": False, # Gating mechanism enable switch

        # Training
        "batch_size": 8,
        "learning_rate": 1e-4,
        "num_epochs": 1,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    return config