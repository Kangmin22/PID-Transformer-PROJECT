# PID-Transformer 🧭🧠

# PID-Transformer: A Control-Theoretic Approach for Stabilizing Large-Scale Neural Networks

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Control-Theoretic Stabilization for Large-Scale Transformers

> A novel Transformer architecture with embedded PID control logic for robust and interpretable representation dynamics.  
> Includes Group-wise PID, AdaptiveDim switching, and Gating-based trajectory feedback.

---

## 🔧 Installation

You can install the package in the following ways:

### ✅ PyPI

pip install pidtransformer

📦 From wheel or tar.gz

pip install pidtransformer-1.0.0-py3-none-any.whl
# or
pip install pidtransformer-1.0.0.tar.gz

---

🚀 Quick Start

1. Clone and prepare the environment

- git clone https://github.com/your-org/pid-transformer.git

- cd pid-transformer

- python3 -m venv venv

- source venv/bin/activate

- pip install -r requirements.txt

---

2. Run a training experiment

python experiments/train_baseline.py \
    --experiment_name "PID_Optimal_Filtered" \
    --kp 0.002852 --ki 0.001240 --kd 0.012805 \
    --d_filter 3

---

3. Visualize hidden state trajectories

python experiments/plot_trajectory.py \
    PID_Off_history.json PID_Optimal_Filtered_history.json \
    --output_file trajectory_comparison.png

---

🧠 Core Components

| Module                   | Description                                                                         |
| ------------------------ | ----------------------------------------------------------------------------------- |
| `GeometricPIDController` | A vector-based controller applying P/I/D logic to model-internal error signals.     |
| `PIDLayer`               | Combines FFN + projection to control space + PID feedback loop.                     |
| `AdaptiveDim`            | Dynamically switches dimensionality (e.g. 256→128) to optimize stability over time. |
| `Group-wise PID`         | PID applied across subgroups of hidden dimensions.                                  |
| `Trajectory Tracker`     | Captures hidden state evolution to visualize smoothness/curvature via PCA.          |

---

📊 Interactive Visualization

🖥 Launch in Browser

[Go to Live Demo](https://kangmin22.github.io/PID-Transformer-PROJECT/)

You’ll see:

PCA latent space trajectories 🧭

Loss/PID Norm curves 📉

GSA bar chart 📊

Diagrammatic explanation of PIDLayer and controller design 🧩

---

🧪 Reproducible Experiments

# Baseline Model (no control)
python experiments/train_baseline.py --experiment_name "PID_Off" --kp 0 --ki 0 --kd 0

# Full Control Model
python experiments/train_baseline.py --experiment_name "AdaptiveDim" \
    --kp 0.1 --ki 0.01 --kd 0.05 --use_adaptive_dim --use_group_pid --ortho_weight 0.01

# Endurance-scale
python experiments/train_baseline.py --experiment_name "Endurance_AdaptiveDim" \
    --num_steps 5000 --log_freq 50 --use_adaptive_dim --use_group_pid

---

📚 Citation
If you use this work in your research, please cite:

@article{pidtransformer2025,
  title={AdaptiveDim+Gating PID-Transformer: A Control-Theoretic Approach for Stabilizing Large-Scale Neural Networks},
  author={KANG JA IL},
  year={2025}
}

