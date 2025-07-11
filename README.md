PID-Transformer: A Control-Theoretic Approach for Stabilizing Large-Scale Neural Networks
This repository contains the official implementation for PID-Transformer, a novel architecture that embeds control-theoretic principles directly into a Transformer to achieve robust, stable, and interpretable training dynamics.

🚀 Interactive Web Report

For a comprehensive overview and interactive visualizations of our key findings, please visit our live project report:

[Live Demo](https://kangmin22.github.io/PID-Transformer-PROJECT/)

# Overview

Training large-scale language models is often hindered by gradient instability and oscillatory behavior. This work introduces a new paradigm for model stabilization by integrating a learnable, geometric PID (Proportional-Integral-Derivative) controller directly into the Transformer's internal architecture. Our final model, AdaptiveDim+Gating PID-Transformer, actively manipulates the hidden state dynamics to ensure a more stable and efficient learning process.

# Key Features

- Group-wise PID Control: Applies specialized PID control to independent subgroups of the hidden state dimensions for more granular feedback.

- Adaptive Dimension: Dynamically adjusts the dimensionality of the control space, using a larger dimension for initial stabilization and a smaller one for fine-tuning.

- Gating Mechanism: Employs a learnable, softmax-based gate to dynamically allocate the influence of different control groups based on the current context.

- End-to-End Analysis: Validated with comprehensive experiments, including Gradient Spectrum Analysis (GSA) and PCA-based trajectory visualization.

# Installation

To set up the project locally for development and experimentation, please follow these steps.

Clone the Repository:

git clone https://github.com/Kangmin22/PID-Transformer-PROJECT.git
cd PID-Transformer-PROJECT
Set Up Virtual Environment:


# Create a new virtual environment

python -m venv venv

# Activate the environment
# On Windows

.\venv\Scripts\activate

# On macOS/Linux

source venv/bin/activate

Install Dependencies:

Install all required libraries from the requirements.txt file.

pip install -r requirements.txt

Install the Package in Editable Mode:

This command registers the pidtransformer package with your environment, allowing you to import it while making your local changes immediately available.

pip install -e .

# Usage

All experiments are run through the central main.py script. You can customize a run by providing command-line arguments.

Running a Training Experiment
To run an experiment, you must provide an --experiment_name. Other flags activate specific features of the model.

Example: Running the final "Hero Model" (AdaptiveDim+Gating)

python main.py --experiment_name "MyHeroModel_Run" \
               --num_steps 1000 \
               --log_freq 20 \
               --use_group_pid \
               --use_adaptive_dim \
               --use_gating \
               --ortho_weight 0.01 \
               --dimension_switch_step 500

Experiment artifacts, including history logs (.json) and model checkpoints (.pth), will be saved to the results/ and checkpoints/ directories, respectively.

# Visualizing Results

Use the provided plotting scripts to analyze the output from your experiments.

- Generate the GSA Dashboard:

python experiments/plot_advanced_gsa.py results/GSA_Baseline_NoControl_history.json results/GSA_Hero_Model_history.json

- Generate a 2D PCA Trajectory Plot:

python experiments/plot_trajectory.py results/GSA_Baseline_NoControl_history.json results/GSA_Hero_Model_history.json --

output_file trajectory_2d.png

- Generate a 3D PCA Trajectory Plot:

python experiments/plot_trajectory.py results/GSA_Baseline_NoControl_history.json results/GSA_Hero_Model_history.json --

output_file trajectory_3d.png --three_d

# Citation

If you use this work in your research, please cite:

@misc{pid-transformer2025,
  author       = {KANG JA IL},
  title        = {PID-Transformer: A Control-Theoretic Approach for Stabilizing Large-Scale Neural Networks},
  year         = {2025},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/Kangmin22/PID-Transformer-PROJECT}}
}