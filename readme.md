# Reinforecement learning SIM to SIM

## Getting Started

### Installation

We test our codes under the following environment:

- Ubuntu 22.04
- NVIDIA Driver: 550.163.01
- CUDA 12.4
- Python 3.10
- PyTorch 2.7.0 build for CUDA 12.8

1. Create an environment and install PyTorch:
   If you don't have any conda env or you wanna create a new env for mujco, then
   * `conda activate rl_mujoco`
   * `conda create -n rl_mujoco python=3.10`
   * `pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128`

    If there exists a conda env, then do install rl_mujoco_sim straightly

2. Install rl_mujoco_sim

   * `pip install -e .`
