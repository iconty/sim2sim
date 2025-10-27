# Reinforcement Learning SIM to SIM

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

   **Recommend:** If there exists a conda env, then install legged_mujoco straightly, make sure your python version >= 3.8, and installed pytorch
2. Install legged_mujoco

   * `pip install -e .`
3. Run sim

   For action actiger2

   * `python scripts/sim2sim_ac2.py`

   For unitree go2:

   * `python scripts/sim2sim_go2.py`
4. Change Policy
   pretrian_policy is the folder of pre-train_policy, you can change it to your own policy.
5. Check TODO, you will find all you need to do.
   Download Todo Tree in vscode Extensions firstly
