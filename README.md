# NetEvolve

## Required

Our checked platforms are:

- Ubuntu22.04 LTS (using CUDA/x86_64)
- Windows 11 Pro (using Windows Subsystem for Linux,CUDA/x86_64)
- Apple Macbook Pro(M1 processor, using CPU/GPU)
  - if you run this code using GPU acceleration, you change to "mps" in the `select_device` variable in `config.py`


[Required]

- Python 3.11.0 (or later)
- Poetry 1.5.1 (or later)

[Optional]

- CUDA 12.X (using NVIDIA GPU)
- NCCL2 required

## how to run

### initialize environment and create poetry virtual env

```
poetry install
```

### Test the code

First, you select the dataset in the `init_real_data` and delete coment out (others 
should comment out).

Second, you execute the two programs.

```
poetry run python optimize_reward.py
poetry run python rl.py
```

```
Network Installer for Ubuntu22.04

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update
sudo apt install libnccl2 libnccl-dev
```