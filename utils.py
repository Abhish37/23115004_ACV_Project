"""
utils.py - Reproducibility, System Environment Logging, and Seed Control
Author: Roll No. 23115004
Course: Advanced Computer Vision (ACV)
Assignment: B - Reproducible Segmentation Experiment
"""

import os
import random
import platform
import numpy as np
import torch

def set_seed(seed: int = 42) -> None:
    """
    Enforces deterministic execution across Python, NumPy, PyTorch CPU, and CUDA.
    
    Args:
        seed (int): The integer random seed to lock all RNG states.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"[REPRODUCIBILITY] System seed locked to: {seed}")

def log_system_environment() -> dict:
    """
    Captures system architecture, hardware specifications, and package versions
    to satisfy the reproducibility contract.
    """
    env_info = {
        "Operating System": f"{platform.system()} {platform.release()} ({platform.version()})",
        "Machine Architecture": platform.machine(),
        "Processor": platform.processor(),
        "Python Version": platform.python_version(),
        "PyTorch Version": torch.__version__,
        "CUDA Available": torch.cuda.is_available(),
        "CUDA Device Name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A (CPU execution)",
        "Device Count": torch.cuda.device_count() if torch.cuda.is_available() else 0
    }
    
    print("=" * 60)
    print("           EXPERIMENTAL ENVIRONMENT SPECIFICATION           ")
    print("=" * 60)
    for k, v in env_info.items():
        print(f"  {k:<22}: {v}")
    print("=" * 60)
    return env_info

if __name__ == "__main__":
    set_seed(42)
    log_system_environment()
