# Shared device-selection helper, mirrors cp_utils.py.
## need to put all imports depengin on numpy here
import numpy as np
import torch
from ultralytics import YOLO

def get_torch_device(use_gpu: bool) -> tuple[bool, torch.device]:
    """Return (use_gpu, device) using best available backend: CUDA > MPS > CPU."""
    if not use_gpu:
        return False, torch.device("cpu")
    if torch.cuda.is_available():
        return True, torch.device("cuda")
    if torch.backends.mps.is_available():
        return True, torch.device("mps")
    return False, torch.device("cpu")
