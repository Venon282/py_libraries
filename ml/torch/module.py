import torch
import os
import random
import numpy as np

from ...other.loggingUtils import getLogger

def setDevice() -> torch.device:
    """
    Returns and set the best available PyTorch device.

    The function automatically detects available hardware accelerators
    and selects the most appropriate device.

    Returns:
        str: Device ("cuda", "xpu", "mps", or "cpu").
    """
    # Resolve device
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    else:
        return torch.device("cpu")

def setSeed(
    seed: int = 42,
    device: str | torch.device = None,
) -> int:
    """
    Seed all relevant RNGs for reproducibility (Python, NumPy, PyTorch CPU/CUDA/MPS).

    Args:
        seed: Seed value.
        device: Optional device ("cpu", "cuda", "mps", or torch.device).
                If a CUDA/MPS device is given (or available), its RNG is
                seeded too. Defaults to auto-detecting CUDA/MPS availability.

    Returns:
        The seed that was used (useful when seed=None).
    """
    if device is None:
        device = setDevice()
    elif isinstance(device, str):
        device = torch.device(device)

    # Core RNGs
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Device-specific RNGs
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    elif device.type == "mps":
        if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
            torch.mps.manual_seed(seed)
        else:
            getLogger().error("Devise type is mps but no mps torch install")
    elif device.type == "xpu":
        if hasattr(torch, "xpu") and hasattr(torch.xpu, "manual_seed_all"):
            torch.xpu.manual_seed_all(seed)
        else:
            getLogger().error("Devise type is xpu but no xpu torch install")

    return seed

def infer(model, datas, device="cpu"):
    # todo add batch
    model.eval()

    with torch.inference_mode():
        # Ensure on the same devise
        datas = datas.to(device)
        model.to(device)

        return model(datas)
