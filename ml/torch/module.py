import torch

def getDevice():
    """
    Returns the best available PyTorch device.

    The function automatically detects available hardware accelerators
    and selects the most appropriate device.

    Returns:
        str: Device name ("cuda", "xpu", "mps", or "cpu").
    """

    if torch.cuda.is_available():
        return "cuda"  # NVIDIA GPU

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"  # Intel GPU

    if torch.backends.mps.is_available():
        return "mps"  # Apple Silicon GPU

    return "cpu"  # CPU fallback

def infer(model, datas, device="cpu"):
    # todo add batch
    model.eval()

    with torch.inference_mode():
        # Ensure on the same devise
        datas = datas.to(device)
        model.to(device)

        return model(datas)
