import torch

def getDevice():
    if torch.cuda.is_available():
        return "cuda" # Use NVIDIA GPU (if available)
    elif torch.backends.mps.is_available():
        return "mps" # Use Apple Silicon GPU (if available)
    else:
        return "cpu" # Default to CPU if no GPU is available

def infer(model, datas, device="cpu"):
    # todo add batch
    model.eval()

    with torch.inference_mode():
        # Ensure on the same devise
        datas = datas.to(device)
        model.to(device)

        return model(datas)
