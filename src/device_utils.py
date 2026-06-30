import torch


def get_device(device_arg="auto"):
    """Resolve a user-facing device option to a torch device."""
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested with --device cuda, but CUDA is not available. "
                "Use --device cpu or --device auto."
            )
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    raise ValueError("device_arg must be one of: auto, cuda, cpu")
