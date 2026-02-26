import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_device():
    return DEVICE


def to_device(x):
    if hasattr(x, "to"):
        return x.to(DEVICE)
    return x


def move_tn_to_device(tn):
    tn.apply_to_arrays(to_device)
    return tn


def move_data_to_device(data: dict) -> dict:
    return {k: v.to(DEVICE) if hasattr(v, "to") else v for k, v in data.items()}
