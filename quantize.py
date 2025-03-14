import torch

def quantize(tensor, scale, dtype=torch.int8):

    scaled_tensor = tensor * scale
    rounded_tensor = torch.round(scaled_tensor)

    q_min = torch.iinfo(dtype).min
    q_max = torch.iinfo(dtype).max

    q_tensor = rounded_tensor.clamp(q_min, q_max).to(dtype)

    return q_tensor
