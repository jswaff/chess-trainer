import torch

def quantize(tensor, scale, dtype=torch.int8):

    scaled_tensor = tensor * scale
    rounded_tensor = torch.round(scaled_tensor)

    q_min = torch.iinfo(dtype).min
    q_max = torch.iinfo(dtype).max

    q_tensor = rounded_tensor.clamp(q_min, q_max).to(dtype)

    return q_tensor

def quantization_error(model):
    n_params = 0
    sum_errors = None

    for parameter in model.parameters():
        if parameter.dim() == 1:
            n_params += parameter.shape[0]
        else:
            n_params += parameter.shape[0] * parameter.shape[1]

        q_parameter = quantize(parameter, 64)
        delta = torch.abs(parameter * 64 - q_parameter)
        param_error = delta.sum()
        if sum_errors is None:
            sum_errors = param_error
        else:
            sum_errors += param_error

    return sum_errors / n_params
