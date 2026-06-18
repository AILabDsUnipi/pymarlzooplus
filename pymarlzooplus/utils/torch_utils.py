from torch import nn


def to_cuda(module, device=None):
    """Assume module in cpu"""
    if device is None:
        return module.cuda()
    elif device == 'cpu':
        return module
    else:
        return module.to(device)

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

def orthogonal_init_(m, gain=1):
    if isinstance(m, nn.Linear):
        init(m, nn.init.orthogonal_,
                    lambda x: nn.init.constant_(x, 0), gain=gain)