import math

import torch


def KL_div(p_mu, p_sigma, q_mu, q_sigma):
    """_summary_
    Args:
        p_mu (bs, dist_dim): _description_
        p_sigma (bs, dist_dim): _description_
        q_mu (bs, dist_dim): _description_
        q_sigma (bs, dist_dim): _description_
    """

    div = (
        torch.log2(q_sigma)
        - torch.log2(p_sigma)
        + (p_sigma**2 + (p_mu - q_mu) ** 2) / (2 * q_sigma**2)
        - 0.5
    )
    div = div.mean(dim=-1)
    return div


def get_gard_norm(it):
    sum_grad = 0
    for x in it:
        if x.grad is None:
            continue
        sum_grad += x.grad.norm() ** 2
    return math.sqrt(sum_grad)
