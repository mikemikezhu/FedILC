import copy
import torch
from torch import nn, autograd

from backpack import backpack, extend
from backpack.extensions import SumGradSquared, Variance

"""
Fishr
"""


def compute_irm_penalty(logits, y, loss_fn):
    scale = torch.tensor(1.).requires_grad_()
    if torch.cuda.is_available():
        scale = torch.tensor(1.).cuda().requires_grad_()
    loss = loss_fn(logits * scale, y)
    grad = autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad**2)


def compute_grad_variance(input, labels, network, loss_fn):
    """
    Main Fishr method that computes the gradient variances using the BackPACK package.
    """
    logits = network(input)
    bce_extended = extend(loss_fn)
    loss = bce_extended(logits, labels)

    # print('Prediction: {}'.format(logits))
    # print('Real: {}'.format(labels))
    # calling first-order derivatives in the network while maintaining the per-sample gradients

    with backpack(Variance()):
        loss.backward(
            inputs=list(network.parameters()), retain_graph=True, create_graph=True
        )

    dict_grads_variance = {
        name: (
            weights.variance.clone().view(-1)
        ) for name, weights in network.named_parameters()
    }

    return dict_grads_variance


def l2_between_dicts(dict_1, dict_2):
    assert len(dict_1) == len(dict_2)
    dict_1_values = [dict_1[key] for key in sorted(dict_1.keys())]
    dict_2_values = [dict_2[key] for key in sorted(dict_1.keys())]
    return (
        torch.cat(tuple([t.view(-1) for t in dict_1_values])) -
        torch.cat(tuple([t.view(-1) for t in dict_2_values]))
    ).pow(2).sum()


"""
ILC
"""


def get_model_grads(input, labels, network, loss_fn):

    _, logits = network(input)

    loss = loss_fn(logits, labels)
    loss.backward()

    model_params = list(network.parameters())
    param_gradients = []
    for model_param in model_params:
        # Get gradients
        # Note: The gradient of the loss each parameter p is stored in p.grad after the backward
        # See: https://discuss.pytorch.org/t/how-to-get-gradient-of-loss/16955
        grad = model_param.grad
        grad_copy = copy.deepcopy(grad)
        param_gradients.append(grad_copy)

    return param_gradients


"""
Arithmetic mean
"""


def compute_arith_mean(model_params, total_param_gradients):

    param_gradients = [[] for _ in model_params]

    # Loop for each environment
    for env_param_gradients in total_param_gradients:
        for idx, grads in enumerate(param_gradients):
            env_grad = env_param_gradients[idx]
            grads.append(env_grad)

    assert len(param_gradients) == len(model_params)

    for param, grads in zip(model_params, param_gradients):

        # Calculate sign matrix
        grads = torch.stack(grads, dim=0)
        avg_grad = torch.mean(grads, dim=0)
        param.grad = avg_grad


"""
Geometric mean
"""


def compute_geo_mean(model_params, total_param_gradients, algorithm, substitute, flags):

    if "geo_substitute" == algorithm:
        compute_substitute_geo_mean(
            model_params, total_param_gradients, substitute)
    elif "geo_weighted" == algorithm:
        compute_weighted_geo_mean(model_params, total_param_gradients, flags)


def compute_substitute_geo_mean(model_params, total_param_gradients, substitute):

    param_gradients = [[] for _ in model_params]

    # Loop for each environment
    for env_param_gradients in total_param_gradients:
        for idx, grads in enumerate(param_gradients):
            env_grad = env_param_gradients[idx]
            grads.append(env_grad)

    assert len(param_gradients) == len(model_params)

    for param, grads in zip(model_params, param_gradients):

        # Calculate sign matrix
        grads = torch.stack(grads, dim=0)
        sign_matrix = torch.sign(grads)

        avg_sign_matrix = torch.mean(sign_matrix, dim=0)

        # If torch.sign(avg_sign_matrix) == 0, then has equal number of positive and negative numbers
        # Regard the positive numbers are majority signs
        avg_sign = torch.sign(avg_sign_matrix) + (avg_sign_matrix == 0)

        majority_sign_matrix = sign_matrix == avg_sign
        minority_sign_matrix = ~majority_sign_matrix

        grads = majority_sign_matrix * grads + minority_sign_matrix * substitute

        n_agreement_envs = len(grads)
        avg_grad = torch.mean(grads, dim=0)
        substitute_prod_grad = torch.sign(avg_grad) * torch.exp(
            torch.sum(torch.log(torch.abs(grads) + 1e-10), dim=0) / n_agreement_envs)

        param.grad = substitute_prod_grad


def compute_weighted_geo_mean(model_params, total_param_gradients, flags):

    param_gradients = [[] for _ in model_params]

    # Loop for each environment
    for env_param_gradients in total_param_gradients:
        for idx, grads in enumerate(param_gradients):
            env_grad = env_param_gradients[idx]
            grads.append(env_grad)

    assert len(param_gradients) == len(model_params)

    for param, grads in zip(model_params, param_gradients):

        # Calculate sign matrix
        grads = torch.stack(grads, dim=0)
        sign_matrix = torch.sign(grads)

        # Positive & Negative gradients
        positive_sign_matrix = sign_matrix > 0
        negative_sign_matrix = ~positive_sign_matrix

        # Temporarily replace 0 with 1 to calculate geometric mean
        positive_gradients = positive_sign_matrix * grads + negative_sign_matrix
        negative_gradients = negative_sign_matrix * grads + positive_sign_matrix

        # Temporarily replace 0 with 1 to prevent demoninator to be 0
        n_agreement_envs = len(grads)
        n_positive_envs = torch.sum(positive_sign_matrix, dim=0)
        n_negative_envs = torch.sum(negative_sign_matrix, dim=0)

        n_positive_envs_denominator = n_positive_envs + (n_positive_envs == 0)
        n_negative_envs_denominator = n_negative_envs + (n_negative_envs == 0)

        # Weighted geometric mean
        positive_prod_gradients = (n_positive_envs / n_agreement_envs) * torch.exp(torch.sum(
            torch.log(torch.abs(positive_gradients) + 1e-10), dim=0) / n_positive_envs_denominator)
        negative_prod_gradients = (n_negative_envs / n_agreement_envs) * torch.exp(torch.sum(
            torch.log(torch.abs(negative_gradients) + 1e-10), dim=0) / n_negative_envs_denominator)

        weighted_prod_grad = positive_prod_gradients - negative_prod_gradients

        # Mask
        mask = torch.mean(sign_matrix, dim=0).abs(
        ) >= flags.agreement_threshold
        mask = mask.to(torch.float32)
        assert mask.numel() == param.numel()

        mask_t = (mask.sum() / mask.numel())
        print(">>>>> Mask: {}".format(mask_t))
        # final_mask = mask / (1e-10 + mask_t)
        final_mask = mask

        param.grad = final_mask * weighted_prod_grad
