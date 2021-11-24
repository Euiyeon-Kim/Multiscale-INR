import torch
from torch.autograd import grad as torch_grad, Variable


def calcul_gradient_penalty(d, real_data, generated_data, device):
    # calculate interpolation
    alpha = torch.rand(real_data.size(0), 1, 1, 1)
    alpha = alpha.expand_as(real_data)
    alpha = alpha.to(device)

    interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
    interpolated = Variable(interpolated, requires_grad=True)
    interpolated = interpolated.to(device)

    # calculate probability of interpolated examples
    prob_interpolated = d(interpolated)

    # calculate gradients of probabilities with respect to examples
    gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                           grad_outputs=torch.ones(prob_interpolated.size()).to(device),
                           create_graph=True, retain_graph=True, only_inputs=True)[0]

    # return gradient penalty
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()