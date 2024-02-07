import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import copy, deepcopy

class TemperatureScaledSoftmax(nn.Module):
    def __init__(self, temperature):
        super(TemperatureScaledSoftmax, self).__init__()
        self.temperature = temperature

    def forward(self, logits):
        scaled_logits = logits / self.temperature
        return F.softmax(scaled_logits, dim=-1)

class SimpleNN(nn.Module):
    def __init__(self, layer_sizes, weights = None, temperature = 1, bias = True):
        super(SimpleNN, self).__init__()

        self.layer_sizes = tuple(copy(layer_sizes))
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(layer_sizes[l], layer_sizes[l + 1], bias=bias),
                    nn.ReLU(),
                )
                for l in range(len(layer_sizes) - 2)
            ]
            +
            [
                nn.Sequential(
                nn.Linear(layer_sizes[-2], layer_sizes[-1], bias=bias),
                TemperatureScaledSoftmax(temperature)
                )
            ]
        )

        self.par_number = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.bias = bias
        self.temperature = temperature

        if weights is not None:
            self.update_weights(weights)

    def forward(self, x):
        # Define the forward pass
        for layer in self.layers:
            x = layer(x)
        return x

    def get_weight_tensor(self):
        trainable_parameters = list(self.parameters())
        detached_weights = torch.cat([param.data.detach().clone().view(-1) for param in trainable_parameters])
        return detached_weights

    def update_weights(self, vector):
        if not vector.dim() == 1:
            vector = vector.flatten()
        assert len(vector) == self.par_number, "Weight vector size does not match parameter count!"
        vector_index = 0
        for param in self.parameters():
            # Determine the number of elements in the current parameter
            num_elements = param.data.numel()

            # Extract elements from the vector and reshape to the parameter shape
            param.data = vector[vector_index:vector_index + num_elements].detach().clone().view(param.data.shape)

            # Move to the next set of elements in the vector
            vector_index += num_elements


class MergeNN(nn.Module):
    def __init__(self, model_1, model_2, plan_2, dataset_1, dataset_2, dataset_star):
        super(MergeNN, self).__init__()
        self.nb_samples = plan_2.shape[0]
        self.model_1 = model_1
        self.model_2 = model_2
        self.plan_1 = torch.eye(self.nb_samples)
        self.plan_2 = plan_2
        self.dataset_1 = deepcopy(dataset_1)
        self.dataset_2 = deepcopy(dataset_2)
        self.dataset_star = deepcopy(dataset_star)
        nonzero_indices = torch.nonzero(plan_2)
        self.forward_indices = nonzero_indices.unbind(1)[1]
        self.inverse_indices = self.forward_indices.sort()[1]

        # align dataset_2 with the other datasets
        for i in range(2):
            self.dataset_star[i] = self.dataset_star[self.forward_indices]

    def forward(self, x):


def forward_plan(x, from_dataset, to_dataset):





def pad_weights(net, layer_sizes, pad_from='top'):
    """
    Creates a new network of size `layer_sizes`, which uses the weights from `net` and pads them in each layer with zeros.
    :param layer_sizes: list of layer sizes of the new network, of the same format as `self.layer_sizes`.
    :param pad_from: either `top` or `bottom`, states where to pad from.
    :return: `SimpleNN` object.
    """
    assert len(layer_sizes) == len(net.layer_sizes), "`layer_sizes` must be of same length as `self.layer_sizes`"
    new_weight_vector = torch.tensor([])
    layer_weights = [param for param in net.parameters() if param.dim()==2]
    if net.bias:
        bias_weights = [param for param in net.parameters() if param.dim()==1]
    for i in range(1, len(layer_sizes)):
        layer_size = layer_sizes[i] * layer_sizes[i - 1]
        nb_zeros = layer_size - net.layer_sizes[i] * net.layer_sizes[i - 1]
        if pad_from == 'bottom':
            padded_weights = torch.cat((layer_weights[i-1].detach().clone().flatten(), torch.zeros(nb_zeros)))
        elif pad_from == 'top':
            padded_weights = torch.cat((torch.zeros(nb_zeros), layer_weights[i - 1].detach().clone().flatten()))
        new_weight_vector = torch.cat((new_weight_vector, padded_weights))
        if net.bias:
            nb_zeros = layer_sizes[i] - net.layer_sizes[i]
            if pad_from == 'bottom':
                padded_bias = torch.cat(
                    (bias_weights[i - 1].detach().clone().flatten(), torch.zeros(nb_zeros)))
            elif pad_from == 'top':
                padded_bias = torch.cat(
                    (torch.zeros(nb_zeros), bias_weights[i - 1].detach().clone().flatten()))
            new_weight_vector = torch.cat((new_weight_vector, padded_bias))

    return SimpleNN(layer_sizes, weights=new_weight_vector, bias=net.bias, temperature=net.temperature)