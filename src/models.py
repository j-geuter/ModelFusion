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
    def __init__(self, model_1, model_2, plan, dataset_1, dataset_2, dataset_star):
        super(MergeNN, self).__init__()
        self.nb_samples = plan.shape[0]
        self.models = (model_1, model_2)
        self.dataset_star = dataset_star

        nonzero_indices = torch.nonzero(plan)
        plan_permutation = nonzero_indices.unbind(1)[1]
        aligned_dataset_2 = deepcopy(dataset_2)
        aligned_dataset_2.permute_data(plan_permutation)
        self.datasets = (dataset_1, aligned_dataset_2)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # this is a k*2-tensor, where each row (i,j) means that the i-sample in x is equal to the j-sample in dataset_star
        matches = torch.eq(
            x.unsqueeze(1), self.dataset_star.features.unsqueeze(0)
        ).all(dim=2).nonzero()

        # reduce the number of matches to at most one per sample
        matches_x, matches_indices = torch.unique(matches[:, 0], return_inverse=True)
        if not len(matches_x) == len(matches_indices): # this means there are samples with multiple matches in the reference dataset
            unique_indices = torch.tensor([torch.where(matches_indices == i)[0][0] for i in matches_x])
            matches = matches[unique_indices]

        # find the x samples that are contained in dataset_star and those that aren't
        match_samples, match_indices = matches.unbind(1)
        unmatched_samples = torch.tensor([i for i in range(len(x)) if i not in match_samples])
        # x_matched = x[match_samples]
        # x_unmatched = x[unmatched_samples]
        assert len(unmatched_samples) == 0, "not implemented yet"

        # transport the samples to the outer datasets and compute the resp. predictions in those datasets
        x_transported = [dataset[match_indices][0] for dataset in self.datasets]
        y_transported = [model(samples) for model, samples in zip(self.models, x_transported)]
        y_projected = [project_labels(y, dataset.unique_labels) for y, dataset in zip(y_transported, self.datasets)]

        # transport the predictions back to dataset_star space and aggregate
        y_star = [self.project_labels(features, y, dataset_from=dataset) for features, y, dataset in zip(x_transported, y_projected, self.datasets)]
        y = sum(y_star) / len(y_star)
        return y

    def project_features(self, x, dataset_to):
        """
        Projects features `x` from `dataset_star` to `dataset_to` using the transport map.
        :param x: feature tensor of size k*d_x, where k is the number of samples, and d_x their dimension.
        :param dataset_to: the dataset that the features are mapped to.
        :return: feature tensor of size k*d_x', where d_x' is the dimension of features in `dataset_to`.
        """

    def project_labels(self, x, y, dataset_from):
        """
        Projects labels `y` from `dataset_from` back onto self.dataset_star using the transport map.
        :param x: tensor of size k*d_x, where k is the number of samples, and d_x their dimension.
        :param y: tensor of size k*d_y, where k is the number of labels, and d_y their dimension.
        :param dataset_from: the dataset that the labels `y` live in.
        :return: tensor of size k*d_star, where d_star is the dimension of labels in self.dataset_star.
        """
        output_labels = torch.zeros((y.shape[0], self.dataset_star.label_dim))
        for i in range(y.shape[0]):
            same_label_features, indices = dataset_from.get_samples_by_label(y[i])
            num_samples = same_label_features.shape[0]
            exponentials = torch.exp(
                -torch.norm(
                    same_label_features - x[i].expand(num_samples, -1),
                    dim=1
                )**2
            ).unsqueeze(1)
            y_transported = self.dataset_star[indices][1].T # labels transported to dataset_star
            y_aggregated = torch.matmul(y_transported, exponentials) / exponentials.sum()
            output_labels[i] = y_aggregated.squeeze()
        return output_labels



def project_labels(preds, labels):
    dists = torch.cdist(preds, labels)
    closest = torch.argmin(dists, dim=1)
    return labels[closest]



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