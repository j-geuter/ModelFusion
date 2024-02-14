import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import copy, deepcopy
import ot

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
    def __init__(self, model_1, model_2, plan, dataset_1, dataset_2, dataset_star, eta=.01):
        super(MergeNN, self).__init__()
        self.nb_samples = plan.shape[0]
        self.models = (model_1, model_2)
        self.dataset_star = dataset_star
        self.eta = eta # weighs the distance between features with that between labels in the sample distance

        nonzero_indices = torch.nonzero(plan)
        plan_permutation = nonzero_indices.unbind(1)[1]
        aligned_dataset_2 = deepcopy(dataset_2)
        aligned_dataset_2.permute_data(plan_permutation)
        self.datasets = (dataset_1, aligned_dataset_2)
        self.label_distances = (
            compute_label_distances(dataset_1, dataset_star),
            compute_label_distances(aligned_dataset_2, dataset_star)
        )

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
        match_samples = match_samples.to(torch.int)
        unmatched_sample_indices = torch.tensor([i for i in range(len(x)) if i not in match_samples], dtype=torch.int)

        if len(unmatched_sample_indices) != 0:
            x_unmatched = x[unmatched_sample_indices]
            unmatched_x_transported = [self.transport_features(x_unmatched, dataset) for dataset in self.datasets]
        else:
            unmatched_x_transported = [torch.zeros((0, dataset.feature_dim)) for dataset in self.datasets]
        # transport the samples to the outer datasets and compute the resp. predictions in those datasets
        if len(match_indices) != 0:
            matched_x_transported = [dataset[match_indices][0] for dataset in self.datasets]
        else:
            matched_x_transported = [torch.zeros((0, dataset.feature_dim)) for dataset in self.datasets]

        x_transported = [torch.zeros(x.shape) for _ in self.datasets]
        for item, x_matched, x_unmatched in zip(x_transported, matched_x_transported, unmatched_x_transported):
            item[unmatched_sample_indices] = x_unmatched
            item[match_samples] = x_matched
        y_transported = [model(samples) for model, samples in zip(self.models, x_transported)]
        y_projected = [project_labels(y, dataset.unique_labels) for y, dataset in zip(y_transported, self.datasets)]

        # transport the predictions back to dataset_star space and aggregate
        y_star = [self.transport_labels(features, y[0], y[1], dataset, label_distances)
                  for features, y, dataset, label_distances in
                  zip(x_transported, y_projected, self.datasets, self.label_distances)]
        y = sum(y_star) / len(y_star)
        return y


    def transport_features(self, x, dataset_to):
        """
        Projects features `x` from `dataset_star` to `dataset_to` using the transport map.
        :param x: feature tensor of size k*d_x, where k is the number of samples, and d_x their dimension.
        :param dataset_to: the dataset that the features are mapped to.
        :return: feature tensor of size k*d_x', where d_x' is the dimension of features in `dataset_to`.
        """
        transported_features = dataset_to.features
        exponentials = torch.exp(
            -torch.cdist(
                self.dataset_star.features, x
            ) ** 2
        )
        weighted_transported_features = torch.matmul(transported_features.T, exponentials)
        normalized_transported_features = weighted_transported_features / exponentials.sum(dim=0)
        return normalized_transported_features.T


    def transport_labels(self, x, y, y_indices, dataset_from, label_distances, method='all_samples'):
        """
        Transports labels `y` from `dataset_from` back onto self.dataset_star using the transport map.
        :param x: tensor of size k*d_x, where k is the number of samples, and d_x their dimension.
        :param y: tensor of size k*d_y, where k is the number of labels, and d_y their dimension.
        :param y_indices: indices of the labels amongst all labels of the dataset.
        :param dataset_from: the dataset that the labels `y` live in.
        :param label_distances: label distance matrix giving pairwise distances between labels across
            the two datasets.
        :param method: defines the method used to compute the transported label. One of 'match_label'
            (in which case only samples with matching labels are considered), 'all_samples' (in which
            case all samples are considered), or 'label_distance' (in which case the distance matrix
            between labels is used for transport).
        :return: tensor of size k*d_star, where d_star is the dimension of labels in self.dataset_star.
        """
        if method == 'match_label':

            # define mask with (i,j)^th entry 1 iff the i^th label in the dataset is equal to the j^th label in y
            mask = torch.all((dataset_from.labels.unsqueeze(1)==y.unsqueeze(0)), dim=2).to(int)

            transported_y = self.dataset_star.labels
            masked_exponentials = torch.exp(
                - torch.cdist(
                    dataset_from.features, x
                ) ** 2
            ) * mask
            weighted_transported_labels = torch.matmul(transported_y.T, masked_exponentials)
            normalized_transported_labels = (weighted_transported_labels / masked_exponentials.sum(dim=0)).T
        elif method == 'all_samples':

            # matrix of size len(dataset_from)*len(y), where entries are distances between labels
            label_distances = label_distances[dataset_from.label_indices, :][:, y_indices]
            print(label_distances.abs().mean())
            print((torch.cdist(
                    dataset_from.features, x
                ) ** 2).abs().mean())
            exponentials = torch.exp(
                - torch.cdist(
                    dataset_from.features, x
                ) ** 2
                - self.eta * label_distances
            )
            transported_y = self.dataset_star.labels
            weighted_transported_labels = torch.matmul(transported_y.T, exponentials)
            normalized_transported_labels = (weighted_transported_labels / exponentials.sum(dim=0)).T


        elif method == 'label_distance':
            pass
        else:
            raise ValueError(f"Method {method} not implemented! "
                             f"Choose one of 'match_label', 'all_samples', or 'label_distance'.")
        return normalized_transported_labels


def compute_label_distances(dataset_1, dataset_2):
    """
    Compute the pairwise distances between labels in `dataset_1` and `dataset_2` via
        the squared Wasserstein-2 distance between the feature distributions of those labels.
    :param dataset_1: first dataset with k labels.
    :param dataset_2: second dataset with l labels.
    :return: k*l label distance matrix containing squared Wasserstein-2 distances.
    """
    labels_1 = dataset_1.unique_labels
    labels_2 = dataset_2.unique_labels
    label_distances = torch.zeros((len(labels_1), len(labels_2)))
    for i in range(len(labels_1)):
        for j in range(len(labels_2)):
            features_1 = dataset_1.get_samples_by_label(labels_1[i])[0]
            features_2 = dataset_2.get_samples_by_label(labels_2[j])[0]
            mu = torch.ones(len(features_1)) / len(features_1)
            nu = torch.ones(len(features_2)) / len(features_2)
            cost = torch.cdist(features_1, features_2)**2
            transport_cost = ot.emd2(mu, nu, cost)
            label_distances[i][j] = transport_cost
    return label_distances



def project_labels(preds, labels):
    dists = torch.cdist(preds, labels)
    closest = torch.argmin(dists, dim=1)
    return labels[closest], closest



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