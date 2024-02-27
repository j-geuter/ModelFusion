from copy import copy, deepcopy

import ot
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemperatureScaledSoftmax(nn.Module):
    def __init__(self, temperature):
        super(TemperatureScaledSoftmax, self).__init__()
        self.temperature = temperature

    def forward(self, logits):
        scaled_logits = logits / self.temperature
        return F.softmax(scaled_logits, dim=-1)


class SimpleNN(nn.Module):
    def __init__(self, layer_sizes, weights=None, temperature=1, bias=True):
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
            + [
                nn.Sequential(
                    nn.Linear(layer_sizes[-2], layer_sizes[-1], bias=bias),
                    TemperatureScaledSoftmax(temperature),
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
        detached_weights = torch.cat(
            [param.data.detach().clone().view(-1) for param in trainable_parameters]
        )
        return detached_weights

    def update_weights(self, vector):
        if not vector.dim() == 1:
            vector = vector.flatten()
        assert (
            len(vector) == self.par_number
        ), "Weight vector size does not match parameter count!"
        vector_index = 0
        for param in self.parameters():
            # Determine the number of elements in the current parameter
            num_elements = param.data.numel()

            # Extract elements from the vector and reshape to the parameter shape
            param.data = (
                vector[vector_index : vector_index + num_elements]
                .detach()
                .clone()
                .view(param.data.shape)
            )

            # Move to the next set of elements in the vector
            vector_index += num_elements


class TransportNN(nn.Module):
    def __init__(
        self,
        models,
        datasets,
        dataset_star,
        plan=None,
        plan_indices=None,
        permute_star=False,
        eta=1,
        aggregate_method="all_features",
        temperature = 10
    ):
        """
        Creates a non-parametric model from `models` trained on `datasets`.
        The model can be used on `dataset_star`, but internally transports samples
        from `dataset_star` to the other `datasets` to use `models` in these domains,
        and then transporting their predictions back to `dataset_star`.
        :param models: iterable of models.
        :param datasets: iterable of datasets corresponding to `models`. Must be of
            same length.
        :param dataset_star: This is the domain the model will be used on.
        :param plan: Optional transport plan. If given, permutes all datasets indexed
            by `plan_indices` according to the plan.
        :param plan_indices: Optional indices of datasets to be permuted by `plan`.
        :param permute_star: If True, also permutes `dataset_star` along the `plan`.
        :param eta: Hyperparameter controlling the tradeoff between a feature- and a
            label-based loss in transporting samples.
        :param aggregate_method: Defines the method used to compute the transported label. One of 'match_label'
            (in which case only samples with matching labels are considered), 'all_samples' (in which
            case all samples are considered), 'all_features' (also averages over all samples, but only
            taking feature distances into account, ignoring label distances), 'all_labels' (also averages over
            all samples, but only considering label distances, ignoring features), or 'label_distance' (in
            which case the distance matrix between labels from the source dataset to dataset_star is used for transport).
        :param temperature: temperature for exponential smoothing of feature and label transport. Higher temperature
            equals less entropy, while lower temperature means more entropy.
        """
        super(TransportNN, self).__init__()
        assert len(models) == len(
            datasets
        ), "Length of `models` and `datasets` must be equal!"
        self.models = tuple(models)
        self.eta = eta  # weighs the distance between features with that between labels in the sample distance
        self.method = aggregate_method
        self.temperature = temperature
        if plan is not None:
            nonzero_indices = torch.nonzero(plan)
            plan_permutation = nonzero_indices.unbind(1)[1]
            if plan_indices is not None:
                datasets = list(datasets)
                for idx in plan_indices:
                    dataset = deepcopy(datasets[idx])
                    dataset.permute_data(plan_permutation)
                    datasets[idx] = dataset
                self.datasets = tuple(datasets)
            else:
                self.datasets = tuple(datasets)
            if permute_star:
                dataset_star = deepcopy(dataset_star)
                dataset_star.permute_data(plan_permutation)
                self.dataset_star = dataset_star
            else:
                self.dataset_star = dataset_star
        else:
            self.dataset_star = dataset_star
            self.datasets = tuple(datasets)

        # distances between labels within datasets
        self.label_distances = tuple(
            [compute_label_distances(dataset, dataset) for dataset in self.datasets]
        )

        # distances between labels across datasets w.r.t. dataset_star
        self.label_distances_star = tuple(
            [
                compute_label_distances(dataset_star, dataset)
                for dataset in self.datasets
            ]
        )

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # reshape x
        initial_shape = x.shape
        x = x.squeeze(1)
        x = x.view(x.shape[0], -1)

        # this is a k*2-tensor, where each row (i,j) means that the i-sample in x is equal to the j-sample in dataset_star
        matches = (
            torch.eq(
                x.unsqueeze(1),
                self.dataset_star.features.view(
                    self.dataset_star.features.shape[0], -1
                ).unsqueeze(0),
            )
            .all(dim=2)
            .nonzero()
        )

        # reduce the number of matches to at most one per sample
        matches_x, matches_indices = torch.unique(matches[:, 0], return_inverse=True)
        if not len(matches_x) == len(
            matches_indices
        ):  # this means there are samples with multiple matches in the reference dataset
            unique_indices = torch.tensor(
                [torch.where(matches_indices == i)[0][0] for i in matches_x]
            )
            matches = matches[unique_indices]

        # find the x samples that are contained in dataset_star and those that aren't
        match_samples, match_indices = matches.unbind(1)
        match_samples = match_samples.to(torch.int)
        unmatched_sample_indices = torch.tensor(
            [i for i in range(len(x)) if i not in match_samples], dtype=torch.int
        )

        if len(unmatched_sample_indices) != 0:
            x_unmatched = x[unmatched_sample_indices]
            unmatched_x_transported = [
                self.transport_features(x_unmatched, dataset)
                for dataset in self.datasets
            ]
        else:
            unmatched_x_transported = [
                torch.zeros((0,) + dataset.flattened_feature_dim)
                for dataset in self.datasets
            ]

        # transport the samples to the outer datasets and compute the resp. predictions in those datasets
        if len(match_indices) != 0:
            matched_x_transported = [
                dataset[match_indices][0] for dataset in self.datasets
            ]
        else:
            matched_x_transported = [
                torch.zeros((0,) + dataset.flattened_feature_dim)
                for dataset in self.datasets
            ]

        x_transported = [torch.zeros(x.shape) for _ in self.datasets]
        for item, x_matched, x_unmatched in zip(
            x_transported, matched_x_transported, unmatched_x_transported
        ):
            item[unmatched_sample_indices] = x_unmatched
            item[match_samples] = x_matched
        y_transported = [
            model(samples.view(torch.Size([samples.shape[0]]) + initial_shape[1:]))
            for model, samples in zip(self.models, x_transported)
        ]
        y_projected = [
            project_labels(y, dataset.unique_labels, dataset.high_dim_labels)
            for y, dataset in zip(y_transported, self.datasets)
        ]

        # transport the predictions back to dataset_star space and aggregate
        y_star = [
            self.transport_labels(
                features, y[0], y[1], dataset, label_distances, label_distances_star
            )
            for features, y, dataset, label_distances, label_distances_star in zip(
                x_transported,
                y_projected,
                self.datasets,
                self.label_distances,
                self.label_distances_star,
            )
        ]
        y = sum(y_star) / len(y_star)
        return y

    def transport_features(self, x, dataset_to):
        """
        Projects features `x` from `dataset_star` to `dataset_to` using the transport map.
        :param x: feature tensor of size k*d_x, where k is the number of samples, and d_x their dimension.
        :param dataset_to: the dataset that the features are mapped to.
        :return: feature tensor of size k*d_x', where d_x' is the dimension of features in `dataset_to`.
        """
        transported_features = dataset_to.features.view(
            dataset_to.features.shape[0], -1
        )
        dists = torch.cdist(
                self.dataset_star.features.view(
                    self.dataset_star.features.shape[0], -1
                ),
                x.view(x.shape[0], -1),
            ) ** 2
        dists /= dists.mean(0) # normalizes distances to avoid NaNs
        exponentials = torch.exp(-self.temperature * dists)
        weighted_transported_features = torch.matmul(
            transported_features.T, exponentials
        )
        normalized_transported_features = (
            weighted_transported_features / exponentials.sum(dim=0)
        )
        return normalized_transported_features.T

    def transport_labels(
        self,
        x,
        y,
        y_indices,
        dataset_from,
        label_distances,
        label_distances_star,
    ):
        """
        Transports labels `y` from `dataset_from` back onto self.dataset_star using the transport map.
        :param x: tensor of size k*d_x, where k is the number of samples, and d_x their dimension.
        :param y: tensor of size k*d_y, where k is the number of labels, and d_y their dimension.
        :param y_indices: indices of the labels amongst all labels of the dataset.
        :param dataset_from: the dataset that the labels `y` live in.
        :param label_distances: label distance matrix giving pairwise distances between labels within
            the two datasets.
        :param label_distances_star: label distance matrix giving pairwise distances between labels across
            datasets, comparing with labels in dataset_star.
        :return: tensor of size k*d_star, where d_star is the dimension of labels in self.dataset_star.
        """
        if self.method == "match_label":
            # define mask with (i,j)^th entry 1 iff the i^th label in the dataset is equal to the j^th label in y
            mask = torch.all(
                (dataset_from.labels.unsqueeze(1) == y.unsqueeze(0)), dim=2
            ).to(int)
            exponentials = (
                torch.exp(
                    -torch.cdist(
                        dataset_from.features.view(dataset_from.features.shape[0], -1),
                        x.view(x.shape[0], -1),
                    )
                    ** 2
                )
                * mask
            )

        elif self.method == "all_samples":
            # matrix of size len(dataset_from)*len(y), where entries are distances between labels
            label_distances = label_distances[dataset_from.label_indices, :][
                :, y_indices
            ]
            exponentials = torch.exp(
                -torch.cdist(
                    dataset_from.features.view(dataset_from.features.shape[0], -1),
                    x.view(x.shape[0], -1),
                )
                ** 2
                - self.eta * label_distances
            )

        elif self.method == "all_features":
            exponentials = torch.exp(
                -torch.cdist(
                    dataset_from.features.view(dataset_from.features.shape[0], -1),
                    x.view(x.shape[0], -1),
                )
                ** 2
            )

        elif self.method == "all_labels":
            label_distances = label_distances[dataset_from.label_indices, :][
                :, y_indices
            ]
            exponentials = torch.exp(-label_distances)

        elif self.method == "label_distance":
            label_distances = label_distances_star[self.dataset_star.label_indices, :][
                :, y_indices
            ]
            exponentials = torch.exp(-label_distances)

        else:
            raise ValueError(
                f"Method {self.method} not implemented! "
                f"Choose one of 'match_label', 'all_samples', 'all_features', or 'label_distance'."
            )

        transported_y = (
            self.dataset_star.labels
            if self.dataset_star.high_dim_labels is None
            else self.dataset_star.high_dim_labels
        )
        weighted_transported_labels = torch.matmul(transported_y.T, exponentials)
        normalized_transported_labels = (
            weighted_transported_labels / exponentials.sum(dim=0)
        ).T

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
            cost = (
                torch.cdist(
                    features_1.view(features_1.shape[0], -1),
                    features_2.view(features_2.shape[0], -1),
                )
                ** 2
            )
            transport_cost = ot.emd2(mu, nu, cost)
            label_distances[i][j] = transport_cost
    return label_distances


def project_labels(preds, labels, high_dim_labels):

    # if predictions and labels have the same dimension
    if preds[0].shape == labels[0].shape:
        dists = torch.cdist(preds, labels)
        closest = torch.argmin(dists, dim=1)
        return labels[closest], closest

    # if they do not have the same dimension, i.e. predictions are high-dimensional, labels are numbers
    else:
        closest = torch.argmax(preds, dim=1)
        return high_dim_labels[closest], closest


def pad_weights(net, layer_sizes, pad_from="top"):
    """
    Creates a new network of size `layer_sizes`, which uses the weights from `net` and pads them in each layer with zeros.
    :param layer_sizes: list of layer sizes of the new network, of the same format as `self.layer_sizes`.
    :param pad_from: either `top` or `bottom`, states where to pad from.
    :return: `SimpleNN` object.
    """
    assert len(layer_sizes) == len(
        net.layer_sizes
    ), "`layer_sizes` must be of same length as `self.layer_sizes`"
    new_weight_vector = torch.tensor([])
    layer_weights = [param for param in net.parameters() if param.dim() == 2]
    if net.bias:
        bias_weights = [param for param in net.parameters() if param.dim() == 1]
    for i in range(1, len(layer_sizes)):
        layer_size = layer_sizes[i] * layer_sizes[i - 1]
        nb_zeros = layer_size - net.layer_sizes[i] * net.layer_sizes[i - 1]
        if pad_from == "bottom":
            padded_weights = torch.cat(
                (layer_weights[i - 1].detach().clone().flatten(), torch.zeros(nb_zeros))
            )
        elif pad_from == "top":
            padded_weights = torch.cat(
                (torch.zeros(nb_zeros), layer_weights[i - 1].detach().clone().flatten())
            )
        new_weight_vector = torch.cat((new_weight_vector, padded_weights))
        if net.bias:
            nb_zeros = layer_sizes[i] - net.layer_sizes[i]
            if pad_from == "bottom":
                padded_bias = torch.cat(
                    (
                        bias_weights[i - 1].detach().clone().flatten(),
                        torch.zeros(nb_zeros),
                    )
                )
            elif pad_from == "top":
                padded_bias = torch.cat(
                    (
                        torch.zeros(nb_zeros),
                        bias_weights[i - 1].detach().clone().flatten(),
                    )
                )
            new_weight_vector = torch.cat((new_weight_vector, padded_bias))

    return SimpleNN(
        layer_sizes,
        weights=new_weight_vector,
        bias=net.bias,
        temperature=net.temperature,
    )


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, location=None):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.par_number = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if location is not None:
            self.load_model(location)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def save_model(self, location):
        """
        Save the model to `location`.
        """
        torch.save(self.state_dict(), location)
        print(f"Model saved to {location}")

    def load_model(self, location):
        """
        Load a model from `location`.
        """
        self.load_state_dict(torch.load(location))
        print(f"Model loaded from {location}")
