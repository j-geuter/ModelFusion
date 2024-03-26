from copy import copy, deepcopy

import math
import ot
import torch
import torch.nn as nn
import torch.nn.functional as F

from synthdatasets import CustomDataset
from costmatrix import euclidean_cost_matrix
from sinkhorn import sinkhorn
from utils import class_correspondences

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        source_datasets,
        target_dataset,
        plans=None,
        eta=1,
        feature_method="plain_softmax",
        label_method="plain_softmax",
        feature_dists=True,
        label_dists=True,
        project_source_labels=False,
        temperature=100,
        f=None,
        g=None,
        reg=None,
    ):
        """
        Creates a non-parametric model from `models` trained on `source_datasets`.
        The model can be used on `target_dataset`, but internally transports samples
        from `target_dataset` to the other `source_datasets` to use `models` in these domains,
        and then transporting their predictions back to `target_dataset`.
        :param models: iterable of models. Can also be a single model.
        :param source_datasets: iterable of source_datasets corresponding to `models`. Must be of
            same length. Can also be a single dataset.
        :param target_dataset: This is the domain the model will be used on.
        :param plans: Optional transport plan(s) - either a single plan, or a list of plans of the
            same length as `source_datasets`. If given, creates aligned versions of the source datasets
            and target dataset according to `plans`.
        :param eta: Hyperparameter controlling the tradeoff between a feature- and a
            label-based loss in transporting samples.
        :param feature_method: Defines the method used to transport the features. One of
            "plain_softmax" (softmax with distances in target feature space), or
            "plugin" (softmax using the dual potential and cross-dataset distances).
        :param label_method: Defines the method used to transport the labels from source to target
            domain. One of "plain_softmax" (softmax with distances in source domain),
            "masked_softmax" (softmax only over samples with matching label; should only be
            used in conjunction with `project_source_labels`==True); or
            "plugin" (softmax using the dual potential and cross-dataset distances); or
            "label_correspondences" (does not transport labels back to target space, but computes
            predictions from soft labels in source space, by comparing against the label correspondences
            given by the transport plan between the train datasets).
        :param feature_dists: Toggles whether to use feature distances in computing cost for label transport.
        :param label_dists: Toggles whether to use label distances in computing cost for label transport.
        :param project_source_labels: if True, projects labels onto hard labels in source dataset space;
            otherwise, leaves them as soft labels.
        :param temperature: temperature for exponential smoothing of feature and label transport. Higher temperature
            equals less entropy, while lower temperature means more entropy.
        :param f: dual potential(s) of the OT map between source_datasets. If passed, compute
            the plug-in estimates using the dual potential. Corresponds to source distribution(s). Can be either
            a tensor if just one model is passed, or a list of dual potentials for multiple models.
        :param g: second dual potential(s), corresponding to target distribution(s).
        :param reg: regularizer used in the transport plan. Only needed if plug-in estimates
            are computed using the dual potential, i.e. when `dual_potential` is not None.
        """
        super(TransportNN, self).__init__()
        if isinstance(models, nn.Module):
            self.models = (models,)
        else:
            self.models = tuple(models)
        self.eta = eta  # weighs the distance between features with that between labels in the sample distance
        self.feature_method = feature_method
        self.label_method = label_method
        self.temperature = temperature
        self.project_source_labels = project_source_labels
        self.feature_dists = feature_dists
        self.label_dists = label_dists
        if isinstance(f, torch.Tensor):
            self.f = [f]
        elif f is None:
            self.f = [None for _ in range(len(self.models))]
        else:
            self.f = f
        if isinstance(g, torch.Tensor):
            self.g = [g]
        elif g is None:
            self.g = [None for _ in range(len(self.models))]
        else:
            self.g = g
        self.reg = reg
        if isinstance(source_datasets, CustomDataset):
            source_datasets = [source_datasets]
        self.target_dataset = target_dataset
        self.source_datasets = tuple(source_datasets)

        if isinstance(plans, torch.Tensor):
            plans = [plans]
        self.plans = tuple(plans)
        aligned_source_datasets = []
        aligned_target_datasets = []
        for plan, dataset in zip(plans, source_datasets):
            # align target dataset
            aligned_target_features = dataset.num_samples * torch.einsum(
                "nl,lxy->nxy", plan, target_dataset.features
            )
            aligned_target_labels = dataset.num_samples * torch.matmul(
                plan, target_dataset.high_dim_labels
            )
            aligned_target_datasets.append(
                CustomDataset(
                    aligned_target_features, aligned_target_labels, low_dim_labels=False
                )
            )

            # align source dataset
            aligned_source_features = target_dataset.num_samples * torch.einsum(
                "nl,lxy->nxy", plan.T, dataset.features
            )
            aligned_source_labels = target_dataset.num_samples * torch.matmul(
                plan.T, dataset.high_dim_labels
            )
            aligned_source_datasets.append(
                CustomDataset(
                    aligned_source_features, aligned_source_labels, low_dim_labels=False
                )
            )
        self.aligned_source_datasets = tuple(aligned_source_datasets)
        self.aligned_target_datasets = tuple(aligned_target_datasets)

        assert len(self.models) == len(
            self.source_datasets
        ), "Length of `models` and `source_datasets` must be equal!"

        # distances between labels within source_datasets
        self.source_label_distances = tuple(
            [
                compute_label_distances(dataset, dataset)
                for dataset in self.source_datasets
            ]
        )

        self.cross_label_distances = tuple(
            [
                compute_label_distances(dataset, self.target_dataset)
                for dataset in self.source_datasets
            ]
        )

        self.label_correspondences = tuple(
            [
                class_correspondences(
                    dataset, self.target_dataset, plan=plan, symmetric=False, plot=False
                )
                for dataset, plan in zip(self.source_datasets, plans)
            ]
        )

        self.direct_interpolation = (
            False  # if True, compute predictions by interpolating between
        )
        # sample predictions in `target_dataset` space, independent of the model

    def forward(self, x):
        if self.direct_interpolation:
            return self.interpolate(x)
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # reshape x
        x = x.squeeze(1)
        x = x.view(x.shape[0], -1)  # now of shape N*d (N=number samples, d=sample dim)

        # transport x to the source dataset
        x_s = [
            self.transport_features(x, dataset, aligned_dataset, f)
            for dataset, aligned_dataset, f in zip(
                self.source_datasets, self.aligned_source_datasets, self.f
            )
        ]

        # make source predictions using model
        y_s = [
            model(samples.view(samples.shape[0], 1, model.dim, model.dim))
            for model, samples in zip(self.models, x_s)
        ]
        y_s = [y.softmax(dim=1) for y in y_s]

        # project predictions to label set
        if self.project_source_labels:
            y_s = [
                project_labels(y, dataset.unique_labels, dataset.high_dim_labels)[0]
                for y, dataset in zip(y_s, self.source_datasets)
            ]

        # transport the predictions back to target_dataset space and aggregate
        y_t = [
            self.transport_labels(
                features,
                y,
                dataset,
                aligned_target_dataset,
                source_label_distances,
                cross_label_distances,
                potential,
                label_correspondences,
            )
            for features, y, dataset, aligned_target_dataset, source_label_distances, cross_label_distances, potential, label_correspondences in zip(
                x_s,
                y_s,
                self.source_datasets,
                self.aligned_target_datasets,
                self.source_label_distances,
                self.cross_label_distances,
                self.g,
                self.label_correspondences,
            )
        ]
        y_t = sum(y_t) / len(y_t)
        return y_t

    def interpolate(self, x):
        """
        Computes predictions in `target_dataset` space by interpolating between samples.
        This is independent of the model or `self.source_datasets`, is learning-free, and serves
        as a baseline for more sophisticated prediction algorithms.
        :param x: input.
        :return: output prediction.
        """
        dists = (
            torch.cdist(
                self.target_dataset.features.view(
                    self.target_dataset.features.shape[0], -1
                ),
                x.view(x.shape[0], -1),
            )
            ** 2
        ).to(device)
        dists /= dists.mean(0)  # normalizes distances to avoid NaNs
        exponentials = torch.exp(-self.temperature * dists).to(device)
        y = (
            self.target_dataset.labels
            if self.target_dataset.high_dim_labels is None
            else self.target_dataset.high_dim_labels
        )
        weighted_labels = torch.matmul(y.T, exponentials)
        normalized_labels = (weighted_labels / exponentials.sum(dim=0)).T
        return normalized_labels

    def transport_features(self, x, dataset, aligned_dataset, f):
        """
        Projects features `x` from target dataset to source dataset using the transport map.
        :param x: feature tensor of size k*d_x, where k is the number of samples, and d_x their dimension.
        :param dataset: the dataset that the features are mapped to.
        :param aligned_dataset: the dataset the features are mapped to, aligned w.r.t. the target dataset.
        :param f: dual potential corresponding to source distribution. If passed, compute the plugin
            estimator using dual potentials. Can be set to None.
        :return: feature tensor of size k*d_x', where d_x' is the dimension of features in `dataset_to`.
        """
        if self.feature_method == "plain_softmax":
            exponentials = self.exponential_matrix(self.target_dataset, x)
            features = aligned_dataset.features.view(
                aligned_dataset.features.shape[0], -1
            )

        elif self.feature_method == "plugin":
            exponentials = self.exponential_matrix(dataset, x, potential=f)
            features = dataset.features.view(dataset.features.shape[0], -1)

        else:
            raise ValueError(
                "`feature_method` must be one of the following:"
                "`plain_softmax`, `plugin`."
            )

        weighted_features = features.T @ exponentials
        normalized_features = weighted_features / exponentials.sum(dim=0)

        return normalized_features.T

    def transport_labels(
        self,
        x,
        y,
        source_dataset,
        aligned_target_dataset,
        source_label_distances,
        cross_label_distances,
        potential,
        label_correspondences,
    ):
        """
        Transports labels `y` from `source_dataset` back onto self.target_dataset using the transport map.
        :param x: tensor of size k*d_x, where k is the number of samples, and d_x their dimension.
        :param y: tensor of size k*d_y, where k is the number of labels, and d_y their dimension.
        :param source_dataset: the dataset that the labels `y` live in.
        :param aligned_target_dataset: target dataset, aligned to `source_dataset`.
        :param source_label_distances: label distance matrix giving pairwise distances between labels within
            the source dataset.
        :param cross_label_distances: label distance matrix giving pairwise distances between labels from
            source and target dataset.
        :param potential: potential used in computing exponentials, if `self.label_method`=="plugin"
            is used.
        :param label_correspondences: matrix containing correspondences between labels. Can e.g. be the OT plan
            between labels with cost being the distance between labels.
        :return: tensor of size k*d^t, where d^t is the dimension of labels in self.target_dataset.
        """
        x = x if self.feature_dists else None
        y = y if self.label_dists else None

        if self.label_method == "masked_softmax":
            # define mask with (i,j)^th entry 1 iff the i^th label in the dataset is equal to the j^th label in y
            mask = torch.all(
                (source_dataset.labels.unsqueeze(1) == y.unsqueeze(0)), dim=2
            ).to(int).to(device)
            exponentials = (
                self.exponential_matrix(source_dataset, x, y, source_label_distances)
                * mask
            )
            labels = aligned_target_dataset.high_dim_labels

        elif self.label_method == "plain_softmax":
            exponentials = self.exponential_matrix(
                source_dataset, x, y, source_label_distances
            )
            labels = aligned_target_dataset.high_dim_labels

        elif self.label_method == "plugin":
            exponentials = self.exponential_matrix(
                self.target_dataset, x, y, cross_label_distances, potential
            )
            labels = self.target_dataset.high_dim_labels

        elif self.label_method == "label_correspondences":
            target_labels = label_correspondences.T @ (y.T)
            target_labels = target_labels.T
            return target_labels

        else:
            raise ValueError(
                "`self.label_method` must be set to one of the following:"
                "`plain_softmax`, `masked_softmax`, `label_correspondences`, or `plugin`."
            )

        weighted_labels = labels.T @ exponentials
        normalized_labels = weighted_labels / exponentials.sum(dim=0)

        return normalized_labels.T.softmax(dim=1)

    def exponential_matrix(
        self, dataset, x=None, y=None, label_distances=None, potential=None
    ):
        """
        Compute a matrix of pairwise distances between samples, used in transporting
        labels and features between datasets.
        :param dataset: Dataset to which the distances of `x` and `y` are computed.
        :param x: Sample features. If None, only sample labels are considered in distance.
        :param y: Sample labels. If None, only sample features are considered in distance.
        :param label_distances: pairwise distance matrix between labels, used for computing
            distances between labels `y` and labels in `dataset`. 2-dimensional tensor,
            where first dimension is the number of labels in the dataset of `y`, and the
            second dimension is the number of labels in `dataset`.
        :param potential: Dual OT potential. If given, computes a dual-plugin estimate
            of the costs (see "Entropic estimation of optimal transport maps" paper).
        :return: Cost matrix.
        """
        num_samples = x.shape[0] if x is not None else y.shape[0]
        dists = torch.zeros((dataset.features.shape[0], num_samples)).to(device)
        if x is not None:
            dists += (
                torch.cdist(
                    dataset.features.view(dataset.features.shape[0], -1),
                    x.view(x.shape[0], -1),
                )
                ** 2
            ).to(device)
        if y is not None:
            label_distances = label_distances.T @ y.T
            label_distances = dataset.high_dim_labels @ label_distances
            dists += self.eta * label_distances
        if potential is not None:
            dists = potential.unsqueeze(1) - dists
            dists /= self.reg
            dists = torch.exp(dists)
        else:
            dists /= dists.mean(0)
            dists = -self.temperature * dists
            dists = torch.exp(dists)
        return dists


def compute_label_distances(
    dataset_1, dataset_2, ot_dists=False, samples_per_label=100
):
    """
    Compute the pairwise distances between labels in `dataset_1` and `dataset_2` via
        the squared Wasserstein-2 distance between the feature distributions of those labels.
    :param dataset_1: first dataset with k labels.
    :param dataset_2: second dataset with l labels.
    :param ot_dists: if True, uses OT distances between feature samples instead of Euclidean distances.
    :param samples_per_label: maximum number of samples per label used in computing
        pairwise OT distances, in order to reduce runtime.
    :return: k*l label distance matrix containing squared Wasserstein-2 distances.
    """
    labels_1 = dataset_1.unique_labels
    labels_2 = dataset_2.unique_labels
    label_distances = torch.zeros((len(labels_1), len(labels_2))).to(device)
    for i in range(len(labels_1)):
        for j in range(len(labels_2)):
            features_1 = dataset_1.get_samples_by_label(labels_1[i])[0]
            features_2 = dataset_2.get_samples_by_label(labels_2[j])[0]
            nb_samples_1 = len(features_1)
            nb_samples_2 = len(features_2)
            features_1 = features_1.view(features_1.shape[0], -1)
            features_2 = features_2.view(features_2.shape[0], -1)
            dim_1 = features_1.shape[1]
            dim_2 = features_2.shape[1]
            if ot_dists == False:
                mu = torch.ones(nb_samples_1) / nb_samples_1
                nu = torch.ones(nb_samples_2) / nb_samples_2
                mu = mu.to(device)
                nu = nu.to(device)
                distances = (
                    torch.cdist(
                        features_1,
                        features_2,
                    )
                    ** 2
                ).to(device)
                # distances /= distances.max()
            else:  # computes pairwise entropic OT distances between features
                mu = torch.ones(samples_per_label) / samples_per_label
                nu = torch.ones(samples_per_label) / samples_per_label
                mu = mu.to(device)
                nu = nu.to(device)
                assert dim_1 == dim_2, "feature dimensions must match!"
                features_1 = features_1[:samples_per_label]
                features_2 = features_2[:samples_per_label]
                features_1 -= features_1.min()
                features_1 += 1e-3
                features_1 /= features_1.sum(dim=1).unsqueeze(1)
                features_2 -= features_2.min()
                features_2 += 1e-3
                features_2 /= features_2.sum(dim=1).unsqueeze(1)
                width = int(math.sqrt(dim_1))
                cost = euclidean_cost_matrix(width, width).to(device)
                cost /= cost.max()
                source_indices = torch.arange(samples_per_label).repeat_interleave(
                    samples_per_label
                ).to(device)
                source_dists = features_1[source_indices]
                target_dists = features_2.repeat((samples_per_label, 1)).view(-1, dim_2).to(device)
                distances = sinkhorn(
                    source_dists,
                    target_dists,
                    cost,
                    0.01,
                    max_iter=200,
                )['plan'].reshape((samples_per_label, samples_per_label, dim_1, dim_2)).to(device)
                distances /= distances.max()

            transport_cost = ot.emd2(mu, nu, distances)
            label_distances[i][j] = transport_cost
    return label_distances


def project_labels(preds, labels, high_dim_labels):

    # if predictions and labels have the same dimension
    if preds[0].shape == labels[0].shape:
        dists = torch.cdist(preds, labels).to(device)
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
    def __init__(self, num_classes=10, dim=28, location=None):
        self.dim = dim
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        reduced_dim = int(self.dim / 4)
        self.fc1 = nn.Linear(64 * reduced_dim * reduced_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.par_number = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if location is not None:
            self.load(location)

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

    def save(self, location):
        """
        Save the model to `location`.
        """
        torch.save(self.state_dict(), location)
        print(f"Model saved to {location}")

    def load(self, location):
        """
        Load a model from `location`.
        """
        self.load_state_dict(torch.load(location))
        print(f"Model loaded from {location}")
