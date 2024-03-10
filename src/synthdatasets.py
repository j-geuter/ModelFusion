import matplotlib.pyplot as plt
import numpy as np
import ot
import torch
from torch.distributions import Categorical, MixtureSameFamily, MultivariateNormal
from torch.utils.data import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CustomDataset(Dataset):
    def __init__(self, features, labels, low_dim_labels=True):
        """
        Creates a dataset.
        :param features: Features of the dataset.
        :param labels: Labels of the dataset.
        :param low_dim_labels: Set to True if the labels are low dimensional,
            e.g. digits. Set to False if labels are high-dimensional, e.g. one-hot-vectors.
        """
        self.features = features
        if labels.dim() == 1:
            labels = labels.unsqueeze(1)
        self._labels = labels
        self.num_samples, self.feature_dim = features.shape[0], features.shape[1:]
        self.flattened_feature_dim = features.view(self.num_samples, -1).shape[1:]
        self.label_dim = labels.shape[1:]

        # Find unique labels and save them as an attribute
        unique_labels, label_counts = torch.unique(labels, dim=0, return_counts=True)
        self.unique_labels = unique_labels
        self.label_counts = label_counts
        self.num_unique_labels = len(unique_labels)
        self.label_indices = (
            torch.all(self._labels[:, None, :] == self.unique_labels[None, :, :], dim=2)
            .to(int)
            .argmax(dim=1)
        )
        if (
            low_dim_labels
        ):  # this means the labels are low dimensional, and will create high dimensional counterparts
            self.high_dim_unique_labels = torch.eye(self.num_unique_labels)
            self.high_dim_labels = self.high_dim_unique_labels[self.label_indices]
        else:
            self.high_dim_labels = None
            self.high_dim_unique_labels = None

    def __len__(self):
        return self.num_samples

    @property
    def labels(self):
        return self._labels

    @labels.setter
    def labels(self, new_labels):
        self._labels = new_labels
        _, self.label_dim = new_labels.shape
        unique_labels, label_counts = torch.unique(
            new_labels, dim=0, return_counts=True
        )
        self.unique_labels = unique_labels
        self.label_counts = label_counts
        self.num_unique_labels = len(unique_labels)
        self.label_indices = (
            torch.all(self._labels[:, None, :] == self.unique_labels[None, :, :], dim=2)
            .to(int)
            .argmax(dim=1)
        )

    def permute_data(self, permutation):
        features, labels = self.features[permutation], self.labels[permutation]
        self.__init__(features, labels)

    def __getitem__(self, index):
        return self.features[index], self._labels[index]

    def get_samples_by_label(self, target_label, idx=None):
        # Given a target_label, return features of samples with that label and the indices
        mask = torch.all(self._labels == target_label, dim=1)
        if idx is not None:
            mask[idx] = False
        return self.features[mask], torch.where(mask)[0]


class GMM:

    def __init__(
        self,
        d=2,
        train_N=1000,
        test_N=500,
        l=2,
        lambdas=None,
        mus=None,
        Sigmas=None,
        exact_lambdas=False,
    ):
        """
        Class to create Gaussian mixture models. Creates `N` training samples.
        :param d: dimension.
        :param train_N: number of training samples.
        :param test_N: number of test samples.
        :param l: number of components.
        :param lambdas: component coefficients.
        :param mus: component means. Tensor of dimension `(l, d)`, defaults to random means.
        :param Sigmas: component covariances. Tensor of dimension `(l, d, d)`, defaults to identity covariance for each component.
        :param exact_lambdas: if True, draws samples precisely according to the `lambdas` probabilities. Otherwise,
            draws them probabilistically.
        """
        self.d = d
        self.train_N = train_N
        self.test_N = test_N
        self.l = l
        self.exact_lambdas = exact_lambdas
        if lambdas == None:
            lambdas = torch.ones(l)
        lambdas /= sum(lambdas)
        self.lambdas = lambdas.to(device)
        if mus == None:
            mus = torch.randn(l, d)
        assert len(lambdas) == l
        assert len(mus) == l
        if Sigmas == None:
            Sigmas = torch.eye(d).unsqueeze(0).expand(l, -1, -1)
        assert len(Sigmas) == l
        self.dists = [
            MultivariateNormal(mus[i].to(device), Sigmas[i].to(device))
            for i in range(l)
        ]
        self.train_samples = self.sample(train_N)
        self.test_samples = self.sample(test_N)

    def sample(self, n):
        """
        Creates `n` samples from the distribution.
        :param n: number of samples.
        :return: tuple `(samples, indices)`, where `samples` is a tensor of shape `(n, d)`, and `indices` a tensor of shape `(n,)`
        corresponding to the indices of the components the samples are drawn from.
        """
        if not self.exact_lambdas:
            indices = torch.multinomial(self.lambdas, n, replacement=True)
        else:
            perm = torch.randperm(n)
            lambda_sum = sum(self.lambdas)
            frequencies = [int(n * lamb / lambda_sum) for lamb in self.lambdas]
            indices = torch.tensor(
                [i for i in range(self.l) for _ in range(frequencies[i])]
            )
            assert (
                len(indices) == n
            ), "cannot produce samples with exact lambdas (number of samples not divisible by sum of lambdas)"
            indices = indices[perm]
        sorted_indices, perm = torch.sort(indices)
        samples = torch.cat(
            (
                [
                    self.dists[i].sample((torch.sum(torch.eq(indices, i)).item(),))
                    for i in range(self.l)
                ]
            ),
            dim=0,
        )
        samples = samples[perm]
        indices = sorted_indices[perm]
        return samples, indices

    def plot_samples(self, samples=None):
        """
        Visualizes `samples`. For this, `self.d` needs to be 2.
        :param samples: Samples to visualize. Default to `self.train_samples`.
        :return: None.
        """
        if samples == None:
            samples = self.train_samples
        samples, components = samples
        n_samples = len(components)
        plt.figure()
        for comp in torch.unique(components):
            mask = components == comp
            plt.scatter(
                samples[mask, 0], samples[mask, 1], label=f"Component {comp.item()}"
            )
        plt.title(f"Plot of {n_samples} samples from the GMM")
        plt.legend()
        plt.show()


def embed_labels(
    label_dim,
    labels_1,
    label_offset_1=0,
    labels_2=None,
    label_offset_2=None,
    t=0,
    one_hot_matrix=None,
):
    """
    Embeds `labels_1` into `label_dim`-dimensional space. Can also construct a convex combination
    of two sets of labels, if `labels_2` and `t` are passed.
    :param label_dim: label dimension.
    :param labels_1: labels to embed. Tensor of shape (n,).
    :param label_offset_1: embeds `labels_1` into dimensions starting at `label_from` (i.e. shifts every entry).
    :param labels_2: if given, constructs a convex combination of `labels_1` and `labels_2`.
    :param label_offset_2: similar to `label_offset_1`, but for `labels_2`. Defaults to 1+max(`labels_1`).
    :param t: scalar used for convex combination (1-`t`)*`labels_1+`t`*`labels_2`.
    :param one_hot_matrix: one-hot-label encoding matrix, i.e. identity matrix of dimension `label_dim`. Can
        be passed for faster computation.
    :return: labels. Tensor of shape (n, `label_dim`).
    """
    n_labels = len(labels_1)
    if one_hot_matrix is None:
        one_hot_matrix = torch.eye(label_dim)
    if labels_2 is None:
        labels = one_hot_matrix[label_offset_1 + labels_1]
    else:
        assert len(labels_2) == n_labels
        if label_offset_2 is None:
            label_offset_2 = max(labels_1) + 1
        labels = (1 - t) * one_hot_matrix[
            label_offset_1 + labels_1
        ] + t * one_hot_matrix[label_offset_2 + labels_2]
    return labels


class InterpolGMMs:

    def __init__(
        self,
        nb_sets=2,
        nb_interpol=4,
        gmm_kwargs1=None,
        gmm_kwargs2=None,
        low_dim_labels=False,
        hard_labels=False,
    ):
        """
        Creates `nb_interpol` interpolated datasets in between `nb_sets` many GMMs. For now only
            supports GMMs, and only 2 in total. All datasets must have the same number of samples.
        :param nb_sets: number of datasets to use for interpolation. Currently only supports `nb_sets=2`.
        :param nb_interpol: number of datasets to be created. Includes the two initial datasets.
        :param gmm_kwargs1: optional kwargs for the first of the two datasets.
        :param gmm_kwargs2: optional kwargs for the second of the two datasets.
        :param low_dim_labels: if True, does not cast all labels to the same dimension, but keeps labels of the datasets initially created
            (i.e. the ones that are not interpolated) to their intrinsic dimension.
        :param hard_labels: if True, converts all soft labels to hard labels. This only works
            if the number of classes across all datasets is the same, and if the classes in the first and
            last dataset are aligned. Do NOT use if low_dim_labels == True.
        """
        self.datasets = []
        self.test_datasets = []
        self.gmms = []
        if gmm_kwargs1:
            self.gmms.append(GMM(**gmm_kwargs1))
        else:
            self.gmms.append(GMM())
        if gmm_kwargs2:
            self.gmms.append(GMM(**gmm_kwargs2))
        else:
            self.gmms.append(GMM())
        mu = torch.ones(self.gmms[0].train_N) / self.gmms[0].train_N
        nu = torch.ones(self.gmms[1].train_N) / self.gmms[1].train_N
        test_mu = torch.ones(self.gmms[0].test_N) / self.gmms[0].test_N
        test_nu = torch.ones(self.gmms[1].test_N) / self.gmms[1].test_N
        assert len(mu) == len(nu)
        assert len(test_mu) == len(test_nu)
        n_samples = self.gmms[0].train_N
        cost = (
            torch.cdist(self.gmms[0].train_samples[0], self.gmms[1].train_samples[0])
            ** 2
        )
        test_cost = (
            torch.cdist(self.gmms[0].test_samples[0], self.gmms[1].test_samples[0]) ** 2
        )
        T = ot.emd(mu, nu, cost)
        test_T = ot.emd(test_mu, test_nu, test_cost)
        self.plan = T
        self.label_dim = sum(gmm.l for gmm in self.gmms)
        l_counter = 0
        for gmm in self.gmms:
            # each entry corresponds to (samples, labels)
            self.datasets.append(
                CustomDataset(
                    gmm.train_samples[0],
                    embed_labels(self.label_dim, gmm.train_samples[1], l_counter),
                )
            )
            self.test_datasets.append(
                CustomDataset(
                    gmm.test_samples[0],
                    embed_labels(self.label_dim, gmm.test_samples[1], l_counter),
                )
            )
            l_counter += gmm.l
        non_zero_indices = torch.nonzero(T)
        non_zero_test = torch.nonzero(test_T)
        assert len(non_zero_indices) == len(mu)
        assert len(non_zero_test) == len(test_mu)
        row_indices, col_indices = non_zero_indices.unbind(1)
        rows_test, cols_test = non_zero_test.unbind(1)
        self.row_indices = row_indices
        assert all(row_indices == torch.tensor([i for i in range(len(mu))]))
        assert all(rows_test == torch.tensor([i for i in range(len(test_mu))]))
        self.forward_indices = col_indices  # mapping indices of T
        self.inv_indices = col_indices.sort()[1]  # mapping indices of T inverse

        x1_features = self.gmms[0].train_samples[0][row_indices]
        x2_features = self.gmms[1].train_samples[0][col_indices]
        x1_labels = self.gmms[0].train_samples[1][row_indices]
        x2_labels = self.gmms[1].train_samples[1][col_indices]
        label_matrix = torch.eye(self.gmms[0].l + self.gmms[1].l)

        x1_features_test = self.gmms[0].test_samples[0][rows_test]
        x2_features_test = self.gmms[1].test_samples[0][cols_test]
        x1_labels_test = self.gmms[0].test_samples[1][rows_test]
        x2_labels_test = self.gmms[1].test_samples[1][cols_test]

        for i in range(nb_interpol - 2):
            t = (i + 1) / (nb_interpol - 1)
            samples = (1 - t) * x1_features + t * x2_features
            test_samples = (1 - t) * x1_features_test + t * x2_features_test
            assert len(samples) == n_samples
            labels = embed_labels(
                self.label_dim,
                x1_labels,
                labels_2=x2_labels,
                t=t,
                one_hot_matrix=label_matrix,
            )
            self.datasets.insert(
                -1, CustomDataset(samples, labels, low_dim_labels=False)
            )
            test_labels = embed_labels(
                self.label_dim,
                x1_labels_test,
                labels_2=x2_labels_test,
                t=t,
                one_hot_matrix=label_matrix,
            )
            self.test_datasets.insert(
                -1, CustomDataset(test_samples, test_labels, low_dim_labels=False)
            )

        if low_dim_labels:
            for i in [0, -1]:
                self.datasets[i].high_dim_labels = self.datasets[i].labels
                self.datasets[i].labels = embed_labels(
                    self.gmms[i].l,
                    self.gmms[i].train_samples[1],
                )
                self.test_datasets[i].high_dim_labels = self.test_datasets[i].labels
                self.test_datasets[i].labels = embed_labels(
                    self.gmms[i].l,
                    self.gmms[i].test_samples[1],
                )

        if hard_labels:
            assert (
                low_dim_labels == False
            ), "can't use hard_labels alongside soft_dim_labels!"
            soft_label_dim = self.datasets[0].label_dim
            for datasets in self.datasets, self.test_datasets:
                for dataset in datasets:
                    assert (
                        dataset.num_unique_labels == soft_label_dim // 2
                    ), "number of labels incorrect"
                    compressed_labels = (
                        dataset.labels[:, : soft_label_dim // 2]
                        + dataset._labels[:, soft_label_dim // 2 :]
                    )

                    # in this case, the labels of the interpolated datasets can be chosen to align with the labels of the
                    # outer datasets, because the outer datasets align
                    if len(compressed_labels) == soft_label_dim and torch.all(
                        torch.unique(compressed_labels) == torch.tensor([0.0, 1.0])
                    ):
                        dataset.labels = compressed_labels

                    # otherwise, give randomly allocated hard labels
                    else:
                        one_hots = torch.eye(soft_label_dim // 2)[
                            torch.randperm(soft_label_dim // 2)
                        ]
                        _, indices = torch.unique(
                            dataset.labels, dim=0, return_inverse=True
                        )
                        dataset.labels = one_hots[indices]

            self.label_dim = soft_label_dim // 2

    def plot_datasets(self, datasets=None):
        """
        Creates plots for all datasets.
        :param datasets: if None, defaults to the train datasets. The test datasets
            can be passed to plot those instead.
        :return: None.
        """
        if datasets is None:
            datasets = self.datasets
        datasets = [
            (
                (np.array(dataset.features), np.array(dataset.high_dim_labels))
                if dataset.high_dim_labels is not None
                else (np.array(dataset.features), np.array(dataset.labels))
            )
            for dataset in datasets
        ]
        k = len(self.datasets)
        d_labels = self.label_dim
        cmap = plt.get_cmap("inferno")
        keys = np.eye(d_labels).tolist()
        keys = [tuple(item) for item in keys]
        unit_vector_colors = dict(zip(keys, cmap(np.linspace(0, 1, d_labels))))

        # Function to interpolate colors based on convex combination
        def interpolate_color(combination):
            result_color = None
            for unit_vector, color in unit_vector_colors.items():
                if result_color is None:
                    result_color = np.dot(np.array(unit_vector), combination) * color
                else:
                    result_color += np.dot(np.array(unit_vector), combination) * color
            result_color = np.clip(
                result_color, None, 1
            )  # clip to 1, because some rounding error might occur and cause errors
            return result_color

        # Create subplots for each dataset in a square-shaped figure
        nrows = int(np.sqrt(k))
        ncols = int(np.ceil(k / nrows))
        fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))

        # Flatten axs if it's a 1D array (for the case when k=1)
        axs = axs.flatten() if k > 1 else [axs]

        for i, (features, labels) in enumerate(datasets):
            ax = axs[i]

            # Convert labels to RGB colors
            colors = [interpolate_color(label) for label in labels]

            # Scatter plot of features with colors defined by labels
            ax.scatter(features[:, 0], features[:, 1], c=colors, marker="o", alpha=0.8)

            ax.set_title(f"Dataset {i + 1}")
            ax.set_xlabel("Feature 1")
            ax.set_ylabel("Feature 2")

        # Remove any empty subplots
        for j in range(i + 1, len(axs)):
            axs[j].axis("off")

        plt.tight_layout()
        plt.show()
