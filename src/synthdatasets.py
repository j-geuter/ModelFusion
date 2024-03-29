import torch
import torch.distributions as D
from torch.distributions import Categorical, MixtureSameFamily, MultivariateNormal
import matplotlib.pyplot as plt
import ot
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GMM:

    def __init__(self, d=2, N=100, l=2, lambdas=None, mus=None, Sigmas=None):
        """
        Class to create Gaussian mixture models. Creates `N` training samples.
        :param d: dimension.
        :param N: number of training samples.
        :param l: number of components.
        :param lambdas: component coefficients.
        :param mus: component means. Tensor of dimension `(l, d)`, defaults to random means.
        :param Sigmas: component covariances. Tensor of dimension `(l, d, d)`, defaults to identity covariance for each component.
        """
        self.d = d
        self.N = N
        self.l = l
        if lambdas == None:
            lambdas = torch.ones(l)
        lambdas /= sum(lambdas)
        self.lambdas = lambdas.to(device)
        if mus == None:
            mus = torch.randn(l, d)
        if Sigmas == None:
            Sigmas = torch.eye(d).unsqueeze(0).expand(l, -1, -1)
        self.dists = [MultivariateNormal(mus[i].to(device), Sigmas[i].to(device)) for i in range(l)]
        self.train_samples = self.sample(N)

    def sample(self, n):
        """
        Creates `n` samples from the distribution.
        :param n: number of samples.
        :return: tuple `(samples, indices)`, where `samples` is a tensor of shape `(n, d)`, and `indices` a tensor of shape `(n,)`
        corresponding to the indices of the components the samples are drawn from.
        """
        indices = torch.multinomial(self.lambdas, n, replacement=True)
        sorted_indices, perm = torch.sort(indices)
        samples = torch.cat(([self.dists[i].sample((torch.sum(torch.eq(indices, i)).item(),)) for i in range(self.l)]),
                            dim=0)
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
            plt.scatter(samples[mask, 0], samples[mask, 1], label=f'Component {comp.item()}')
        plt.title(f'Plot of {n_samples} samples from the GMM')
        plt.legend()
        plt.show()


def embed_labels(
        label_dim,
        labels_1,
        label_offset_1=0,
        labels_2=None,
        label_offset_2=None,
        t=0,
        one_hot_matrix=None
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
        labels = ((1 - t) * one_hot_matrix[label_offset_1 + labels_1]
                      + t * one_hot_matrix[label_offset_2 + labels_2])
    return labels


class InterpolGMMs:

    def __init__(self, nb_sets=2, nb_interpol=4, gmm_kwargs1=None, gmm_kwargs2=None):
        """
        Creates `nb_interpol` interpolated datasets in between `nb_sets` many GMMs. For now only
            supports GMMs, and only 2 in total. All datasets must have the same number of samples.
        :param nb_sets: number of datasets to use for interpolation. Currently only supports `nb_sets=2`.
        :param nb_interpol: number of datasets to be created. Includes the two initial datasets.
        :param gmm_kwargs1: optional kwargs for the first of the two datasets.
        :param gmm_kwargs2: optional kwargs for the second of the two datasets.
        """
        self.datasets = []
        self.gmms = []
        if gmm_kwargs1:
            self.gmms.append(GMM(**gmm_kwargs1))
        else:
            self.gmms.append(GMM())
        if gmm_kwargs2:
            self.gmms.append(GMM(**gmm_kwargs2))
        else:
            self.gmms.append(GMM())
        mu = torch.ones(self.gmms[0].N) / self.gmms[0].N
        nu = torch.ones(self.gmms[1].N) / self.gmms[1].N
        assert len(mu) == len(nu)
        n_samples = self.gmms[0].N
        cost = torch.cdist(self.gmms[0].train_samples[0], self.gmms[1].train_samples[0])
        T = ot.emd(mu, nu, cost)
        self.label_dim = sum(gmm.l for gmm in self.gmms)
        l_counter = 0
        for gmm in self.gmms:
            # each entry corresponds to (samples, labels)
            self.datasets.append(
                (
                    gmm.train_samples[0],
                    embed_labels(
                        self.label_dim,
                        gmm.train_samples[1],
                        l_counter
                    )
                )
            )
            l_counter += gmm.l
        non_zero_indices = torch.nonzero(T)
        assert len(non_zero_indices) == len(mu)
        row_indices, col_indices = non_zero_indices.unbind(1)
        x1_features = self.gmms[0].train_samples[0][row_indices]
        x2_features = self.gmms[1].train_samples[0][col_indices]
        x1_labels = self.gmms[0].train_samples[1][row_indices]
        x2_labels = self.gmms[1].train_samples[1][col_indices]
        label_matrix = torch.eye(self.gmms[0].l + self.gmms[1].l)
        for i in range(nb_interpol - 2):
            t = (i + 1) / (nb_interpol - 1)
            samples = (1 - t) * x1_features + t * x2_features
            assert len(samples) == n_samples
            labels = embed_labels(
                self.label_dim,
                x1_labels,
                labels_2=x2_labels,
                t=t,
                one_hot_matrix=label_matrix
            )
            self.datasets.insert(-1, (samples, labels))

    def plot_datasets(self):
        """
        Creates plots for all datasets.
        :return: None.
        """
        datasets = [(np.array(dataset[0]), np.array(dataset[1])) for dataset in self.datasets]
        k = len(self.datasets)
        d_labels = self.label_dim
        cmap = plt.get_cmap('inferno')
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
            result_color = np.clip(result_color, None, 1) # clip to 1, because some rounding error might occur and cause errors
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
            ax.scatter(features[:, 0], features[:, 1], c=colors, marker='o', alpha=0.8)

            ax.set_title(f'Dataset {i + 1}')
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')

        # Remove any empty subplots
        for j in range(i + 1, len(axs)):
            axs[j].axis('off')

        plt.tight_layout()
        plt.show()