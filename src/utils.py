import torch
import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np

def plot_images(images):
    # Assuming images is an array of shape (n, 1, d, d)
    if images.dim() == 4:
        n, _, d, _ = images.shape
    elif images.dim() == 2:
        n, dd = images.shape
        d = int(np.sqrt(dd))
        images = images.reshape(n, d, d)
        images = images.unsqueeze(1)
    elif images.dim() == 3:
        n, d, _ = images.shape
        images = images.unsqueeze(1)
    else:
        raise TypeError("`Images` of wrong shape.")

    # Determine the size of the grid
    grid_size = int(np.ceil(np.sqrt(n)))

    # Create a blank canvas for the grid
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))

    # Flatten the axes if the grid is not square
    axes = axes.flatten()

    # Plot each image in the grid
    for i in range(n):
        ax = axes[i]
        ax.axis('off')  # Turn off axis labels
        ax.imshow(images[i, 0], cmap='gray')  # Assuming images are grayscale

    # Hide any remaining empty subplots
    for i in range(n, grid_size * grid_size):
        axes[i].axis('off')

    plt.show()

def class_correspondences(dataset_1, dataset_2, plan=None, symmetric=False, plot=False):
    """
    Compute class correspondences of a transport plan `plan` between
        `dataset_1` and `dataset_2`. This returns a matrix of size
        #classes(dataset_1) * #classes(dataset_2), where each entry
        corresponds to what percentage of that class from `dataset_1`
        is mapped to the respective class in `dataset_2` by the plan.
        If `symmetric` is True, instead computes the average value across
        both directions (from `dataset_1` to `dataset_2` and vice versa).
    :param dataset_1: First dataset.
    :param dataset_2: Second dataset.
    :param plan: Transport plan. If none, assumes that the datasets are already
        aligned according to the plan.
    :param symmetric: If True, symmetrizes the values.
    :param plot: If True, plots the matrix with a heat map.
    :return: Matrix containing class correspondences.
    """
    nb_samples_1, nb_samples_2 = dataset_1.num_samples, dataset_2.num_samples
    assert (
        nb_samples_1 == nb_samples_2
    ), "For now only implemented for equal number of samples"
    if plan is not None:
        nonzero_indices = torch.nonzero(plan)
        rows, permutation = nonzero_indices.unbind(1)
        aligned_dataset_2 = deepcopy(dataset_2)
        aligned_dataset_2.permute_data(permutation)
    else:
        aligned_dataset_2 = dataset_2
    indices = (
        dataset_1.label_indices * aligned_dataset_2.num_unique_labels
        + aligned_dataset_2.label_indices
    )
    occurrences = torch.bincount(
        indices,
        minlength=dataset_1.num_unique_labels * aligned_dataset_2.num_unique_labels,
    )
    occurrences = occurrences.reshape(
        dataset_1.num_unique_labels, aligned_dataset_2.num_unique_labels
    )
    normalize = occurrences.sum(dim=1).unsqueeze(1)
    normalize[normalize == 0] = 1  # avoid division by zero in the normalizing step
    occurrences = occurrences / normalize
    if symmetric:
        occurrences = (
            occurrences
            + class_correspondences(
                aligned_dataset_2, dataset_1, plan=None, symmetric=False, plot=False
            )
        ) / 2
    if plot:
        plt.imshow(occurrences.numpy(), cmap="viridis", interpolation="nearest")
        plt.xlabel("Classes of dataset 2")
        plt.ylabel("Classes of dataset 1")
        plt.title("Class correspondences according to transport plan")
        plt.show()
    return occurrences


def models_equal(model1, model2, tolerance=1e-5):
    """
    Checks if two models `model1` and `model2` are identical, up to tolerance `tolerance`.
    :return: True if they are identical, False otherwise.
    """
    # Check if the models have the same architecture (same layers and shapes)
    assert model1.__class__ == model2.__class__, "Model classes do not match"

    # Check if the models have the same number of parameters
    assert sum(p.numel() for p in model1.parameters()) == sum(
        p.numel() for p in model2.parameters()
    ), "Parameter count does not match"

    # Check if the parameters are identical up to the tolerance
    for param1, param2 in zip(model1.parameters(), model2.parameters()):
        assert param1.shape == param2.shape, "Parameter shape does not match"

        if torch.any(torch.abs(param1 - param2) > tolerance):
            return False

    return True


def weights_equal(weights_1, weights_2, tol=1e-5):
    """
    Checks if two weight vectors are identical, up to tolerance `tol`.
    :return: True if they are identical, False otherwise.
    """
    assert weights_1.shape == weights_2.shape, "Parameter shape does not match"

    if torch.any(torch.abs(weights_1 - weights_2) > tol):
        return False
    return True
