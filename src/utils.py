import torch
import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np


def plot_images(images):
    # Assuming images is an array of shape (n, 1, d, d)
    if images.dim() == 4:
        n, _, d, _ = images.shape
    # Assuming image is an array of shape (n, d^2)
    elif images.dim() == 2:
        n, dd = images.shape
        d = int(np.sqrt(dd))
        images = images.reshape(n, d, d)
        images = images.unsqueeze(1)
    # Assuming image is an array of shape (n, d, d)
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
        ax.axis("off")  # Turn off axis labels
        ax.imshow(images[i, 0], cmap="gray")  # Assuming images are grayscale

    # Hide any remaining empty subplots
    for i in range(n, grid_size * grid_size):
        axes[i].axis("off")

    plt.show()


def class_correspondences(dataset_1, dataset_2, plan, symmetric=False, plot=True):
    correspondences = torch.zeros(
        (dataset_1.num_unique_labels, dataset_2.num_unique_labels)
    )
    indices_1 = [dataset_1.get_samples_by_label(l)[1] for l in dataset_1.unique_labels]
    indices_2 = [dataset_2.get_samples_by_label(l)[1] for l in dataset_2.unique_labels]
    for i, idx1 in enumerate(indices_1):
        for j, idx2 in enumerate(indices_2):
            correspondence = plan[idx1, :][:, idx2]
            correspondences[i][j] = correspondence.sum().item()
    correspondences /= correspondences.sum(dim=1)[:, None]
    if symmetric:
        correspondences += class_correspondences(
            dataset_2, dataset_1, plan.T, symmetric=False, plot=False
        )
        correspondences /= 2
    if plot:
        plt.imshow(correspondences)
        plt.xlabel("Classes of dataset 2")
        plt.ylabel("Classes of dataset 1")
        plt.title("Class correspondences according to transport plan")
        plt.show()
    return correspondences


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
