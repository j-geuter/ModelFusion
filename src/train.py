import torch
import torch.nn as nn
import torch.optim as optim

from models import *


def train(
    model,
    dataset,
    num_epochs=20,
    batch_size=128,
    loss_fn=nn.CrossEntropyLoss(),
    lr=0.1,
    opt=optim.SGD,
):
    """
    Train a `model` on a `dataset`.
    :param model: model, nn.Module type.
    :param dataset: dataset of type `CustomDataset`.
    :param num_epochs: number of epochs.
    :param batch_size: batch size.
    :param loss_fn: loss function.
    :return: train error.
    """
    model.train()
    optimizer = opt(model.parameters(), lr=lr, weight_decay=1e-3)
    batches = batch_generator(dataset, num_epochs, batch_size)

    for i, (batch_x, batch_y) in enumerate(batches):

        outputs = model(batch_x)
        loss = loss_fn(outputs, batch_y)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print training statistics
        # if i % 10 == 0:
        # print(f'Iteration {i}, Loss: {loss.item():.4f}')
    accuracy = get_accuracy(model, dataset)
    print(f"Final loss: {loss}; accuracy: {accuracy}")


def test(model, dataset):
    """
    Returns average error of `model` on `dataset`.
    """
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    outputs = model(dataset.features)
    loss = loss_fn(outputs, dataset.labels)
    return loss.item()


def get_accuracy(model, dataset):
    """
    Compute the accuracy of a `model` on a `dataset`. For each datapoint, output the closest label out of all
    labels present in the dataset. Return a value between 0 and 1.
    :param model: model, nn.Module type.
    :param dataset: 2-tuple of Tensors, where the first is `(N,d)` shaped and corresponds to features,
        and the second is `(N,l)` shaped and corresponds to labels.
    :return: accuracy as a value between 0 and 1.
    """
    model.eval()

    labels = dataset.unique_labels  # all labels present in the dataset
    outputs = model(dataset.features)
    true_labels = get_labels(dataset.labels, labels)
    predicted_labels = get_labels(outputs, labels)
    correct_preds = true_labels == predicted_labels
    accuracy = correct_preds.float().mean()
    return round(accuracy.item(), 2)


def get_labels(preds, labels):
    dists = torch.cdist(preds, labels)
    closest = torch.argmin(dists, dim=1)
    return closest


def batch_generator(dataset, num_epochs, batch_size):

    x_samples, labels = dataset.features, dataset.labels
    num_samples = len(x_samples)

    for epoch in range(num_epochs):
        # Shuffle indices at the beginning of each epoch
        indices = torch.randperm(num_samples).tolist()

        # Iterate over batches
        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i : i + batch_size]

            # Extract batch data
            batch_x = x_samples[batch_indices]
            batch_labels = labels[batch_indices]

            yield batch_x, batch_labels
