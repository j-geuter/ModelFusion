import torch
import torch.nn as nn
import torch.optim as optim

from models import *


def train(
    model,
    dataset,
    num_epochs=1,
    batch_size=128,
    loss_fn=nn.MSELoss(),
    lr=0.001,
    opt=optim.Adam,
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


def get_accuracy(model, dataset, project_labels=True):
    """
    Compute the accuracy of a `model` on a `dataset`. For each datapoint, output the closest label out of all
    labels present in the dataset. Return a value between 0 and 1.
    :param model: model, nn.Module type.
    :param dataset: 2-tuple of Tensors, where the first is `(N,d)` shaped and corresponds to features,
        and the second is `(N,l)` shaped and corresponds to labels.
    :param project_labels: if True, projects model outputs to the nearest label. Otherwise,
        uses model output directly as label.
    :return: accuracy as a value between 0 and 1.
    """
    model.eval()

    labels = dataset.labels  # all labels present in the dataset
    predicted_labels = model(dataset.features)
    # true_labels = get_labels(dataset.labels, labels)
    if project_labels:
        labels = get_labels(labels, dataset.unique_labels)
        predicted_labels = get_labels(predicted_labels, dataset.unique_labels)
    correct_preds = labels == predicted_labels
    accuracy = correct_preds.float().mean()
    return round(accuracy.item(), 2)


def get_labels(preds, labels):
    if preds.dim() == 1:
        preds = preds.unsqueeze(1)
        print('Warning! You probably want to pass `project_labels`=False to `get_accuracy`.')
    dists = torch.cdist(preds.to(torch.float32), labels.to(torch.float32))
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
