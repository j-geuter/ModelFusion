import torch
import torch.optim as optim
import torch.nn as nn
from models import *

def train(
        model,
        dataset,
        num_epochs = 20,
        batch_size = 128,
        loss_fn = nn.CrossEntropyLoss(),
        lr = 0.1,
        opt = optim.SGD,
):
    """
    Train a `model` on a `dataset`.
    :param model: model, nn.Module type.
    :param dataset: 2-tuple of Tensors, where the first is `(N,d)` shaped and corresponds to features,
        and the second is `(N,l)` shaped and corresponds to labels.
    :param num_epochs: number of epochs.
    :param batch_size: batch size.
    :param loss_fn: loss function.
    :return: train error.
    """
    model.train()
    optimizer = opt(model.parameters(), lr=lr)
    batches = batch_generator(dataset, num_epochs, batch_size)

    for (i, (batch_x, batch_y)) in enumerate(batches):

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
    print(f'Final loss: {loss}; accuracy: {accuracy}')

def test(model, dataset):
    """
    Returns average error of `model` on `dataset`.
    """
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    outputs = model(dataset[0])
    loss = loss_fn(outputs, dataset[1])
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

    labels = torch.unique(dataset[1], dim=0) # all labels present in the dataset
    outputs = model(dataset[0])

    def get_labels(preds, labels):
        dists = torch.cdist(preds, labels)
        closest = torch.argmin(dists, dim=1)
        return closest

    true_labels = get_labels(dataset[1], labels)
    predicted_labels = get_labels(outputs, labels)
    correct_preds = true_labels == predicted_labels
    accuracy = correct_preds.float().mean()
    return accuracy




def batch_generator(dataset, num_epochs, batch_size):

    x_samples, labels = dataset[0], dataset[1]
    num_samples = len(x_samples)

    for epoch in range(num_epochs):
        # Shuffle indices at the beginning of each epoch
        indices = torch.randperm(num_samples).tolist()

        # Iterate over batches
        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:i + batch_size]

            # Extract batch data
            batch_x = x_samples[batch_indices]
            batch_labels = labels[batch_indices]

            yield batch_x, batch_labels