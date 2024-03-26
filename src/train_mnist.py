import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, train_loader, test_loader, num_epochs=2):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation
        accuracy = test_accuracy(model, test_loader)
        print(
            f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}"
        )

    print("Training finished.")


def test_accuracy(model, data_loader, max_samples=torch.inf):
    model.eval()
    total_correct = 0
    total_samples = 0
    if max_samples < torch.inf:
        total_iters = min(len(data_loader), max_samples // data_loader.batch_size)
    else:
        total_iters = len(data_loader)
    with torch.no_grad():
        for images, labels in tqdm(data_loader, total=total_iters):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            if total_samples >= max_samples:
                break

    accuracy = total_correct / total_samples
    return accuracy


def load_data(dataset, resize=False, batch_size=64):
    """
    Loads a dataset and returns the train and test loaders.
    :param dataset: One of `mnist`, `fashion`, or `usps`.
    :param resize: If True, resizes all data to 28*28 sizes images.
    :param batch_size: batch size for the data loaders.
    :return: train and test loader objects.
    """
    if not resize:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),  # Convert images to tensors
                transforms.Normalize(
                    (0.5,), (0.5,)
                ),  # Normalize pixel values to the range [-1, 1]
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),  # Convert images to tensors
                transforms.Normalize(
                    (0.5,), (0.5,)
                ),  # Normalize pixel values to the range [-1, 1]
                transforms.Resize((28, 28)),
            ]
        )
    if dataset == "mnist":
        train_dataset = datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )
        test_dataset = datasets.MNIST(
            root="./data", train=False, download=True, transform=transform
        )
    elif dataset == "fashion":
        train_dataset = datasets.FashionMNIST(
            root="./data", train=True, download=True, transform=transform
        )
        test_dataset = datasets.FashionMNIST(
            root="./data", train=False, download=True, transform=transform
        )
    elif dataset == "usps" or "usps28" or "usps16":
        train_dataset = datasets.USPS(
            root="./data", train=True, download=True, transform=transform
        )
        test_dataset = datasets.USPS(
            root="./data", train=False, download=True, transform=transform
        )
    else:
        raise ValueError("`dataset` must be one of `mnist`, `fashion`, or `usps`.")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
