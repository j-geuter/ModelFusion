import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from models import SimpleCNN


# Training loop
def train(model, train_loader, test_loader, num_epochs=2):
    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
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


def test_accuracy(model, data_loader):
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for images, labels in data_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = total_correct / total_samples
    return accuracy


if __name__ == "__main__":
    torch.manual_seed(42)

    # Define a data transform to normalize the data
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # Convert images to tensors
            transforms.Normalize(
                (0.5,), (0.5,)
            ),  # Normalize pixel values to the range [-1, 1]
        ]
    )

    # Download MNIST dataset
    mnist_train = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    mnist_test = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    # Download FashionMNIST dataset
    fashion_train = datasets.FashionMNIST(
        root="./data", train=True, download=True, transform=transform
    )
    fashion_test = datasets.FashionMNIST(
        root="./data", train=False, download=True, transform=transform
    )

    BATCH_SIZE = 64
    TRAIN_MODEL = "fashion"
    if TRAIN_MODEL == "mnist":
        train_loader = DataLoader(mnist_train, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(mnist_test, batch_size=BATCH_SIZE, shuffle=False)

    elif TRAIN_MODEL == "fashion":
        train_loader = DataLoader(fashion_train, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(fashion_test, batch_size=BATCH_SIZE, shuffle=False)

    else:
        raise ValueError("TRAIN_MODEL needs to be either `mnist` or `fashion`.")

    # Initialize the model, loss function, and optimizer
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train(model, train_loader, test_loader)
