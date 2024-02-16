import torch
from torchvision import datasets, transforms
from synthdatasets import CustomDataset

# Define a data transform to normalize the data
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to tensors
    transforms.Normalize((0.5,), (0.5,))  # Normalize pixel values to the range [-1, 1]
])

# Download MNIST dataset
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Download FashionMNIST dataset
fashion_train = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
fashion_test = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# Convert MNIST to CustomDataset
mnist_train_features = torch.cat([sample[0] for sample in mnist_train], dim=0)
mnist_train_labels = torch.tensor([sample[1] for sample in mnist_train])
mnist_test_features = torch.cat([sample[0] for sample in mnist_test], dim=0)
mnist_test_labels = torch.tensor([sample[1] for sample in mnist_test])
mnist_train_dataset = CustomDataset(mnist_train_features, mnist_train_labels)

# Convert FashionMNIST to CustomDataset
fashion_train_features = torch.cat([sample[0] for sample in fashion_train], dim=0)
fashion_train_labels = torch.tensor([sample[1] for sample in fashion_train])
fashion_test_features = torch.cat([sample[0] for sample in fashion_test], dim=0)
fashion_test_labels = torch.tensor([sample[1] for sample in fashion_test])
fashion_train_dataset = CustomDataset(fashion_train_features, fashion_train_labels)



