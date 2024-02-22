import torch
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import ot

from synthdatasets import CustomDataset
from models import SimpleCNN, TransportNN
from train_mnist import test_accuracy

TRAIN_DATASET = "mnist"  # this is the dataset on which a model is trained; either `mnist` or `fashion`
BATCH_SIZE = 64
NUM_SAMPLES = 100  # number of training samples used in the transport plan
assert NUM_SAMPLES <= 60000

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

# Convert MNIST to CustomDataset
mnist_train_features = torch.cat([mnist_train[i][0] for i in range(NUM_SAMPLES)], dim=0)
mnist_train_labels = torch.tensor([mnist_train[i][1] for i in range(NUM_SAMPLES)])
mnist_test_features = torch.cat([mnist_test[i][0] for i in range(NUM_SAMPLES)], dim=0)
mnist_test_labels = torch.tensor([mnist_test[i][1] for i in range(NUM_SAMPLES)])
mnist_train_dataset = CustomDataset(mnist_train_features, mnist_train_labels)
mnist_test_dataset = CustomDataset(mnist_test_features, mnist_test_labels)

# Convert FashionMNIST to CustomDataset
fashion_train_features = torch.cat(
    [fashion_train[i][0] for i in range(NUM_SAMPLES)], dim=0
)
fashion_train_labels = torch.tensor([fashion_train[i][1] for i in range(NUM_SAMPLES)])
fashion_test_features = torch.cat(
    [fashion_test[i][0] for i in range(NUM_SAMPLES)], dim=0
)
fashion_test_labels = torch.tensor([fashion_test[i][1] for i in range(NUM_SAMPLES)])
fashion_train_dataset = CustomDataset(fashion_train_features, fashion_train_labels)
fashion_test_dataset = CustomDataset(fashion_test_features, fashion_test_labels)

if TRAIN_DATASET == "mnist":
    train_datasets = [mnist_train_dataset, fashion_train_dataset]
    test_datasets = [mnist_test_dataset, fashion_test_dataset]
    location = "./models/mnist.pth"

    # creating a train_loader as well but just for testing purposes
    train_loader = DataLoader(mnist_train, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(mnist_test, batch_size=BATCH_SIZE, shuffle=False)

elif TRAIN_DATASET == "fashion":
    train_datasets = [fashion_train_dataset, mnist_train_dataset]
    test_datasets = [fashion_test_dataset, mnist_test_dataset]
    location = "./models/fashion.pth"

    # creating a train_loader as well but just for testing purposes
    train_loader = DataLoader(fashion_train, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(fashion_test, batch_size=BATCH_SIZE, shuffle=False)
else:
    raise ValueError("TRAIN_DATASET must be either `mnist` or `fashion`.")

model = SimpleCNN(location=location)
acc = test_accuracy(model, test_loader)
print(f"{TRAIN_DATASET} model accuracy on test set: {acc}")

mu = torch.ones(train_datasets[0].num_samples) / train_datasets[0].num_samples
nu = torch.ones(train_datasets[1].num_samples) / train_datasets[1].num_samples
cost = (
    torch.cdist(
        train_datasets[0].features.view(train_datasets[0].features.shape[0], -1),
        train_datasets[1].features.view(train_datasets[1].features.shape[0], -1),
    )
    ** 2
)

T = ot.emd(mu, nu, cost, numItermax=1000000)
nonzero_indices = torch.nonzero(T)
assert len(nonzero_indices) == len(mu)
rows, permutation = nonzero_indices.unbind(1)
assert all(rows == torch.tensor([i for i in range(len(mu))]))
train_datasets[1].permute_data(permutation)

transportNN = TransportNN([model], [train_datasets[0]], train_datasets[1])
train_acc = test_accuracy(transportNN, train_loader)
print(f"TransportNN accuracy on train set: {train_acc}")
test_acc = test_accuracy(transportNN, test_loader)
print(f"TransportNN accuracy on test set: {test_acc}")
