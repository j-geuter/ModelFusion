import torch
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import ot
from sinkhorn import sinkhorn

from synthdatasets import CustomDataset
from models import SimpleCNN, TransportNN
from train_mnist import test_accuracy

TRAIN_DATASET = "mnist"  # `mnist`, `usps`, or `fashion`. The dataset on which a trained model is given
TEST_DATASET = "fashion" # `mnist`, `usps`, or `fashion` The dataset on which we want to test
BATCH_SIZE = 64
NUM_SAMPLES = 7000  # number of training samples used in the transport plan
NUM_TEST_SAMPLES = 2500 # number of test samples

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

# same as above, but additionally resizing to 28*28
resize_transform = transforms.Compose(
    [
        transforms.ToTensor(),  # Convert images to tensors
        transforms.Normalize(
            (0.5,), (0.5,)
        ),  # Normalize pixel values to the range [-1, 1]
        transforms.Resize((28, 28))
    ]
)

if TRAIN_DATASET == "mnist":
    train_source_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
elif TRAIN_DATASET == "fashion":
    train_source_dataset = datasets.FashionMNIST(
        root="./data", train=True, download=True, transform=transform
    )
elif TRAIN_DATASET == "usps":
    train_source_dataset = datasets.USPS(
    root="./data", train=True, download=True, transform=resize_transform
)
else:
    raise ValueError("`TRAIN_DATASET` must be one of `mnist`, `fashion`, or `usps`.")

if TEST_DATASET == "mnist":
    train_target_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
elif TEST_DATASET == "fashion":
    train_target_dataset = datasets.FashionMNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.FashionMNIST(
        root="./data", train=False, download=True, transform=transform
    )
elif TEST_DATASET == "usps":
    train_target_dataset = datasets.USPS(
        root="./data", train=True, download=True, transform=resize_transform
    )
    test_dataset = datasets.USPS(
        root="./data", train=False, download=True, transform=resize_transform
    )
else:
    raise ValueError("`TEST_DATASET` must be one of `mnist`, `fashion`, or `usps`.")

train_source_features = torch.cat([train_source_dataset[i][0] for i in range(NUM_SAMPLES)], dim=0)
train_source_labels = torch.tensor([train_source_dataset[i][1] for i in range(NUM_SAMPLES)])
train_target_features = torch.cat([train_target_dataset[i][0] for i in range(NUM_SAMPLES)], dim=0)
train_target_labels = torch.tensor([train_target_dataset[i][1] for i in range(NUM_SAMPLES)])
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

train_source_dataset = CustomDataset(train_source_features, train_source_labels)
train_target_dataset = CustomDataset(train_target_features, train_target_labels)
location = f"./models/{TRAIN_DATASET}.pth"

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

cost /= cost.max()
T = sinkhorn(mu, nu, cost, 0.006, 10000, normThr=1e-7, show_progress_bar=True, tens_type=torch.float64)[1].to(torch.float32)

aligned_features = torch.einsum(
    'nl,lxy->nxy',
    T,
    train_datasets[1].features
)

aligned_labels = torch.matmul(T, train_datasets[1].high_dim_labels)
train_datasets[1] = CustomDataset(aligned_features, aligned_labels, low_dim_labels=False)


transportNN = TransportNN([model], [train_datasets[0]], train_datasets[1])

# train_acc = test_accuracy(transportNN, train_loader, max_samples=2000)
# print(f"TransportNN accuracy on train set: {train_acc}")
test_acc = test_accuracy(transportNN, test_loader_star, max_samples=NUM_TEST_SAMPLES)
print(f"TransportNN accuracy on test set: {test_acc}")
