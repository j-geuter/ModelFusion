import torch
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import ot
from logger import logging

from sinkhorn import sinkhorn
from synthdatasets import CustomDataset
from models import SimpleCNN, TransportNN, compute_label_distances
from train_mnist import test_accuracy
from utils import class_correspondences

TRAIN_DATASET = "mnist"  # `mnist`, `usps`, or `fashion`. The dataset on which a trained model is given
TEST_DATASET = (
    "usps"  # `mnist`, `usps`, or `fashion` The dataset on which we want to test
)
BATCH_SIZE = 64
NUM_SAMPLES = 7000  # number of training samples used in the transport plan
NUM_TEST_SAMPLES = 2500  # number of test samples
RESIZE_USPS = True  # if True, resizes usps images to 28*28
REGULARIZER = None  # Regularizer for entropic OT problem. If set to None, computes unregularized plan
TEMPERATURE = 100 # temperature for the TransportNN plug-in estimations of OT maps
FEATURE_METHOD = "plain_softmax"  # one of plain_softmax, plugin
LABEL_METHOD = (
    "plain_softmax"  # one of plain_softmax, masked_softmax, plugin
)
FEATURE_DIST = True  # whether to use feature distances for label transport
LABEL_DIST = True  # whether to use label distances for label transport
OTDD_COST = (
    True  # if True, the cost between source_datasets takes label distances into account
)
LABEL_DIST_COEFF = 1 # controls how much the label distances contribute to the overall transport
                     # cost; 1 means both label and feature dists contribute equally
PROJECT_LABELS = (
    False  # if True, projects label predictions on source dataset space before transport
)

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
        transforms.Resize((28, 28)),
    ]
)

if RESIZE_USPS:
    usps_transform = resize_transform
else:
    usps_transform = transform

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
        root="./data", train=True, download=True, transform=usps_transform
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
        root="./data", train=True, download=True, transform=usps_transform
    )
    test_dataset = datasets.USPS(
        root="./data", train=False, download=True, transform=usps_transform
    )
    if (
        not RESIZE_USPS
    ):  # the resized dataset is still needed for computing the cost matrix
        resized_train_target_dataset = datasets.USPS(
            root="./data", train=True, download=True, transform=resize_transform
        )
        resized_train_target_features = torch.cat(
            [resized_train_target_dataset[i][0] for i in range(NUM_SAMPLES)], dim=0
        )
        resized_train_target_labels = torch.tensor(
            [resized_train_target_dataset[i][1] for i in range(NUM_SAMPLES)]
        )
        resized_train_target_dataset = CustomDataset(
            resized_train_target_features, resized_train_target_labels
        )


else:
    raise ValueError("`TEST_DATASET` must be one of `mnist`, `fashion`, or `usps`.")

train_source_features = torch.cat(
    [train_source_dataset[i][0] for i in range(NUM_SAMPLES)], dim=0
)
train_source_labels = torch.tensor(
    [train_source_dataset[i][1] for i in range(NUM_SAMPLES)]
)
train_target_features = torch.cat(
    [train_target_dataset[i][0] for i in range(NUM_SAMPLES)], dim=0
)
train_target_labels = torch.tensor(
    [train_target_dataset[i][1] for i in range(NUM_SAMPLES)]
)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

train_source_dataset = CustomDataset(train_source_features, train_source_labels)
train_target_dataset = CustomDataset(train_target_features, train_target_labels)
if TRAIN_DATASET == "usps":
    if RESIZE_USPS:
        location = f"./models/{TRAIN_DATASET}28.pth"
    else:
        location = f"./models/{TRAIN_DATASET}16.pth"
else:
    location = f"./models/{TRAIN_DATASET}.pth"

input_dim = 28 if RESIZE_USPS or TRAIN_DATASET != "usps" else 16
model = SimpleCNN(dim=input_dim, location=location)

try:
    acc = test_accuracy(model, test_loader)
    print(f"{TRAIN_DATASET} model accuracy on test set {TEST_DATASET}: {acc}")
except Exception as e:
    print(f"Cannot test model on target dataset; dimensions do not match. Error: {e}")

mu = torch.ones(train_source_dataset.num_samples) / train_source_dataset.num_samples
nu = torch.ones(train_target_dataset.num_samples) / train_target_dataset.num_samples
if TEST_DATASET == "usps" and not RESIZE_USPS:
    cost = (
        torch.cdist(
            train_source_dataset.features.view(
                train_source_dataset.features.shape[0], -1
            ),
            resized_train_target_features.view(
                resized_train_target_features.shape[0], -1
            ),
        )
        ** 2
    )
else:
    cost = (
        torch.cdist(
            train_source_dataset.features.view(
                train_source_dataset.features.shape[0], -1
            ),
            train_target_dataset.features.view(
                train_target_dataset.features.shape[0], -1
            ),
        )
        ** 2
    )
cost /= cost.max()

if OTDD_COST:
    #label_distances = compute_label_distances(
    #    train_source_dataset, train_target_dataset
    #)
    label_distances = torch.ones((10, 10)) - torch.eye(10)
    label_distances /= label_distances.max()
    label_distances = label_distances[train_source_dataset.labels.squeeze(), :][
        :, train_target_dataset.labels.squeeze()
    ]
    cost += LABEL_DIST_COEFF * label_distances
    cost /= cost.max()


if REGULARIZER is None:
    T = ot.emd(mu, nu, cost)
    f = None
    g = None
else:
    log = sinkhorn(
        mu,
        nu,
        cost,
        REGULARIZER,
        1000,
        normThr=1e-7,
        show_progress_bar=True,
        tens_type=torch.float64,
    )
    T = log["plan"].to(torch.float32)
    f = log["f"][0].to(torch.float32)
    g = log["g"][0].to(torch.float32)


transportNN = TransportNN(
    model,
    train_source_dataset,
    train_target_dataset,
    plans=T,
    feature_method=FEATURE_METHOD,
    label_method=LABEL_METHOD,
    feature_dists=FEATURE_DIST,
    label_dists=LABEL_DIST,
    project_source_labels=PROJECT_LABELS,
    temperature=TEMPERATURE,
    f=f,
    g=g,
    reg=REGULARIZER,
)

# train_acc = test_accuracy(transportNN, train_loader, max_samples=2000)
# print(f"TransportNN accuracy on train set: {train_acc}")
n_test_loader = len(test_loader) * BATCH_SIZE
if n_test_loader < NUM_TEST_SAMPLES:
    logging.warning(
        f"`NUM_TEST_SAMPLES`={NUM_TEST_SAMPLES}, but the DataLoader only has {n_test_loader} samples."
    )
test_acc = test_accuracy(transportNN, test_loader, max_samples=NUM_TEST_SAMPLES)
print(f"TransportNN accuracy on test set: {test_acc}")
