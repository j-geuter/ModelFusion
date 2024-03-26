import torch
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import ot
from logger import logging
import matplotlib.pyplot as plt

from sinkhorn import sinkhorn
from synthdatasets import CustomDataset
from models import SimpleCNN, TransportNN, compute_label_distances
from train_mnist import test_accuracy
from utils import class_correspondences, plot_images

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_DATASET = "mnist"  # `mnist`, `usps`, or `fashion`. The dataset on which a trained model is given
TEST_DATASET = (
    "fashion"  # `mnist`, `usps`, or `fashion` The dataset on which we want to test
)
BATCH_SIZE = 64
NUM_SAMPLES = 7000  # number of training samples used in the transport plan
NUM_TEST_SAMPLES = 2500  # number of test samples
RESIZE_USPS = True  # if True, resizes usps images to 28*28
REGULARIZER = 0.01  # Regularizer for entropic OT problem. If set to None, computes unregularized plan
TEMPERATURE = 100  # temperature for the TransportNN plug-in estimations of OT maps
FEATURE_METHOD = "plain_softmax"  # one of plain_softmax, plugin
LABEL_METHOD = "plain_softmax"  # one of plain_softmax, masked_softmax, plugin
FEATURE_DIST = True  # whether to use feature distances for label transport
LABEL_DIST = True  # whether to use label distances for label transport
OTDD_COST = (
    True  # if True, the cost between source_datasets takes label distances into account
)
LABEL_DIST_COEFF = (
    1  # controls how much the label distances contribute to the overall transport
)
# cost; 1 means both label and feature dists contribute equally
PROJECT_LABELS = False  # if True, projects label predictions on source dataset space before transport

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
        ).to(device)
        resized_train_target_labels = torch.tensor(
            [resized_train_target_dataset[i][1] for i in range(NUM_SAMPLES)]
        ).to(device)
        resized_train_target_dataset = CustomDataset(
            resized_train_target_features, resized_train_target_labels
        ).to(device)


else:
    raise ValueError("`TEST_DATASET` must be one of `mnist`, `fashion`, or `usps`.")

train_source_features = torch.cat(
    [train_source_dataset[i][0] for i in range(NUM_SAMPLES)], dim=0
).to(device)
train_source_labels = torch.tensor(
    [train_source_dataset[i][1] for i in range(NUM_SAMPLES)]
).to(device)
train_target_features = torch.cat(
    [train_target_dataset[i][0] for i in range(NUM_SAMPLES)], dim=0
).to(device)
train_target_labels = torch.tensor(
    [train_target_dataset[i][1] for i in range(NUM_SAMPLES)]
).to(device)
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
model = SimpleCNN(dim=input_dim, location=location).to(device)

try:
    acc = test_accuracy(model, test_loader)
    print(f"{TRAIN_DATASET} model accuracy on test set {TEST_DATASET}: {acc}")
except Exception as e:
    print(f"Cannot test model on target dataset; dimensions do not match. Error: {e}")

mu = torch.ones(train_source_dataset.num_samples) / train_source_dataset.num_samples
mu = mu.to(device)
nu = torch.ones(train_target_dataset.num_samples) / train_target_dataset.num_samples
nu = nu.to(device)
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
    ).to(device)
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
    ).to(device)
cost /= cost.max()

if OTDD_COST:
    label_distances = compute_label_distances(
        train_source_dataset, train_target_dataset
    )

    # this is the W^2_2 based label distances, based on 20 samples per label
    """
    label_distances = torch.tensor([[0.4397, 0.6722, 0.6446, 0.6543, 0.5606, 0.5232, 0.6182, 0.5966, 0.6271,
         0.5339],
        [0.5722, 0.6186, 0.6841, 0.6837, 0.6842, 0.6085, 0.5901, 0.5812, 0.6505,
         0.5052],
        [0.4273, 0.6231, 0.4612, 0.5412, 0.5040, 0.4955, 0.5295, 0.5376, 0.5716,
         0.4720],
        [0.5266, 0.5687, 0.5596, 0.5466, 0.5833, 0.5278, 0.5743, 0.5746, 0.5594,
         0.4954],
        [0.5709, 0.6224, 0.6137, 0.6841, 0.5046, 0.5969, 0.6306, 0.6276, 0.6607,
         0.5068],
        [0.5314, 0.5892, 0.5249, 0.5809, 0.5365, 0.5090, 0.5420, 0.5908, 0.5578,
         0.5222],
        [0.5258, 0.6741, 0.5906, 0.6438, 0.6099, 0.5473, 0.5557, 0.6050, 0.6250,
         0.5174],
        [0.5447, 0.6120, 0.5972, 0.6226, 0.5472, 0.5419, 0.6126, 0.5298, 0.5968,
         0.4965],
        [0.5255, 0.6821, 0.6314, 0.6708, 0.6354, 0.5642, 0.5894, 0.5801, 0.5902,
         0.5176],
        [0.5459, 0.6540, 0.6173, 0.6887, 0.5877, 0.5676, 0.5948, 0.5578, 0.6029,
         0.4484]])
    """
    label_distances /= label_distances.max()
    expanded_label_distances = label_distances[
        train_source_dataset.labels.squeeze(), :
    ][:, train_target_dataset.labels.squeeze()]
    cost += LABEL_DIST_COEFF * expanded_label_distances
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

x, labels = next(iter(test_loader))
x = x[:16].to(device)
labels = labels[:16].to(device)
x = x.squeeze(1)
x = x.view(x.shape[0], -1)
self = transportNN
x_s = [
    self.transport_features(x, dataset, aligned_dataset, f)
    for dataset, aligned_dataset, f in zip(
        self.source_datasets, self.aligned_source_datasets, self.f
    )
][0]

transportNN.label_method = "label_correspondences"
acc = test_accuracy(transportNN, test_loader, max_samples=NUM_TEST_SAMPLES)
print(f"Plan-summed correspondences: {acc}")

l1 = torch.ones(10) / 10
l2 = torch.ones(10) / 10
l1 = l1.to(device)
l2 = l2.to(device)
L = label_distances
T0 = ot.emd(l1, l2, L).to(device)
T01 = ot.sinkhorn(l1, l2, L, 0.1, numItermax=100000).to(device)
T001 = ot.sinkhorn(l1, l2, L, 0.01, numItermax=100000).to(device)

for i, plan in enumerate([T0, T01, T001]):
    transportNN.label_correspondences = (plan,)
    acc = test_accuracy(transportNN, test_loader, max_samples=NUM_TEST_SAMPLES)
    print(f"Test accuracy with current plan {i}: {acc}")
