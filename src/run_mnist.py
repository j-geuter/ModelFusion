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
    "usps"  # `mnist`, `usps`, or `fashion` The dataset on which we want to test
)
BATCH_SIZE = 64
NUM_SOURCE_SAMPLES = 7000  # number of training samples used in the transport plan
NUM_LABELED_TARGET_SAMPLES = 100
NUM_UNLABELED_TARGET_SAMPLES = 1000
NUM_TEST_SAMPLES = 2500  # number of test samples
RESIZE_USPS = True  # if True, resizes usps images to 28*28
REGULARIZER = 0.007  # Regularizer for entropic OT problem. If set to None, computes unregularized plan
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
        resized_labeled_train_target_features = torch.cat(
            [resized_train_target_dataset[i][0] for i in range(NUM_LABELED_TARGET_SAMPLES)], dim=0
        ).to(device)
        if NUM_UNLABELED_TARGET_SAMPLES > 0:
            resized_unlabeled_train_target_features = torch.cat(
                [resized_train_target_dataset[NUM_LABELED_TARGET_SAMPLES + i][0] for i in range(NUM_UNLABELED_TARGET_SAMPLES)], dim=0
            ).to(device)
            resized_unlabeled_train_target_dataset = CustomDataset(
                resized_unlabeled_train_target_features, None
            ).to(device)
        else:
            resized_unlabeled_train_target_dataset = None
        resized_train_target_labels = torch.tensor(
            [resized_train_target_dataset[i][1] for i in range(NUM_LABELED_TARGET_SAMPLES)]
        ).to(device)
        resized_labeled_train_target_dataset = CustomDataset(
            resized_labeled_train_target_features, resized_train_target_labels
        ).to(device)



else:
    raise ValueError("`TEST_DATASET` must be one of `mnist`, `fashion`, or `usps`.")

train_source_features = torch.cat(
    [train_source_dataset[i][0] for i in range(NUM_SOURCE_SAMPLES)], dim=0
).to(device)
train_source_labels = torch.tensor(
    [train_source_dataset[i][1] for i in range(NUM_SOURCE_SAMPLES)]
).to(device)
train_labeled_target_features = torch.cat(
    [train_target_dataset[i][0] for i in range(NUM_LABELED_TARGET_SAMPLES)], dim=0
).to(device)
if NUM_UNLABELED_TARGET_SAMPLES > 0:
    train_unlabeled_target_features = torch.cat(
        [train_target_dataset[NUM_LABELED_TARGET_SAMPLES + i][0] for i in range(NUM_UNLABELED_TARGET_SAMPLES)], dim=0
    ).to(device)
train_target_labels = torch.tensor(
    [train_target_dataset[i][1] for i in range(NUM_LABELED_TARGET_SAMPLES)]
).to(device)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

train_source_dataset = CustomDataset(train_source_features, train_source_labels)
train_labeled_target_dataset = CustomDataset(train_labeled_target_features, train_target_labels)
if NUM_UNLABELED_TARGET_SAMPLES > 0:
    train_unlabeled_target_dataset = CustomDataset(train_unlabeled_target_features, None)
else:
    train_unlabeled_target_dataset = None
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
except Exception as e: # if dimensions do not match
    print(f"Cannot test model on target dataset. Error: {e}")

mu = torch.ones(train_source_dataset.num_samples) / train_source_dataset.num_samples
mu = mu.to(device)
if NUM_UNLABELED_TARGET_SAMPLES > 0:
    num_target_samples = train_labeled_target_dataset.num_samples + train_unlabeled_target_dataset.num_samples
else:
    num_target_samples = train_labeled_target_dataset.num_samples
nu = torch.ones(num_target_samples) / num_target_samples
nu = nu.to(device)
if TEST_DATASET == "usps" and not RESIZE_USPS:
    if NUM_UNLABELED_TARGET_SAMPLES > 0:
        cost = (
            torch.cdist(
                train_source_dataset.features.view(
                    train_source_dataset.features.shape[0], -1
                ),
                torch.cat((
                resized_labeled_train_target_features.view(
                        resized_labeled_train_target_features.shape[0], -1
                ),
                resized_unlabeled_train_target_features.view(
                    resized_unlabeled_train_target_features.shape[0], -1
                )), 0
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
            resized_labeled_train_target_features.view(
                resized_labeled_train_target_features.shape[0], -1
            )
        ) ** 2).to(device)
else:
    if NUM_UNLABELED_TARGET_SAMPLES > 0:
        cost = (
            torch.cdist(
                train_source_dataset.features.view(
                    train_source_dataset.features.shape[0], -1
                ),
                torch.cat((
                train_labeled_target_dataset.features.view(
                        train_labeled_target_dataset.features.shape[0], -1
                ),
                train_unlabeled_target_dataset.features.view(
                    train_unlabeled_target_dataset.features.shape[0], -1
                )
                ), 0)
                ,
            )
            ** 2
        ).to(device)
    else:
        cost = (
            torch.cdist(
                train_source_dataset.features.view(
                    train_source_dataset.features.shape[0], -1
                ),
                train_labeled_target_dataset.features.view(
                    train_labeled_target_dataset.features.shape[0], -1
                )
            ) ** 2
        ).to(device)
cost /= cost.max()

if OTDD_COST:
    label_distances = compute_label_distances(
        train_source_dataset, train_labeled_target_dataset
    )

    # this is the W^2_2 based label distances, based on 20 samples per label, for MNIST->USPS
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
    """
    # this is the same tensor, but for MNIST->Fashion
    label_distances = torch.tensor([[0.6046, 0.5752, 0.3677, 0.5446, 0.4667, 0.4578, 0.4563, 0.6986, 0.3641,
         0.5709],
        [0.7222, 0.6090, 0.4836, 0.5263, 0.6586, 0.5276, 0.6554, 0.7106, 0.6796,
         0.6611],
        [0.5572, 0.5513, 0.3131, 0.4850, 0.4137, 0.3943, 0.4382, 0.4818, 0.4192,
         0.5025],
        [0.5478, 0.5588, 0.4151, 0.4339, 0.4865, 0.4452, 0.4656, 0.6910, 0.4754,
         0.4984],
        [0.6645, 0.6021, 0.4689, 0.5828, 0.5258, 0.4644, 0.5528, 0.5980, 0.4342,
         0.5738],
        [0.5491, 0.5437, 0.4337, 0.4605, 0.4724, 0.4416, 0.4799, 0.5117, 0.4373,
         0.5142],
        [0.6817, 0.6460, 0.4174, 0.5548, 0.4912, 0.5412, 0.5610, 0.7437, 0.5856,
         0.6697],
        [0.6397, 0.6105, 0.4466, 0.5588, 0.5731, 0.4956, 0.5439, 0.5975, 0.5056,
         0.6173],
        [0.6560, 0.5483, 0.4220, 0.5091, 0.5458, 0.5307, 0.5146, 0.7532, 0.5808,
         0.6492],
        [0.6879, 0.6817, 0.4364, 0.6013, 0.6140, 0.4806, 0.5961, 0.6090, 0.5472,
         0.5944]])
    """
    label_distances /= label_distances.max()
    expanded_label_distances = label_distances[
        train_source_dataset.labels.squeeze(), :
    ][:, train_labeled_target_dataset.labels.squeeze()]
    cost[:, :NUM_LABELED_TARGET_SAMPLES] += LABEL_DIST_COEFF * expanded_label_distances
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
    train_labeled_target_dataset,
    train_unlabeled_target_dataset,
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

transportNN.direct_interpolation = True
acc = test_accuracy(transportNN, test_loader, max_samples=NUM_TEST_SAMPLES)
print(f"Direct interpolation: {acc}")

transportNN.direct_interpolation = False

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
