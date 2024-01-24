from synthdatasets import *
from train import *
from utils import *
from fusion import *

gmms = InterpolGMMs(
    gmm_kwargs1={'N':1000, 'mus':torch.tensor([[-3.,-3.],[-3.,3.]]),'lambdas':torch.tensor([2.,10.])},
    gmm_kwargs2={'l':3,'N':1000,'mus':torch.tensor([[1.,4.],[0.,0.],[0.,-3.]])}
)

HIDDEN_SIZE = 100
NUM_HIDDEN_LAYERS = 0
BIAS = False
NUM_DATASETS = len(gmms.datasets)

# for each dataset, contains a list with 2-tuples of (model,weight)
data = {j: {'models': [], 'weights': []} for j in range(NUM_DATASETS)}

for i in range(5):
    models = [SimpleNN([2] + [HIDDEN_SIZE] * NUM_HIDDEN_LAYERS + [5], temperature=1, bias=BIAS) for _ in range(NUM_DATASETS)]
    for j in range(NUM_DATASETS):
        train(
            models[j],
            gmms.datasets[j],
            num_epochs=100,
            loss_fn=nn.MSELoss(),
            opt=optim.Adam,
            lr=0.1
        )
        weights = models[j].get_weight_tensor()
        data[j]['models'].append(models[j])
        data[j]['weights'].append(weights[None, :])
for key, val in data.items():
    val['weights'] = torch.cat(val['weights'])
print('----------------------------------------------------------------\n')
print(f'Parameter count: {data[0]["models"][0].par_number}')
print('----------------------------------------------------------------\n')
print('----------------------------------------------------------------\n\n')
print('Average of pairwise distances between models across datasets\n')
pairwise_distances = torch.tensor(
    [
        [torch.cdist(data[i]['weights'], data[j]['weights']).mean() if i != j else
         torch.cdist(data[i]['weights'], data[j]['weights'])[~torch.eye(len(data[j]['weights']), dtype=bool)].mean()
         for i in range(NUM_DATASETS)]
        for j in range(NUM_DATASETS)
    ]
)
print(pairwise_distances)
print('Aligned:\n')
def compute_average_aligned_distance(models_A, models_B, skip_diag = False):
    """
    Compute the average distance between `models_A` and `models_B`, where `models_A` are aligned to `models_B`.
    :param models_A: list of models.
    :param models_B: list of models, of same length as `models_A`.
    :param skip_diag: if True, skips distances on the diagonal (useful if `models_A`==`models_B`).
    :return: average pairwise aligned distance.
    """
    assert len(models_A) == len(models_B), "both input lists must have the same length"
    num_models = len(models_A)
    distance_sum = 0
    for i in range(num_models):
        for j in range(num_models):
            if skip_diag:
                if i == j:
                    continue
            distance_sum += aligned_distance(models_A[i], models_B[j])
    if skip_diag:
        distance_sum /= (num_models**2 - num_models)
    else:
        distance_sum /= num_models**2
    return distance_sum

pairwise_aligned_distances = torch.tensor(
    [
        [compute_average_aligned_distance(data[i]['models'], data[j]['models'], i == j)
         for i in range(NUM_DATASETS)]
        for j in range(NUM_DATASETS)
    ]
)
print(pairwise_aligned_distances)
print('----------------------------------------------------------------\n')
print('----------------------------------------------------------------\n\n')
print('Pairwise distances between avg. models for each dataset\n')
print('----------------------------------------------------------------\n')
mean_weights = torch.cat([data[j]['weights'].mean(dim=0)[None, :] for j in range(NUM_DATASETS)])
pairwise_distances = torch.cdist(mean_weights, mean_weights)
print(pairwise_distances) # Print pairwise distance matrix

print('----------------------------------------------------------------\n')
print('----------------------------------------------------------------\n\n')
print('Accuracy for models on dataset 2')
accs = [get_accuracy(model, gmms.datasets[1]) for model in data[1]['models']]
avg = sum(accs) / len(accs)
print(f'Accuracies: {accs}, avg. accuracy: {avg}\n')
print('----------------------------------------------------------------\n')
print('Accuracy for random weight initialization on dataset 2')
accs = []
for i in range(5):
    model = SimpleNN([2] + [HIDDEN_SIZE] * NUM_HIDDEN_LAYERS + [5], bias=BIAS)
    accs.append(get_accuracy(model, gmms.datasets[1]))
avg = sum(accs) / len(accs)
print(f'Accuracies: {accs}, avg. accuracy: {avg}\n')
print('----------------------------------------------------------------\n')
print('Accuracy for average of model 1 and model 3 on dataset 2')
accs = []
for i in range(5):
    weights = ((data[0]['weights'][i] + data[2]['weights'][i]) / 2).detach().clone()
    model = SimpleNN([2] + [HIDDEN_SIZE] * NUM_HIDDEN_LAYERS + [5], weights, bias=BIAS)
    accs.append(get_accuracy(model, gmms.datasets[1]))
avg = sum(accs) / len(accs)
print(f'Accuracies: {accs}, avg. accuracy: {avg}\n')

print('Aligned:')
accs = []
for i in range(5):
    fused_model = fuse_models(data[0]['models'][i], data[2]['models'][i])
    accs.append(get_accuracy(fused_model, gmms.datasets[1]))
avg = sum(accs) / len(accs)
print(f'Accuracies: {accs}, avg. accuracy: {avg}\n')

print('Model 1 on dataset 2:')
accs = []
for i in range(5):
    accs.append(get_accuracy(data[0]['models'][i], gmms.datasets[1]))
avg = sum(accs) / len(accs)
print(f'Accuracies: {accs}, avg. accuracy: {avg}\n')

print('Model 3 on dataset 2:')
accs = []
for i in range(5):
    accs.append(get_accuracy(data[2]['models'][i], gmms.datasets[1]))
avg = sum(accs) / len(accs)
print(f'Accuracies: {accs}, avg. accuracy: {avg}\n')

print('Model 1 aligned to 2:')
accs = []
for i in range(5):
    fused_model = fuse_models(data[0]['models'][i], data[1]['models'][i], delta=1)
    accs.append(get_accuracy(fused_model, gmms.datasets[1]))
avg = sum(accs) / len(accs)
print(f'Accuracies: {accs}, avg. accuracy: {avg}\n')

print('Model 3 aligned to 2:')
accs = []
for i in range(5):
    fused_model = fuse_models(data[2]['models'][i], data[1]['models'][i], delta=1)
    accs.append(get_accuracy(fused_model, gmms.datasets[1]))
avg = sum(accs) / len(accs)
print(f'Accuracies: {accs}, avg. accuracy: {avg}\n')

print('Aligned and regularized:')
accs = []
for i in range(5):
    fused_model = fuse_models(data[0]['models'][i], data[2]['models'][i], reg=0.1)
    accs.append(get_accuracy(fused_model, gmms.datasets[1]))
avg = sum(accs) / len(accs)
print(f'Accuracies: {accs}, avg. accuracy: {avg}\n')

"""
print('----------------------------------------------------------------\n')
print('Accuracies for average of model 1 and model 4 on dataset 2')
accs = []
for i in range(5):
    weights = ((2*models_and_weights[0]['weights'][i]+models_and_weights[3]['weights'][i])/3).detach().clone()
    model = SimpleNN([2] + [HIDDEN_SIZE] * NUM_HIDDEN_LAYERS + [5], weights, bias=BIAS)
    accs.append(get_accuracy(model, gmms.datasets[1]))
avg = sum(accs) / len(accs)
print(f'Accuracies: {accs}, avg. accuracy: {avg}\n\n')

print('Aligned:')
accs = []
for i in range(5):
    fused_model = fuse_models(models_and_weights[0]['models'][i], models_and_weights[3]['models'][i], delta=2/3)
    accs.append(get_accuracy(fused_model, gmms.datasets[1]))
avg = sum(accs) / len(accs)
print(f'Accuracies: {accs}, avg. accuracy: {avg}\n')

print('Aligned and regularized:')
accs = []
for i in range(5):
    fused_model = fuse_models(models_and_weights[0]['models'][i], models_and_weights[3]['models'][i], delta=2/3, reg=0.1)
    accs.append(get_accuracy(fused_model, gmms.datasets[1]))
avg = sum(accs) / len(accs)
print(f'Accuracies: {accs}, avg. accuracy: {avg}\n')
"""

