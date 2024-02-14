import torch.nn as nn
import torch.optim as optim

from synthdatasets import *
from train import *
from fusion import *


#gmms = InterpolGMMs(
#    gmm_kwargs1={'N':1000, 'mus':torch.tensor([[-3.,-3.],[-3.,3.]]),'lambdas':torch.tensor([2.,10.])},
#    gmm_kwargs2={'l':3,'N':1000,'mus':torch.tensor([[1.,4.],[0.,0.],[0.,-3.]])}
#)

gmms = InterpolGMMs(
    nb_interpol=3,
    gmm_kwargs1={'l':3, 'N':2000, 'mus':5*torch.tensor([[0., 0.],[5., 1.], [3., 4.]]),'lambdas':torch.tensor([2.,10., 20.])},
    gmm_kwargs2={'l':4,'N':2000,'mus':5*torch.tensor([[0., 2.],[1., 1.],[2., 2.], [4., 4.]]), 'lambdas': torch.tensor([10., 2., 2., 2.])},
    low_dim_labels=True
)

#gmms = InterpolGMMs(
#    nb_interpol=9,
#    gmm_kwargs1={'l':4, 'N':2000, 'mus':torch.tensor([[2., 0.],[17., 2.],[-2., 23.], [17., 17.]])},
#    gmm_kwargs2={'l':4,'N':2000,'mus':torch.tensor([[-5., -15.],[14., 7.], [0., 40.], [30., 30.]])[torch.tensor([2,3,0,1])]},
#    low_dim_labels=False,
#    hard_labels=True
#)

HIDDEN_SIZE = 16
NUM_HIDDEN_LAYERS = 2
BIAS = True
NUM_DATASETS = len(gmms.datasets)
D_IN = 2
D_OUT = sum([gmm.l for gmm in gmms.gmms])
ITERS = 2

# for each dataset, contains a list with 2-tuples of (model,weight)
data = {j: {'models': [], 'weights': []} for j in range(NUM_DATASETS)}

padded_data = {j: {'models': [], 'weights': []} for j in range(NUM_DATASETS)}

for i in range(ITERS):
    models = [
                 SimpleNN([D_IN] + [HIDDEN_SIZE] * NUM_HIDDEN_LAYERS + [gmms.datasets[0].label_dim], temperature=1, bias=BIAS)
             ] + [
        SimpleNN([D_IN] + [HIDDEN_SIZE] * NUM_HIDDEN_LAYERS + [D_OUT], temperature=1, bias=BIAS) for _ in range(NUM_DATASETS - 2)
    ] + [
        SimpleNN([D_IN] + [HIDDEN_SIZE] * NUM_HIDDEN_LAYERS + [gmms.datasets[-1].label_dim], temperature=1, bias=BIAS)
    ]
    for j in range(NUM_DATASETS):
        train(
            models[j],
            gmms.datasets[j],
            num_epochs=100,
            loss_fn=nn.MSELoss(),
            opt=optim.Adam,
            lr=0.005
        )
        weights = models[j].get_weight_tensor()
        if j == 0:
            padded_model = pad_weights(models[j], [D_IN] + [HIDDEN_SIZE] * NUM_HIDDEN_LAYERS + [D_OUT],
                                       pad_from='bottom')
        elif j == NUM_DATASETS - 1:
            padded_model = pad_weights(models[j], [D_IN] + [HIDDEN_SIZE] * NUM_HIDDEN_LAYERS + [D_OUT],
                                       pad_from='top')
        else:
            padded_model = models[j]
        padded_weights = padded_model.get_weight_tensor()
        data[j]['models'].append(models[j])
        data[j]['weights'].append(weights[None, :])
        padded_data[j]['models'].append(padded_model)
        padded_data[j]['weights'].append(padded_weights[None, :])
for key, val in data.items():
    val['weights'] = torch.cat(val['weights'])
for key, val in padded_data.items():
    val['weights'] = torch.cat(val['weights'])

print(f'Parameter counts: {[data[i]["models"][0].par_number for i in range(NUM_DATASETS)]}')

mergedNNs = [
    MergeNN(data[0]['models'][i], data[2]['models'][i], gmms.plan, gmms.datasets[0], gmms.datasets[2], gmms.datasets[1])
    for i in range(ITERS)
]

print('Model 1 on dataset 1:')
accs = []
for i in range(ITERS):
    accs.append(get_accuracy(data[0]['models'][i], gmms.datasets[0]))
avg = sum(accs) / len(accs)
print(f'Accuracies: {accs}, avg. accuracy: {avg}\n')

print('Model 2 on dataset 2:')
accs = []
for i in range(ITERS):
    accs.append(get_accuracy(data[1]['models'][i], gmms.datasets[1]))
avg = sum(accs) / len(accs)
print(f'Accuracies: {accs}, avg. accuracy: {avg}\n')

print('Model 3 on dataset 3:')
accs = []
for i in range(ITERS):
    accs.append(get_accuracy(data[2]['models'][i], gmms.datasets[2]))
avg = sum(accs) / len(accs)
print(f'Accuracies: {accs}, avg. accuracy: {avg}\n')

print('Weight average of 1 and 3 on 2:')
accs = []
for i in range(ITERS):
    weights = (padded_data[0]['weights'][i]+padded_data[2]['weights'][i]) / 2
    model = SimpleNN([D_IN] + [HIDDEN_SIZE] * NUM_HIDDEN_LAYERS + [D_OUT], temperature=1, bias=BIAS, weights=weights)
    accs.append(get_accuracy(model, gmms.datasets[1]))
avg = sum(accs) / len(accs)
print(f'Accuracies: {accs}, avg. accuracy: {avg}\n')

print('Fused(1,3) on 2:')
accs = []
for i in range(ITERS):
    fused_model = fuse_models(data[0]['models'][i], padded_data[2]['models'][i], delta=0.5)
    accs.append(get_accuracy(fused_model, gmms.datasets[1]))
avg = sum(accs) / len(accs)
print(f'Accuracies: {accs}, avg. accuracy: {avg}\n')


print('MergedNN on 2:')
accs = []
for mergedNN in mergedNNs:
    accs.append(get_accuracy(mergedNN, gmms.datasets[1]))
avg = sum(accs) / len(accs)
print(f'Accuracies: {accs}, avg. accuracy: {avg}\n')

print('Model 1 on dataset 2:')
accs = []
for i in range(ITERS):
    accs.append(get_accuracy(padded_data[0]['models'][i], gmms.datasets[1]))
avg = sum(accs) / len(accs)
print(f'Accuracies: {accs}, avg. accuracy: {avg}\n')

print('Model 1 aligned to 2:')
accs = []
for i in range(ITERS):
    fused_model = fuse_models(data[0]['models'][i], data[1]['models'][i], delta=1)
    accs.append(get_accuracy(fused_model, gmms.datasets[1]))
avg = sum(accs) / len(accs)
print(f'Accuracies: {accs}, avg. accuracy: {avg}\n')

print('Model 3 on dataset 2:')
accs = []
for i in range(ITERS):
    accs.append(get_accuracy(padded_data[2]['models'][i], gmms.datasets[1]))
avg = sum(accs) / len(accs)
print(f'Accuracies: {accs}, avg. accuracy: {avg}\n')

print('Model 3 aligned to 2:')
accs = []
for i in range(ITERS):
    fused_model = fuse_models(data[2]['models'][i], data[1]['models'][i], delta=1)
    accs.append(get_accuracy(fused_model, gmms.datasets[1]))
avg = sum(accs) / len(accs)
print(f'Accuracies: {accs}, avg. accuracy: {avg}\n')

"""
print('Model 1 aligned to the previous 1:')
accs = []
for i in range(ITERS):
    fused_model = fuse_models(data[0]['models'][i], data[0]['models'][i-1], delta=1)
    accs.append(get_accuracy(fused_model, gmms.datasets[0]))
avg = sum(accs) / len(accs)
print(f'Accuracies: {accs}, avg. accuracy: {avg}\n')


print('----------------------------------------------------------------\n')
print('----------------------------------------------------------------\n\n')
print('Average of pairwise distances between models across datasets\n')
pairwise_distances = torch.tensor(
    [
        [torch.cdist(padded_data[i]['weights'], padded_data[j]['weights']).mean() if i != j else
         torch.cdist(padded_data[i]['weights'], padded_data[j]['weights'])[~torch.eye(len(data[j]['weights']), dtype=bool)].mean()
         for i in range(NUM_DATASETS)]
        for j in range(NUM_DATASETS)
    ]
)
print(pairwise_distances)
print('Aligned:\n')
def compute_average_aligned_distance(models_A, models_B, skip_diag = False):
    '''
    Compute the average distance between `models_A` and `models_B`, where `models_A` are aligned to `models_B`.
    :param models_A: list of models.
    :param models_B: list of models, of same length as `models_A`.
    :param skip_diag: if True, skips distances on the diagonal (useful if `models_A`==`models_B`).
    :return: average pairwise aligned distance.
    '''
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
mean_weights = torch.cat([padded_data[j]['weights'].mean(dim=0)[None, :] for j in range(NUM_DATASETS)])
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
    model = SimpleNN([D_IN] + [HIDDEN_SIZE] * NUM_HIDDEN_LAYERS + [D_OUT], bias=BIAS)
    accs.append(get_accuracy(model, gmms.datasets[1]))
avg = sum(accs) / len(accs)
print(f'Accuracies: {accs}, avg. accuracy: {avg}\n')
print('----------------------------------------------------------------\n')
print('Accuracy for average of model 1 and model 3 on dataset 2')
accs = []
for i in range(5):
    weights = ((padded_data[0]['weights'][i] + padded_data[2]['weights'][i]) / 2).detach().clone()
    model = SimpleNN([D_IN] + [HIDDEN_SIZE] * NUM_HIDDEN_LAYERS + [D_OUT], weights=weights, bias=BIAS)
    accs.append(get_accuracy(model, gmms.datasets[1]))
avg = sum(accs) / len(accs)
print(f'Accuracies: {accs}, avg. accuracy: {avg}\n')

#print('Average layer norm for model 1:')
#print(torch.norm(torch.sum(torch.stack([data[0]['models'][i].layers[0][0].weight for i in range(5)]), dim=0), dim=1) / 5)

#print('Average layer norm for model 4:')
#print(torch.norm(torch.sum(torch.stack([data[3]['models'][i].layers[0][0].weight for i in range(5)]), dim=0), dim=1) / 5)

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
    accs.append(get_accuracy(padded_data[0]['models'][i], gmms.datasets[1]))
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
    fused_model = fuse_models(data[0]['models'][i], data[2]['models'][i], reg=0.01)
    accs.append(get_accuracy(fused_model, gmms.datasets[1]))
avg = sum(accs) / len(accs)
print(f'Accuracies: {accs}, avg. accuracy: {avg}\n')
"""

