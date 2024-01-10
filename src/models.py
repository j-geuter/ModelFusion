import torch
import torch.nn as nn
import torch.nn.functional as F

class TemperatureScaledSoftmax(nn.Module):
    def __init__(self, temperature):
        super(TemperatureScaledSoftmax, self).__init__()
        self.temperature = temperature

    def forward(self, logits):
        scaled_logits = logits / self.temperature
        return F.softmax(scaled_logits, dim=-1)

class SimpleNN(nn.Module):
    def __init__(self, layer_sizes, weights = None, temperature = 1):
        super(SimpleNN, self).__init__()

        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(layer_sizes[l], layer_sizes[l + 1]),
                    nn.ReLU(),
                )
                for l in range(len(layer_sizes) - 2)
            ]
            +
            [
                nn.Sequential(
                nn.Linear(layer_sizes[-2], layer_sizes[-1]),
                TemperatureScaledSoftmax(temperature)
                )
            ]
        )

        self.par_number = sum(p.numel() for p in self.parameters() if p.requires_grad)

        if weights is not None:
            self.update_weights(weights)

    def forward(self, x):
        # Define the forward pass
        for layer in self.layers:
            x = layer(x)
        return x

    def get_weight_tensor(self):
        trainable_parameters = list(self.parameters())
        detached_weights = torch.cat([param.data.detach().clone().view(-1) for param in trainable_parameters])
        return detached_weights

    def update_weights(self, vector):
        assert len(vector) == self.par_number, "Weight vector size does not match parameter count!"
        vector_index = 0
        for param in self.parameters():
            # Determine the number of elements in the current parameter
            num_elements = param.data.numel()

            # Extract elements from the vector and reshape to the parameter shape
            param.data = vector[vector_index:vector_index + num_elements].detach().clone().view(param.data.shape)

            # Move to the next set of elements in the vector
            vector_index += num_elements