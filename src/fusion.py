import ot
import numpy as np
import torch

def compute_plan(A, B, reg = None):
    """
    Compute a transport plan between two uniform histograms, where the cost function is computed from the input matrices A and B.
    This corresponds to a transport plan between the layers of two neural networks, where A and B correspond to the weight matrices.
    :param A: weight matrix for first neural network, (n,d)-dimensional matrix.
    :param B: weight matrix for second neural network, (m,d)-dimensional matrix.
    :param reg: if passed, the entropic OT problem is solved instead, with regularizer `reg`.
    :return: transport plan, (n,m)-dimensional matrix.
    """
    n, m = A.shape[0], B.shape[0]
    C = np.zeros((n, m))

    # Calculate Euclidean distance for each pair of rows
    for i in range(n):
        for j in range(m):
            C[i, j] = np.linalg.norm(A[i] - B[j])

    mu = np.ones(n) / n
    nu = np.ones(m) / m
    if reg is None:
        T = ot.emd(mu, nu, C)
    else:
        T = ot.sinkhorn(mu, nu, C, reg)
    return T

def fuse_multiple_models(
        models_A,
        models_B,
        nnClass,
        reg = None,
        delta = 0.5
):
    """
    Wraps the `fuse_models` function for multiple models. Returns a list of models, where the i^th entry corresponds to the fused model
    of the i^th entry of `models_A` with the i^th entry of `models_B`.
    """
    assert len(models_A) == len(models_B), "both input lists must have the same length"
    fused_models = [
        fuse_models(
            model_A,
            model_B,
            nnClass,
            reg,
            delta
        )
        for model_A, model_B in zip(models_A, models_B)
    ]
    return fused_models

def aligned_distance(model_A, model_B):
    """
    Aligns `model_A` w.r.t. to `model_B`, then returns the distance between the aligned models.
    """
    aligned_A = fuse_models(model_A, model_B, type(model_A), delta=1)
    distance = torch.norm(aligned_A.get_weight_tensor() - model_B.get_weight_tensor())
    return distance

def fuse_models(
        model_A,
        model_B,
        nnClass,
        reg = None,
        delta = 0.5
):
    """
    Fuse two models together.
    :param model_A: fully connected `nn.Module` neural network with the same number of layers as `model_B`.
    :param model_B: fully connected `nn.Module` neural network with the same number of layers as `model_A`. Hidden layer sizes may differ from those of `model_A`.
    :param nnClass: neural network class whose instances `model_A` and `model_B` are. Must accept a list of layer dimensions as first argument, and an optional
        one-dimensional vector containing network weights as second argument.
    :param reg: if passed, the entropic OT problem is solved instead the regular one for each layer, with regularizer `reg`.
    :param delta: controls what convex combination to return. Returns `delta`*`model_A`+(1-`delta`)*`model_B` (note, if `delta`=1, this returns a permuted version of `model_A`.
    :return: fused model. Of same size as `model_B`.
    """
    sizes_A = [param.shape for param in model_A.parameters()]
    params_A = list(model_A.parameters())
    sizes_B = [param.shape for param in model_B.parameters()]
    params_B = list(model_B.parameters())
    num_layers = len(sizes_A) // 2 #number of hidden+output layers; not counting input layer

    # test that the models are of compatible shape and type
    assert isinstance(model_A, nnClass), "model A is of the wrong class"
    assert isinstance(model_B, nnClass), "model B is of the wrong class"
    assert len(sizes_A) == len(sizes_B), "models do not have the same number of layers"
    assert sizes_A[0][1] == sizes_B[0][1], "models do not have the same input dimension"
    assert sizes_A[-1][-1] == sizes_B[-1][-1], "models do not have the same output dimension"
    for i in range(num_layers):
        assert len(sizes_A[2*i]) == len(sizes_B[2*i]) == 2, "expected linear layer, but found wrong parameter shape"
        assert len(sizes_A[2*i + 1]) == len(sizes_B[2*i + 1]) == 1, "expected bias layer, but found wrong parameter shape"
        assert sizes_A[2*i][0] == sizes_A[2*i + 1][0], "bias layer of model A does not match size of linear layer"
        assert sizes_B[2*i][0] == sizes_B[2*i + 1][0], "bias layer of model B does not match size of linear layer"

    layer_sizes_A = [sizes_A[0][1]]+[sizes_A[2*i+1][0] for i in range(num_layers)] #sizes of all layers including input and output layers; e.g. [5, 10, 2]
    layer_sizes_B = [sizes_B[0][1]] + [sizes_B[2 * i + 1][0] for i in range(num_layers)]
    input_dim = layer_sizes_A[0]
    fused_params = []
    T = np.eye(input_dim) / input_dim
    for l in range(num_layers):
        w_hat = np.matmul(params_A[2 * l].detach().numpy(), T) * T.shape[1]
        w_hat = np.concatenate((w_hat, params_A[2 * l + 1].detach().numpy()[None, :].T), axis=1) # add bias layer
        weights_B = np.concatenate((params_B[2 * l].detach().numpy(), params_B[2 * l + 1].detach().numpy()[None, :].T), axis=1)
        T = compute_plan(w_hat, weights_B, reg)
        w_tilde = T.shape[1] * np.matmul(T.T, w_hat)
        fused_weights = delta * w_tilde + (1 - delta) * weights_B
        fused_layer = fused_weights[:, :-1]
        fused_bias = fused_weights[:, -1:].flatten()
        fused_params += [torch.tensor(fused_layer).flatten(), torch.tensor(fused_bias).flatten()]
    return nnClass(layer_sizes_B, torch.tensor(np.concatenate(fused_params)).detach().to(params_B[0].dtype))