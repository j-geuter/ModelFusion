import torch


def models_equal(model1, model2, tolerance=1e-5):
    """
    Checks if two models `model1` and `model2` are identical, up to tolerance `tolerance`.
    :return: True if they are identical, False otherwise.
    """
    # Check if the models have the same architecture (same layers and shapes)
    assert model1.__class__ == model2.__class__, "Model classes do not match"

    # Check if the models have the same number of parameters
    assert sum(p.numel() for p in model1.parameters()) == sum(
        p.numel() for p in model2.parameters()
    ), "Parameter count does not match"

    # Check if the parameters are identical up to the tolerance
    for param1, param2 in zip(model1.parameters(), model2.parameters()):
        assert param1.shape == param2.shape, "Parameter shape does not match"

        if torch.any(torch.abs(param1 - param2) > tolerance):
            return False

    return True


def weights_equal(weights_1, weights_2, tol=1e-5):
    """
    Checks if two weight vectors are identical, up to tolerance `tol`.
    :return: True if they are identical, False otherwise.
    """
    assert weights_1.shape == weights_2.shape, "Parameter shape does not match"

    if torch.any(torch.abs(weights_1 - weights_2) > tol):
        return False
    return True
