import unittest
import random
from utils import models_equal, weights_equal
from fusion import *
from models import *

def permute_neurons(net):
    """
    Permutes all hidden neurons in a network.
    :param net: `SimpleNN` instance.
    :return: network with permuted neurons.
    """
    weights = torch.tensor([])
    perm = torch.tensor([i for i in range(net.layer_sizes[0])]) # identity permutation for first iteration
    for layer in net.layers:
        out_features = layer[0].out_features
        perm_layer = layer[0].weight[:, perm] # permute layer along second axis according to previous permutation
        perm = torch.randperm(out_features)
        perm_layer = perm_layer[perm] # permute layer along first axis according to current permutation
        weights = torch.cat((weights, perm_layer.view(-1)))
        if net.bias:
            perm_bias = layer[0].bias[perm]
            weights = torch.cat((weights, perm_bias))
    perm_net = SimpleNN(net.layer_sizes, weights=weights, bias=net.bias)
    return perm_net

class TestStringMethods(unittest.TestCase):

    def test_fusion(self):

        # Test networks of some pre-defined size
        for net_size in [[2,5], [1,1], [3,3,3], [2,10,10,5], [2, 2, 2], [10, 2, 10]]:
            for bias in [True, False]:
                net = SimpleNN(net_size, bias=bias)
                perm_net = permute_neurons(net)
                scaled_net = SimpleNN(net_size, bias=bias, weights=2*net.get_weight_tensor())

                # Check that aligning the permuted net to the original yields the original, and vice versa
                self.assertTrue(models_equal(fuse_models(perm_net, net, delta=1), net))
                self.assertTrue(models_equal(fuse_models(net, perm_net, delta=1), perm_net))

                # Check that fusing the net with itself does not change it
                self.assertTrue(models_equal(fuse_models(net, net), net))

                # Check that fusing the net with its scaled version yields a scaled in-between version
                self.assertTrue(
                    weights_equal(1.5 * net.get_weight_tensor(), fuse_models(net, scaled_net, delta=0.5).get_weight_tensor())
                )



        def generate_random_list():
            return [random.randint(2, 50) for _ in range(random.randint(2, 5))]

        # As above, but test on 20 networks of random size
        for net_size in [generate_random_list() for _ in range(20)]:
            for bias in [True, False]:
                net = SimpleNN(net_size, bias=bias)
                perm_net = permute_neurons(net)
                scaled_net = SimpleNN(net_size, bias=bias, weights=2 * net.get_weight_tensor())

                # Check that aligning the permuted net to the original yields the original, and vice versa
                self.assertTrue(models_equal(fuse_models(perm_net, net, delta=1), net))
                self.assertTrue(models_equal(fuse_models(net, perm_net, delta=1), perm_net))

                # Check that fusing the net with itself does not change it
                self.assertTrue(models_equal(fuse_models(net, net), net))

                # Check that fusing the net with its scaled version yields a scaled in-between version
                self.assertTrue(
                    weights_equal(1.5 * net.get_weight_tensor(), fuse_models(net, scaled_net, delta=0.5).get_weight_tensor())
                )

if __name__ == '__main__':
    unittest.main()