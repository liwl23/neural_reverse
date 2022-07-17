from torch import nn
import torch


class Dense(nn.Module):
    def __init__(self, in_dim, out_dim, layers=2, activation=torch.relu, bias=False):
        """
        A simple implementation of the multi-layer perceptron.
        :param in_dim: the size of input tensor.
        :param out_dim: the size of output tensor.
        :param layers: the layer number of perceptron.
        :param activation: the activation function.
        :param bias: whether to use bias, whether the Linear layers should use bias.
        """
        super(Dense, self).__init__()
        self.layers = torch.nn.ModuleList([nn.Linear(in_dim, out_dim, bias=bias)])
        for i in range(layers - 1):
            self.layers.append(nn.Linear(out_dim, out_dim, bias=bias))
        self.activation = activation

    def forward(self, inputs)->torch.Tensor:
        """
        Forward propagation.
        :param inputs: A tensor of shape (N_1, N_2, ..., N_k, in_dim).
        :return:  A tensor of shape (N_1, N_2, ..., N_k, out_dim).
        """
        for layer in self.layers[:-1]:
            inputs = self.activation(layer(inputs))
        return self.layers[-1](inputs)
