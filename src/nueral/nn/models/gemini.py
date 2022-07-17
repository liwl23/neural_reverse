from ..layers import Structure2vec
from torch.nn import Module, Linear
from dgl import sum_nodes
import torch
import dgl


class Gemini(Module):
    def __init__(self, in_dim, out_dim):
        """
        An implementation of [Gemini](https://arxiv.org/abs/1708.06525).
        :param in_dim: The dimension size of node features in in_graph.
        :param out_dim: The dimension size of node features in out_graph.
        """
        super(Gemini, self).__init__()
        self.layer = Structure2vec(in_dim, out_dim)
        self.linear = Linear(out_dim, out_dim, bias=False)

    def forward(self, graph: dgl.DGLGraph, features: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation.
        :param graph: A batch of graphs which represents the CFGs of a binary functions.
        :param features: The node features of shape (num_of_nodes, in_dim).
        :return: The updated node features of shape (num_of_nodes, out_dim).
        """
        new_features = self.layer(graph, features)
        graph.ndata['x'] = new_features
        res = self.linear(sum_nodes(graph, 'x'))
        graph.ndata.pop('x')
        return res
