import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from torch_geometric.nn import GATConv


class PRGAT2(torch.nn.Module):
    def __init__(self, num_features, n_classes, num_heads, num_rels, num_bases, num_hidden, num_hidden_layer_pairs, dropout, activation, alpha, bias=True):
        super(PRGAT2, self).__init__()
        self.neg_slope = alpha
        self.num_hidden_layer_pairs = num_hidden_layer_pairs
        # dropout
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = nn.Dropout(p=0.)
        # activation
        self.activation = activation

        if num_bases < 0:
            num_bases = num_rels

        self.layers = nn.ModuleList()
        for num_layer in range(num_hidden_layer_pairs):
            # RGCN
            if num_layer == 0:
                self.layers.append(RGCNConv(num_features,         num_hidden, num_rels, num_bases, bias=bias))
            else:
                self.layers.append(RGCNConv(num_hidden*num_heads, num_hidden, num_rels, num_bases, bias=bias))
            # GAT
            if num_layer == num_hidden_layer_pairs-1:
               self.layers.append(GATConv(num_hidden, n_classes, heads=num_heads, concat=False, negative_slope=self.neg_slope, dropout=dropout, bias=bias))
            else:
                self.layers.append(GATConv(num_hidden, num_hidden, heads=num_heads, concat=True, negative_slope=self.neg_slope, dropout=0, bias=bias))


    def forward(self, data):
        x, edge_index, edge_type = data.x, data.edge_index, data.edge_type
        for i in range(self.num_hidden_layer_pairs*2):
            # Layer
            if i%2==0:
                x = self.layers[i](x, edge_index, edge_type)
            else:
                x = self.layers[i](x, edge_index)
            #  Dropout & activation for hidden layers (not output)
            if i != self.num_hidden_layer_pairs*2-1:
                x = self.activation(x)
                x = self.dropout(x)
        x = x.mean(1).unsqueeze(dim=1)
        return torch.tanh(x)
