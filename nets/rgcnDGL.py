import torch.nn as nn

from dgl.nn.pytorch.conv.relgraphconv import RelGraphConv

class RGCN(nn.Module):
    def __init__(self, g, gnn_layers, in_dim, hidden_dimensions, num_rels, activations, feat_drop, num_bases=-1):
        super(RGCN, self).__init__()
        self.g = g
        self.in_dim = in_dim
        self.hidden_dimensions = hidden_dimensions
        self.num_channels = hidden_dimensions[-1]
        self.num_rels = num_rels
        self.feat_drop = feat_drop
        self.num_bases = num_bases
        self.activations = activations
        self.gnn_layers = gnn_layers
        # create RGCN layers
        self.build_model()
        
    def set_g(self, g):
        self.g = g

    def build_model(self):
        self.layers = nn.ModuleList()
        # input to hidden
        i2h = self.build_input_layer()
        self.layers.append(i2h)
        # hidden to hidden
        for i in range(self.gnn_layers-2):
            h2h = self.build_hidden_layer(i)
            self.layers.append(h2h)
        # hidden to output
        h2o = self.build_output_layer()
        self.layers.append(h2o)


    def build_input_layer(self):
        print('Building an INPUT  layer of {}x{} (rels:{})'.format(self.in_dim, self.hidden_dimensions[0],
                                                                   self.num_rels))
        return RelGraphConv(self.in_dim, self.hidden_dimensions[0], self.num_rels, regularizer='basis',
                            dropout=self.feat_drop, num_bases=self.num_bases, activation=self.activations[0])

    def build_hidden_layer(self, i):
        print('Building an HIDDEN  layer of {}x{}'.format(self.hidden_dimensions[i], self.hidden_dimensions[i+1]))
        return RelGraphConv(self.hidden_dimensions[i], self.hidden_dimensions[i+1],  self.num_rels, regularizer='basis',
                            dropout=self.feat_drop, num_bases=self.num_bases, activation=self.activations[i+1])

    def build_output_layer(self):
        print('Building an OUTPUT  layer of {}x{}'.format(self.hidden_dimensions[-2], self.hidden_dimensions[-1]))
        return RelGraphConv(self.hidden_dimensions[-2], self.hidden_dimensions[-1], self.num_rels, regularizer='basis',
                            dropout=self.feat_drop, num_bases=self.num_bases, activation=self.activations[-1])

    def forward(self, features, etypes):
        h = features
        self.g.edata['norm'] = self.g.edata['norm'].to(device=features.device)

        for layer in self.layers:
            h = layer(self.g, h, etypes)
        return h

