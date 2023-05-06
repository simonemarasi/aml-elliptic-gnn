from torch.nn import Module, Sequential, ELU
from models.custom_gat.layer import GATLayer

class GAT(Module):

    def __init__(self, num_of_layers, num_heads_per_layer, num_features_per_layer, device):
        super().__init__()
        
        num_heads_per_layer = [1] + num_heads_per_layer
        gat_layers = []

        for i in range(num_of_layers):
            layer = GATLayer(
                num_in_features = num_features_per_layer[i] * num_heads_per_layer[i], # consequence of concatenation
                num_out_features = num_features_per_layer[i+1],
                num_of_heads = num_heads_per_layer[i+1],
                concat = True if i < num_of_layers - 1 else False, # last GAT layer does average, the others do concat
                activation = ELU() if i < num_of_layers - 1 else None,
                device = device) # last layer just outputs raw scores
            gat_layers.append(layer)

        self.gat_net = Sequential(*gat_layers)

    # data is a (in_nodes_features, edge_index) tuple
    def forward(self, data):
        return self.gat_net(data)